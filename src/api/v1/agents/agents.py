import os
import traceback
from contextlib import asynccontextmanager
from typing import Literal, AsyncIterator

import cohere
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from pydantic import BaseModel

from src.api.v1.states.rag_state import RAGState
from src.api.v1.tools.tools import (
    vector_search_node,
    fts_search_node,
    hybrid_search_node,
    extract_images_node,
)
from src.api.v1.schemas.query_schema import AIResponse
from src.core.rdbm import get_sql_database

load_dotenv()


# ============================================================
# ENVIRONMENT
# ============================================================

CHECKPOINT_DB_URI = os.getenv("LANGGRAPH_CHECKPOINT_DB_URI")

if not CHECKPOINT_DB_URI:
    raise RuntimeError(
        "LANGGRAPH_CHECKPOINT_DB_URI environment variable is not configured."
    )


# ============================================================
# LLM
# ============================================================


def _get_llm():
    return ChatOpenAI(
        model=os.getenv("OPENAI_CHAT_MODEL"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )


# ============================================================
# STRUCTURED OUTPUT MODELS
# ============================================================


class RouteDecision(BaseModel):
    route: Literal[
        "VECTOR_DB",
        "HYBRID",
        "FTS",
        "RDBMS",
        "IMAGE",
    ]
    reason: str


class IntentDecision(BaseModel):
    intent: Literal[
        "CHITCHAT",
        "REPHRASER",
    ]
    reason: str


# ============================================================
# NODE 1 - ADD USER MESSAGE
# ============================================================


def add_user_message_node(state: RAGState):

    return {
        **state,
        "messages": state.get("messages", []) + [HumanMessage(content=state["query"])],
    }


# ============================================================
# NODE 2 - INTENT CHECK
# ============================================================


def intent_check_node(state: RAGState):

    print("========== INSIDE intent_check_node ==========")

    llm = _get_llm()

    structured_llm = llm.with_structured_output(IntentDecision)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are an intent classifier.

                Classify the user message into one of two categories.

                CHITCHAT:
                - hi
                - hello
                - hey
                - good morning
                - good evening
                - thanks
                - thank you
                - how are you
                - who are you
                - casual conversation

                Conversation behavior rules:

                1. Greeting handling:
                - If the user only sends a greeting such as
                  "hi", "hello", "hey", "good morning", or similar:
                - Respond politely.
                - Politely reject any questions that are outside
                  credit card information.
                - Keep the response brief.
                - Ask what the user needs help with.

                Examples:

                User: "Hi"
                Assistant: "Hi! How can I help you today?"

                User: "Good evening"
                Assistant: "Good evening! How can I assist you?"

                2. Normal conversation:
                - After the greeting exchange, answer the user's
                  questions normally.
                - Do not repeat greeting responses on every message.
                - Maintain context from previous messages.
                - Provide accurate, useful, and clear answers.
                - Ask clarification questions when the user's request
                  is unclear.

                REPHRASER:
                - credit card questions
                - policy questions
                - document questions
                - product questions
                - transaction questions
                - any information request

                Return ONLY one exact value:
                CHITCHAT
                or
                REPHRASER

                Do not shorten, modify, or create new values.
                """,
            ),
            (
                "human",
                """
                User message:
                {query}
                """,
            ),
        ]
    )

    result = (prompt | structured_llm).invoke({"query": state["query"]})

    print(
        f"[intent_check_node] " f"Intent: {result.intent}, " f"Reason: {result.reason}"
    )

    return {
        **state,
        "intent": result.intent,
    }


# ============================================================
# NODE 3 - CHAT RESPONSE
# ============================================================


def chat_response_node(state: RAGState):

    print("========== INSIDE chat_response_node ==========")

    llm = _get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a friendly credit card assistant.

                If a question is outside the scope of credit card
                assistance, politely reject it and state that you
                can only assist with credit card information
                provided in context or documents.

                Reply briefly and politely.

                Examples:

                User: Hi
                Assistant: Hi! How can I help you today?

                User: How are you?
                Assistant: I'm doing great! How can I assist you?

                User: Thanks
                Assistant: You're welcome!
                """,
            ),
            ("human", "{query}"),
        ]
    )

    result = (prompt | llm).invoke({"query": state["query"]})

    response = {
        "answer": result.content,
        "policy_citations": "",
        "page_no": "",
        "document_name": "",
    }

    return {
        **state,
        "answer": result.content,
        "response": response,
        "messages": state.get("messages", []) + [AIMessage(content=result.content)],
    }


# ============================================================
# NODE 4 - REPHRASER
# ============================================================


def rephraser_node(state: RAGState) -> RAGState:

    print("========== INSIDE rephraser_node ==========")

    llm = _get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a query rephraser for a RAG system.

                Your job is to rephrase the current user question
                into a standalone question using the previous
                conversation.

                Rules:

                1. Preserve the user's original intent.
                2. Resolve references such as:
                   - it
                   - this
                   - that
                   - they
                   - them
                   - above
                   - previous
                   - same
                3. Do not answer the question.
                4. Do not add information that is not present
                   in the conversation.
                5. If the question is already standalone,
                   return it unchanged.
                6. Return ONLY the rephrased question.
                """,
            ),
            (
                "human",
                """
                Previous conversation:
                {history}

                Current user question:
                {query}
                """,
            ),
        ]
    )

    history = "\n".join(
        [f"{msg.type}: {msg.content}" for msg in state.get("messages", [])]
    )

    chain = prompt | llm

    result = chain.invoke(
        {
            "history": history,
            "query": state["query"],
        }
    )

    rephrased_query = result.content.strip()

    print(f"[Rephraser] Original: {state['query']}")

    print(f"[Rephraser] Rephrased: {rephrased_query}")

    return {
        **state,
        "query": rephrased_query,
    }


# ============================================================
# NODE 5 - ROUTER
# ============================================================


def router_node(state: RAGState) -> RAGState:

    print("========== INSIDE router_node ==========")

    llm = _get_llm()

    structured_llm = llm.with_structured_output(RouteDecision)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a helpful credit card assistant for an
                Agentic RAG System.

                Answer the user's question using only the
                provided context.

                Politely reject questions that are out of scope.

                Classify the user's query into EXACTLY one of
                the following routes:

                VECTOR_DB:
                The query asks about policies, procedures,
                guides, guidelines, regulations, or any topic
                that requires reading text documents.

                HYBRID:
                Use when BOTH semantic understanding AND
                keyword matching are important.

                Use HYBRID when:
                - the query contains important keywords AND
                - the query also requires understanding the
                  meaning/context of those keywords.

                FTS:
                Use when the query contains specific keywords,
                names, phrases, document terms, identifiers,
                or exact text that should be matched lexically.

                RDBMS:
                The query asks about products, product prices,
                stock/inventory, product categories, customer
                orders, order items, or anything answerable
                from structured database tables:
                products, categories, orders, order_items.

                IMAGE:
                User requests images, pictures, photos,
                diagrams, or visual assets.

                Reply with the route and one sentence of reason.
                """,
            ),
            (
                "human",
                """
                Question:
                {query}
                """,
            ),
        ]
    )

    chain = prompt | structured_llm

    decision = chain.invoke({"query": state["query"]})

    print(
        f"[router_node's decision]: "
        f"{decision.route} "
        f"and reason: {decision.reason}"
    )

    return {
        **state,
        "route": decision.route,
    }


# ============================================================
# NODE 6 - NL2SQL
# ============================================================


def nl2sql_node(state: RAGState) -> RAGState:

    print("About to generate nl2sql")

    llm = _get_llm()

    db = get_sql_database()

    schema_info = db.get_table_info()

    sql_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a PostgreSQL expert.

                Given the database schema below, write a single
                valid SELECT query that answers the user's question.

                Rules:

                - Return ONLY the raw SQL.
                - No explanation.
                - No summary.
                - No markdown fences.
                - No backticks.
                - Use only tables and columns present in the schema.
                - Do NOT generate INSERT, UPDATE, DELETE, DROP,
                  or any DML/DDL statements.
                - Always add a LIMIT clause (max 50 rows) unless
                  the question asks for aggregates.

                For product or text searches:

                NEVER search for the full multi-word phrase as
                one ILIKE pattern.

                Instead, split the search into individual
                meaningful keywords and OR them together across
                both name and description columns.

                Example:

                User asks:
                "wireless headset"

                Use:

                WHERE
                    (name ILIKE '%wireless%'
                     OR description ILIKE '%wireless%')
                    OR
                    (name ILIKE '%headset%'
                     OR description ILIKE '%headset%')
                    OR
                    (name ILIKE '%headphones%'
                     OR description ILIKE '%headphones%')

                Use your knowledge of synonyms
                (headset/headphones, laptop/notebook, etc.)
                to cast a wider net when the exact term may not match.

                Database schema:
                {schema}
                """,
            ),
            (
                "human",
                """
                Question:
                {question}
                """,
            ),
        ]
    )

    sql_chain = sql_prompt | llm

    raw_sql = sql_chain.invoke(
        {
            "schema": schema_info,
            "question": state["query"],
        }
    )

    print("======== GENERATED raw_sql query is: =====")

    print(raw_sql.content)

    generated_sql = raw_sql.content.strip()

    try:
        sql_result = db.run(generated_sql)
    except Exception as err:
        sql_result = f"Generated SQL execution error: {err}"

    structured_llm = llm.with_structured_output(AIResponse)

    nl_answer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a helpful data analyst.

                Answer the user's question using the SQL query
                results below.

                Be concise and format numbers/lists clearly.

                Set:
                policy_citations = empty string
                page_no = 'N/A'
                document_name = 'agentic_rag_db'

                Do NOT execute INSERT, UPDATE, DELETE, DROP,
                or any DML/DDL statements even if requested.

                Politely deny when users are asking for these
                actions in their queries.

                Never use technical jargon in your response.
                """,
            ),
            (
                "human",
                """
                Question:
                {query}

                SQL Used:
                {sql}

                Query Results:
                {result}
                """,
            ),
        ]
    )

    nl_chain = nl_answer_prompt | structured_llm

    answer = nl_chain.invoke(
        {
            "query": state["query"],
            "sql": generated_sql,
            "result": sql_result,
        }
    )

    print("[nl2sql_node] Answer generated.")

    response = answer.model_dump()

    response["policy_citations"] = "N/A"
    response["sql_query_executed"] = generated_sql

    return {
        **state,
        "generated_sql": generated_sql,
        "sql_result": str(sql_result),
        "response": response,
        "answer": response.get("answer", ""),
        "messages": state.get("messages", [])
        + [AIMessage(content=response.get("answer", ""))],
    }


# ============================================================
# NODE 7 - RERANK
# ============================================================


def rerank_node(state: RAGState):

    co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

    docs = state.get("retrieved_docs", [])

    print("=======3. INSIDE rerank_node. " "Before calling reranker =========")

    print(
        "Retrieved docs count:",
        len(docs),
    )

    if not docs:

        print("No documents found. Skipping reranking.")

        return {
            **state,
            "reranked_docs": [],
        }

    rerank_response = co.rerank(
        model="rerank-v3.5",
        query=state["query"],
        documents=[doc.page_content for doc in docs],
        top_n=5,
    )

    reranked_docs = [docs[r.index] for r in rerank_response.results]

    print(f"[rerank_node] " f"Top {len(reranked_docs)} chunks after reranking:")

    for i, r in enumerate(rerank_response.results):
        print(
            f"  Rank {i + 1} | "
            f"Cohere score: {r.relevance_score:.4f} | "
            f"original index: {r.index}"
        )

    return {
        **state,
        "reranked_docs": reranked_docs,
    }


# ============================================================
# NODE 8 - GENERATE ANSWER
# ============================================================


def generate_answer_node(state: RAGState):

    reranked_docs = state.get(
        "reranked_docs",
        [],
    )

    if not reranked_docs:

        answer = "I could not find relevant information."

        return {
            **state,
            "answer": answer,
            "response": {
                "answer": answer,
                "policy_citations": "",
                "page_no": "",
                "document_name": "",
            },
            "messages": state.get("messages", []) + [AIMessage(content=answer)],
        }

    llm = _get_llm()

    structured_llm = llm.with_structured_output(AIResponse)

    print("=========4. INSIDE GENERATE ANSWER NODE==========")

    for doc in reranked_docs:
        print(
            "Metadata: ",
            doc.metadata,
        )

    context = "\n\n".join(
        [
            (
                f"[Source: "
                f"{doc.metadata.get('source', 'unknown')} "
                f"| Page: "
                f"{doc.metadata.get('page', -1) + 1 if doc.metadata.get('page') is not None else '?'}]"
                f"\n{doc.page_content}"
            )
            for doc in reranked_docs
        ]
    )

    history = "\n".join(
        [f"{msg.type}: {msg.content}" for msg in state.get("messages", [])]
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a helpful credit card assistant.

                Answer the user's question using only the
                provided context.

                Politely reject questions that are out of scope.

                Citation rules:

                - document_name:
                  comma-separated list of EVERY source
                  document you used.

                - page_no:
                  comma-separated page numbers aligned
                  with the documents above.

                - policy_citations:
                  a readable citation combining each
                  document and its page.

                Example:
                KB_Credit_Card_Spend_Summarizer.docx, Page 1

                Always cite ALL versions you drew the answer from,
                not just one.
                """,
            ),
            (
                "human",
                """
                Conversation history:
                {history}

                Context:
                {context}

                Question:
                {query}
                """,
            ),
        ]
    )

    chain = prompt | structured_llm

    result = chain.invoke(
        {
            "history": history,
            "context": context,
            "query": state["query"],
        }
    )

    return {
        **state,
        "context": context,
        "answer": result.answer,
        "response": result.model_dump(),
        "messages": state.get("messages", []) + [AIMessage(content=result.answer)],
    }


# ============================================================
# NODE 9 - EVALUATION
# ============================================================


def evaluation_node(state: RAGState) -> RAGState:

    print("-------- Evaluating Answer ----------")

    history = "\n".join(
        [f"{msg.type}: {msg.content}" for msg in state.get("messages", [])]
    )

    llm = _get_llm()

    prompt = f"""
    User Preferences:
    {history}

    Question:
    {state["query"].lower()}

    Context:
    {state.get("context", "")}

    Answer:
    {state.get("answer", "")}

    Is the answer correct and complete based on the context?

    Respond with only:
    yes
    or
    no
    """

    result = llm.invoke(prompt).content.strip().lower()

    attempts = (
        state.get(
            "attempts",
            0,
        )
        + 1
    )

    print("========== EVALUATION RESULT ==========")

    print(result)

    print(
        "Attempt:",
        attempts,
    )

    print("========================================")

    return {
        **state,
        "is_good": result == "yes",
        "attempts": attempts,
    }


# ============================================================
# EVALUATION ROUTER
# ============================================================


def route(state: RAGState):

    if state.get("is_good") or state.get("attempts", 0) >= 3:
        return "NO_RETRY_REQUIRED"

    return "RETRY_REQUIRED"


# ============================================================
# BUILD GRAPH
# ============================================================


def build_rag_graph(checkpoint):

    workflow = StateGraph(RAGState)

    # --------------------------------------------------------
    # Nodes
    # --------------------------------------------------------

    workflow.add_node(
        "in_memory",
        add_user_message_node,
    )

    workflow.add_node(
        "intent_check",
        intent_check_node,
    )

    workflow.add_node(
        "chat_response",
        chat_response_node,
    )

    workflow.add_node(
        "rephraser",
        rephraser_node,
    )

    workflow.add_node(
        "router",
        router_node,
    )

    workflow.add_node(
        "vector_search",
        vector_search_node,
    )

    workflow.add_node(
        "hybrid_search",
        hybrid_search_node,
    )

    workflow.add_node(
        "fts",
        fts_search_node,
    )

    workflow.add_node(
        "nl2sql",
        nl2sql_node,
    )

    workflow.add_node(
        "rerank",
        rerank_node,
    )

    workflow.add_node(
        "generate_answer",
        generate_answer_node,
    )

    workflow.add_node(
        "evaluation",
        evaluation_node,
    )

    workflow.add_node(
        "image_search",
        extract_images_node,
    )

    # --------------------------------------------------------
    # Entry
    # --------------------------------------------------------

    workflow.set_entry_point("in_memory")

    # --------------------------------------------------------
    # Intent
    # --------------------------------------------------------

    workflow.add_edge(
        "in_memory",
        "intent_check",
    )

    workflow.add_conditional_edges(
        "intent_check",
        lambda state: state["intent"],
        {
            "CHITCHAT": "chat_response",
            "REPHRASER": "rephraser",
        },
    )

    workflow.add_edge(
        "chat_response",
        END,
    )

    # --------------------------------------------------------
    # Rephrase -> Router
    # --------------------------------------------------------

    workflow.add_edge(
        "rephraser",
        "router",
    )

    # --------------------------------------------------------
    # Router
    # --------------------------------------------------------

    workflow.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "FTS": "fts",
            "VECTOR_DB": "vector_search",
            "HYBRID": "hybrid_search",
            "RDBMS": "nl2sql",
            "IMAGE": "image_search",
        },
    )

    # --------------------------------------------------------
    # Retrieval -> Rerank
    # --------------------------------------------------------

    workflow.add_edge(
        "vector_search",
        "rerank",
    )

    workflow.add_edge(
        "fts",
        "rerank",
    )

    workflow.add_edge(
        "hybrid_search",
        "rerank",
    )

    # --------------------------------------------------------
    # Rerank -> Generate
    # --------------------------------------------------------

    workflow.add_edge(
        "rerank",
        "generate_answer",
    )

    # --------------------------------------------------------
    # Generate -> Evaluation
    # --------------------------------------------------------

    workflow.add_edge(
        "generate_answer",
        "evaluation",
    )

    # --------------------------------------------------------
    # Evaluation -> Retry / END
    # --------------------------------------------------------

    workflow.add_conditional_edges(
        "evaluation",
        route,
        {
            "RETRY_REQUIRED": "router",
            "NO_RETRY_REQUIRED": END,
        },
    )

    # --------------------------------------------------------
    # Image
    # --------------------------------------------------------

    workflow.add_edge(
        "image_search",
        END,
    )

    # --------------------------------------------------------
    # Compile with ASYNC checkpointer
    # --------------------------------------------------------

    search_agent = workflow.compile(checkpointer=checkpoint)

    # --------------------------------------------------------
    # Graph visualization
    # --------------------------------------------------------

    try:
        graph_image = search_agent.get_graph().draw_mermaid_png()

        with open(
            "search_agent.png",
            "wb",
        ) as f:
            f.write(graph_image)

    except Exception:
        traceback.print_exc()

    return search_agent


# ============================================================
# ASYNC CHECKPOINTER + GRAPH LIFECYCLE
# ============================================================


@asynccontextmanager
async def create_rag_graph() -> AsyncIterator:
    """
    Create and maintain the async Postgres checkpointer
    for the lifetime of the application.

    IMPORTANT:
    The checkpointer must remain inside its async context
    while the graph is being used.
    """

    print("========== INITIALIZING ASYNC POSTGRES CHECKPOINTER ==========")

    async with AsyncPostgresSaver.from_conn_string(CHECKPOINT_DB_URI) as checkpoint:

        print("========== RUNNING CHECKPOINTER SETUP ==========")

        await checkpoint.setup()

        print("========== CHECKPOINTER READY ==========")

        rag_graph = build_rag_graph(checkpoint)

        try:
            yield rag_graph

        finally:
            print("========== CLOSING ASYNC CHECKPOINTER ==========")


# ============================================================
# RUN SEARCH AGENT
# ============================================================


async def run_search_agent(
    rag_graph,
    query: str,
    thread_id: str,
):

    print("============1. INSIDE run_search_agent")

    initial_state = {
        "query": query,
        "retrieved_docs": [],
        "reranked_docs": [],
        "response": {},
        "images": [],
        "evaluation": {},
        "is_good": False,
        "attempts": 0,
    }

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    final_state = await rag_graph.ainvoke(
        initial_state,
        config=config,
    )

    if final_state.get("route") == "IMAGE":

        print("FINAL STATE:")

        print(final_state)

        print("FINAL IMAGES:")

        print(final_state.get("images"))

        return {
            "query": query,
            "images": final_state.get(
                "images",
                [],
            ),
        }

    # Get the latest checkpoint state.
    state = await rag_graph.aget_state(config)

    print(
        "*************** " "printing state messages ",
        state.values.get(
            "messages",
            [],
        ),
    )

    return final_state.get(
        "response",
        {},
    )


# ============================================================
# STREAMING SEARCH AGENT
# ============================================================


async def run_search_agent_stream(
    rag_graph,
    query: str,
    thread_id: str,
):

    print("============ INSIDE run_search_agent_stream")

    initial_state = {
        "query": query,
        "retrieved_docs": [],
        "reranked_docs": [],
        "response": {},
        "images": [],
        "evaluation": {},
        "is_good": False,
        "attempts": 0,
    }

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    try:

        async for event in rag_graph.astream_events(
            initial_state,
            config=config,
            version="v2",
        ):

            kind = event.get("event")

            print(
                "EVENT:",
                kind,
            )

            node_name = event.get(
                "metadata",
                {},
            ).get("langgraph_node")

            # ------------------------------------------------
            # Stream only final answer nodes
            # ------------------------------------------------

            if kind == "on_chat_model_stream":

                print(
                    "STREAM NODE:",
                    node_name,
                )

                if node_name not in [
                    "chat_response",
                    "generate_answer",
                ]:
                    continue

                chunk = event.get(
                    "data",
                    {},
                ).get("chunk")

                if chunk is None:
                    continue

                content = chunk.content

                if content:
                    yield content

        # ----------------------------------------------------
        # Get final checkpoint state
        # ----------------------------------------------------

        final_state = await rag_graph.aget_state(config)

        state_values = final_state.values

        # ----------------------------------------------------
        # Send final metadata
        # ----------------------------------------------------

        yield {
            "done": True,
            "sources": state_values.get(
                "sources",
                [],
            ),
            "images": state_values.get(
                "images",
                [],
            ),
            "policy_citations": state_values.get(
                "policy_citations",
                "",
            ),
            "page_no": state_values.get(
                "page_no",
                "",
            ),
            "document_name": state_values.get(
                "document_name",
                "",
            ),
        }

    except Exception as e:

        print("Streaming agent error:")

        traceback.print_exc()

        raise e


# ============================================================
# OPTIONAL HELPER FOR FASTAPI / APPLICATION LIFESPAN
# ============================================================

# _rag_graph = None


# @asynccontextmanager
# async def rag_lifespan():

#     """
#     Application-level lifecycle.

#     Use this if you are running FastAPI, for example:

#         @asynccontextmanager
#         async def lifespan(app):
#             async with rag_lifespan() as rag_graph:
#                 app.state.rag_graph = rag_graph
#                 yield
#     """

#     global _rag_graph

#     async with create_rag_graph() as rag_graph:

#         _rag_graph = rag_graph

#         print(
#             "========== RAG GRAPH READY =========="
#         )

#         yield rag_graph

#     _rag_graph = None

#     print(
#         "========== RAG GRAPH SHUTDOWN =========="
#     )
