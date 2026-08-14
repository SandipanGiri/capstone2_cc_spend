# nodes we want
# 1. vector_search (top-k=20)
# 2. rerank
# 3. generate_answer


import os
import cohere
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import Literal
from src.api.v1.states.rag_state import RAGState
from src.api.v1.tools.tools import vector_search_node
from src.api.v1.schemas.query_schemas import AIResponse
from src.core.db import get_sql_database
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()
_USER_ID = "user_123"


def _get_llm():
    return ChatOpenAI(
        model=os.getenv("OPENAI_CHAT_MODEL"), api_key=os.getenv("OPENAI_API_KEY")
    )


class RouteDecision(BaseModel):
    route: Literal["VECTOR_DB", "RDBMS"]
    reason: str  # for debugging


def router_node(state: RAGState) -> RAGState:
    llm = _get_llm()
    structured_llm = llm.with_structured_output(RouteDecision)
    print("========At Router Node=============")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                      You are a query router for an Agentic RAG System.
                      Classify the user's query into EXACTLY one of the following routes: 
                     
                      'VECTOR_DB' -  the auery asks about credit card variants, features, 
                       spend category, billing cycle, reqards, cost, fees, business rule,
                       credit limit management or any topic that requires reading text documents


                      'RBDMS' - the query asks about transactions, statements, credit card with 
                      customer name or anything that needs to read a structured database


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
    print(f"[router_node's decision]: {decision.route} and reason: {decision.reason}")

    return {**state, "route": decision.route}


def nl2sql_node(state: RAGState) -> RAGState:
    print("About to generate nl2sql")
    # connect to LLM
    llm = _get_llm()
    # connect to rdbms
    db = get_sql_database()
    # get the tables' live schema
    schema_info = db.get_table_info()
    # write the system prompt and pass on the schema to get only sql query
    sql_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                   You are a PostgreSQL expert. Given the database schema below,
                   write a single valid SELECT query that answers the user's question.
                   You should never write any DDL, DML, DCL query.


                   Rules:
                   - Return ONLY the raw SQL - no explanation, no summary, no markdown fences, no backticks.
                   - Use only the tables and columns present in the schema.
                   - Do NOT generate INSERT, UPDATE, DELETE, DROP, or any DML/DDL statements.
                   - Always add a LIMIT clause (max 50 rows) unless the question asks for aggregates.
                   - For product or text searches: NEVER search for the full multi-word phrase as one
                       ILIKE pattern. Instead, split the search into individual meaningful keywords
                       and OR them together across both name and description columns.
                       Example - user asks "credit card":
                           WHERE (name ILIKE '%credit%' OR description ILIKE '%credit%')
                           OR (name ILIKE '%card%'  OR description ILIKE '%card%')
                       Use your knowledge of synonyms (billing/payment, transaction/sale/purchase, etc.)
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
    # preprare the chain and invoke with a query
    sql_chain = sql_prompt | llm
    # look for sql query only
    raw_sql = sql_chain.invoke({"schema": schema_info, "question": state["query"]})
    print("========GENERATED raw_sql query is: =====")
    print(raw_sql.content)
    generated_sql = raw_sql.content

    # execute the generated sql query  to get the outout from RDMBS
    try:
        sql_result = db.run(generated_sql)
    except Exception as err:
        sql_result = f"Generated SQL execution error: {err}"

    # connect to LLM to get the natural language response
    structured_llm = llm.with_structured_output(AIResponse)
    nl_answer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful data analyst. Answer the user's question using
               the SQL query results below. Be concise and format numbers/lists clearly.
               Set policy_citations to empty string,
               page_no to 'N/A', and document_name to 'credit_card_rag_db'.
               - Do NOT execute INSERT, UPDATE, DELETE, DROP, or any DML/DDL statements
               even if requested.
               - Politely deny when users are asking for these actions in their queries.
               - Never use tech jargons in your response""",
            ),
            (
                "human",
                "Question: {query}\n\n"
                "SQL Used:\n{sql}\n\n"
                "Query Results:\n{result}",
            ),
        ]
    )

    nl_chain = nl_answer_prompt | structured_llm
    answer = nl_chain.invoke(
        {"query": state["query"], "sql": generated_sql, "result": sql_result}
    )
    print("[nl2sql_node] Answer generated.")
    response = answer.model_dump()
    response["policy_citations"] = "N/A"
    response["sql_query_executed"] = generated_sql
    # return the sql query is RAGState
    # and also the output in sql_result of RAGState
    return {
        **state,
        "generated_sql": generated_sql,
        "sql_result": str(sql_result),
        "response": response,
    }


def rerank_node(state: RAGState):
    # establish connection with the cohere reranking model
    co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
    # send the query and the retrieved_docs to the reranking model

    docs = state["retrieved_docs"]

    print("=======3. INSIDE rerank_node. Before calling reranker =========")
    rerank_response = co.rerank(
        model="rerank-v3.5",
        query=state["query"],
        documents=[doc.page_content for doc in docs],
        top_n=5,
    )

    # Map Cohere result indices back to LangChain Document objects
    reranked_docs = [docs[r.index] for r in rerank_response.results]

    print(f"[rerank_node] Top {len(reranked_docs)} chunks after reranking:")
    for i, r in enumerate(rerank_response.results):
        print(
            f"  Rank {i+1} | Cohere score: {r.relevance_score:.4f} | original index: {r.index}"
        )

    return {**state, "reranked_docs": reranked_docs}


def generate_answer_node(state: RAGState):
    llm = _get_llm()
    structured_llm = llm.with_structured_output(AIResponse)

    print("=========4. INSIDE GENERATE ANSWER NODE==========")

    for doc in state["reranked_docs"]:
        print("Metadata: ", doc.metadata)

    # let's prepare the context
    context = "\n\n".join(
        [
            f"[Source: {doc.metadata.get('source', 'unknown')} | Page: {doc.metadata.get('page', -1) + 1 if doc.metadata.get('page') is not None else '?'}]\n{doc.page_content}"
            for doc in state["reranked_docs"]
        ]
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                   You are a helpful assistant. Answer the user's question using only the
                   provided context. You can combine the answers form all the nodes and
                   finally generate an appropriate answer.


                   Citation rules (fill the structured fields):
                   - document_name: comma-separated list of EVERY source document you used.
                   - page_no: comma-separated page numbers, aligned with the documents above.
                   - policy_citations: a readable citation combining each document and its page
                   (e.g. "KB_Credit_Card_Spend_Summarizer, Page 1").
                   - Always cite ALL versions you drew the answer from, not just one.
           """,
            ),
            (
                "human",
                """
                   Context:
                   {context}


                   Question:
                   {query}
               """,
            ),
        ]
    )

    chain = prompt | structured_llm
    result = chain.invoke({"context": context, "query": state["query"]})

    print(f"[generate_answer_node] Answer generated.")
    return {**state, "response": result.model_dump()}


def build_rag_graph():
    workflow = StateGraph(RAGState)

    workflow.add_node("router", router_node)
    workflow.add_node("nl2sql", nl2sql_node)
    workflow.add_node("vector_search", vector_search_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("generate_answer", generate_answer_node)

    # the following is the starting point
    workflow.set_entry_point("router")

    # conditional routing: "vectordb" -> vector_search (or) "rdbms" -> nl2sql
    workflow.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {"VECTOR_DB": "vector_search", "RDBMS": "nl2sql"},
    )

    workflow.add_edge("vector_search", "rerank")
    workflow.add_edge("rerank", "generate_answer")
    workflow.add_edge("generate_answer", END)

    # -----------------
    # checkpoints
    # -----------------

    checkpoint = InMemorySaver()
    search_agent = workflow.compile(checkpointer=checkpoint)

    # generating and saving the graph visualization
    graph_image = search_agent.get_graph().draw_mermaid_png()
    with open("search_agent.png", "wb") as f:
        f.write(graph_image)

    return search_agent


rag_graph = build_rag_graph()


# non streaming response
def run_search_agent(query: str, user_id: str):
    print("============1. INSIDE run_search_agent ")
    initial_state = {
        "query": query,
        "retrieved_docs": [],
        "reranked_docs": [],
        "response": {},
    }

    # ---------------
    # using thread
    # ---------------
    user = user_id if user_id else _USER_ID
    config = {"configurable": {"thread_id": user}}
    final_state = rag_graph.invoke(initial_state, config=config)
    return final_state["response"]


async def run_search_agent_stream(query: str):
    print("============1. INSIDE run_search_agent ")
    initial_state = {
        "query": query,
        "retrieved_docs": [],
        "reranked_docs": [],
        "response": {},
    }

    async for event in rag_graph.astream_events(initial_state, version="v1"):
        kind = event["event"]
        # print(kind)

        # if it is a token generated by the chat model
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                # format as an Server Side Event data straem payload
                yield f"data: {json.dumps({'token': content})}\n\n"

    yield "data: [DONE]\n\n"
