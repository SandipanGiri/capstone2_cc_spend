import streamlit as st
import requests
from datetime import datetime
import re
from css import load_styles
import uuid 
import base64
from PIL import Image
from io import BytesIO  
import base64

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(page_title="Credit Card Chatbot", layout="wide")
load_styles()

BACKEND_URL = "http://localhost:8000/api/v1/query"
# UPLOAD_URL = "http://localhost:8000/api/v1/upload"
UPLOAD_URL = "http://localhost:8000/api/v1/upload"


st.markdown(
    """
    <style>

    /* ---------- Global ---------- */

    .stApp {
        background: #e8f2ef;
    }

    </style>
    """,
    unsafe_allow_html=True,
)
# backgrund color  #f7f8fc
# -------------------------
# Sidebar
# -------------------------

# st.sidebar.title("⚙️ Settings")

# backend_url = st.sidebar.text_input("Backend URL", value="http://localhost:8000/chat")

# temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.1)

# top_k = st.sidebar.slider("Retrieved Documents", 1, 10, 5)

st.sidebar.markdown("---")
# st.sidebar.info("""
#     **RAG Chatbot**

#     - Ask questions
#     - View retrieved documents
#     - Powered by your backend
#     """)

# -------------------------
# Session State
# -------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "upload_results" not in st.session_state:
    st.session_state.upload_results = []

if "is_streaming" not in st.session_state:
    st.session_state.is_streaming = False

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())    

if "reset_uploader" not in st.session_state:
    st.session_state.reset_uploader = False

# Reset the uploader on the rerun
if st.session_state.reset_uploader:
    st.session_state.reset_uploader = False
    st.rerun()

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
# -------------------------
# Header
# -------------------------

st.title("💳 Credit Card AI Assistant")
st.caption("Your credit card spend analyzer.")

# -------------------------
# Display Previous Messages
# -------------------------

for message in st.session_state.messages:
    # with st.chat_message(message["role"]):
    #     st.markdown(message["content"])

    css_class = "user-msg" if message["role"] == "user" else "assistant-msg"

    st.markdown(
            f"""
            <div class="{css_class}">
                {message["content"]}
            </div>
            """,
            unsafe_allow_html=True,
        )

    if message["role"] == "assistant":

        for img in message.get("images", []):

            try:
                image_bytes = base64.b64decode(img)

                image_bytes = base64.b64decode(
                        img["content"]
                    )

                st.image(image,
                    caption=img.get("source_file", "Agent Image")
                )

            except Exception:
                print("unable to display the images")

        sources = message.get("sources", [])

        if sources:
            with st.expander("📚 Sources"):
                for i, src in enumerate(sources, start=1):
                    st.markdown(f"**{i}. {src.get('title','Document')}**")
                    st.write(src.get("content", ""))
                    st.markdown("---")

# -------------------------
# Chat Input
# -------------------------

prompt = st.chat_input("Ask something...")

if prompt:

    st.session_state.messages.append(
            {"role": "user", "content": prompt}
    )

    st.markdown(
        f"""
        <div class="user-msg">
            {prompt}
        </div>
        """,
        unsafe_allow_html=True,
    )

    #placeholder = st.empty()

    placeholder = st.empty()

    with st.spinner("Thinking..."):

            try:
                response = requests.post(
                    BACKEND_URL,
                    json={
                        "query": prompt,
                        "thread_id": st.session_state.thread_id,
                        # "temperature": temperature,
                        # "top_k": top_k,
                    },
                    timeout=120,
                )

                response.raise_for_status()

                data = response.json()

                answer = data.get("answer", "No answer returned.")

                sources = data.get("sources", [])

                # Receive image returned by agent
                images = data.get("images", [])

               
                placeholder.markdown(
                        f"""
                        <div class="assistant-msg">
                            {answer}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Display agent returned image
                # Display agent returned images
                for img in images:

                    try:

                        if isinstance(img, dict):
                            image_data = img.get("content")
                            caption = img.get("source_file", "Agent Image")

                        else:
                            image_data = img
                            caption = "Agent Image"


                        if image_data:

                            image_bytes = base64.b64decode(image_data)

                            image = Image.open(
                                BytesIO(image_bytes)
                            )

                            st.image(
                                image,
                                caption=caption,
                                use_container_width=True
                            )


                    except Exception as e:
                        st.error(
                            f"Unable to display image: {e}"
                        )

                if sources:
                    with st.expander("📚 Retrieved Sources", expanded=False):

                        for idx, source in enumerate(sources, start=1):

                            st.markdown(f"### {idx}. {source.get('title','Document')}")

                            if "score" in source:
                                st.caption(f"Similarity: {source['score']:.3f}")

                            st.write(source.get("content", ""))

                            if source.get("metadata"):

                                st.json(source["metadata"])

                            st.markdown("---")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "images": images,
                        "time": datetime.now().isoformat(),
                    }
                )

            except requests.exceptions.ConnectionError:
                st.error(
                    f"Cannot connect to backend at {BACKEND_URL}. Is the backend running?"
                )

            except requests.exceptions.RequestException as e:
                st.error(f"Backend Error:\n\n{e}")

# ---------------
# stream answer
# ---------------


# def stream_answer(prompt, backend_url):

#     response = requests.post(backend_url, json={"question": prompt}, stream=True)

#     for line in response.iter_lines():

#         if line:
#             yield line.decode("utf-8")


# with st.chat_message("assistant"):

#     response = st.write_stream(stream_answer(prompt, backend_url))


# ---------------
# document upload
# ---------------
# uploaded_files = st.sidebar.file_uploader(
#     "Upload Documents", type=["pdf", "docx"], accept_multiple_files=True
# )

# if uploaded_files:

#     if st.sidebar.button("Ingest"):

#         # Prepare the file payload for form-data transmission
#         # files = {"file": (uploaded_files.name, uploaded_files.getvalue())}

#         with st.sidebar.spinner("Ingesting file..."):
#             try:

#                 for file in uploaded_files:

#                     response = requests.post(
#                         UPLOAD_URL,
#                         files={"file": (file.name, file.getvalue())},
#                     )

#                     # Handle API Exceptions and Successes
#                     if response.status_code == 201:
#                         st.sidebar.success("Document(s) uploaded.")
#                         st.sidebar.json(response.json())
#                     else:
#                         error_msg = response.json().get("detail", "File upload failed")
#                         st.sidebar.error(f"Error {response.status_code}: {error_msg}")

#             except requests.exceptions.ConnectionError:
#                 st.error("Could not connect to the server. Check server status.")
#             except Exception as e:
#                 st.error(f"An unexpected error occurred: {e}")



uploaded_files = st.sidebar.file_uploader(
    "Upload Documents",
    type=["pdf", "docx"],
    accept_multiple_files=True,
    key=f"document_uploader_{st.session_state.uploader_key}",
)


if uploaded_files and st.sidebar.button("Ingest"):
    successful_files = []
    failed_files = []

    with st.sidebar.spinner("Ingesting files..."):
        for file in uploaded_files:
            try:
                files = {
                    "file": (
                        file.name,
                        file.getvalue(),
                        file.type,
                    )
                }

                response = requests.post(
                    UPLOAD_URL,
                    files=files,
                    timeout=60,
                )

                if response.status_code == 201:
                    successful_files.append(file.name)
                else:
                    try:
                        error_msg = response.json().get(
                            "detail",
                            "File upload failed"
                        )
                    except ValueError:
                        error_msg = response.text or "File upload failed"

                    failed_files.append(
                        f"{file.name}: Error {response.status_code} - {error_msg}"
                    )

            except requests.exceptions.ConnectionError:
                failed_files.append(
                    f"{file.name}: Could not connect to the server."
                )

            except requests.exceptions.Timeout:
                failed_files.append(
                    f"{file.name}: Request timed out."
                )

            except Exception as e:
                failed_files.append(
                    f"{file.name}: {str(e)}"
                )

    # Display results
    if successful_files:
        st.sidebar.success(
            f"Successfully uploaded {len(successful_files)} file(s)."
        )

        for filename in successful_files:
            st.sidebar.write(f"✅ {filename}")

    if failed_files:
        st.sidebar.error(
            f"{len(failed_files)} file(s) failed to upload."
        )

        for error in failed_files:
            st.sidebar.write(f"❌ {error}")



# ---------------
# Citation Highlighting
# ---------------

# def format_citations(answer):

#     return re.sub(r"\[(\d+)\]", r"<sup style='color:blue'>[\1]</sup>", answer)


# st.markdown(format_citations(answer), unsafe_allow_html=True)


# with st.expander("Sources"):

#     for idx, source in citations.items():
#         st.markdown(f"**[{idx}]** {source}")


# ---------------
# save chat
# ---------------

# save_chat(st.session_state.session_id, st.session_state.messages)

# st.sidebar.write(f"Session: {st.session_state.session_id[:8]}")

if st.sidebar.button("🗑 Clear Chat"):
    st.session_state.messages = []
    st.session_state.uploader_key = 0
    st.rerun()