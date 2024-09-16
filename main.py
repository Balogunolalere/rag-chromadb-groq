# Imports
import streamlit as st
import groq
import os
import json
import time
from typing import List, Dict, Tuple, Any
from dotenv import load_dotenv
from io import BytesIO
import logging
from functools import lru_cache
from collections import deque

import spacy
from langdetect import detect

import chromadb
from chromadb.utils import embedding_functions

from PyPDF2 import PdfReader

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

REQUIRED_ENV_VARS = ["GROQ_API_KEY"]
PERSIST_DIRECTORY = "chromadb_data"
COLLECTION_NAME = "document_collection"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
LLM_MODEL = "llama-3.1-70b-versatile"

# Initialization
nlp = spacy.load("en_core_web_lg")

def check_env_vars():
    missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

check_env_vars()

client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))
chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
)

# Caching
@lru_cache(maxsize=100)
def cached_api_call(messages: Tuple[Tuple[str, str]], max_tokens: int) -> str:
    messages_list = [{"role": role, "content": content} for role, content in messages]
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages_list,
        max_tokens=max_tokens,
        temperature=0.2
    )
    return response.choices[0].message.content

# Text Chunking
class TextChunker:
    def chunk_text(self, text: str) -> List[str]:
        raise NotImplementedError("Subclasses must implement this method")

class SlidingWindowChunker(TextChunker):
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        doc = nlp(text)
        sentences = list(doc.sents)
        chunks = []
        window = deque(maxlen=self.chunk_size)
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence.text.split())
            
            if current_size + sentence_size > self.chunk_size:
                chunks.append(" ".join([s.text for s in window]))
                while current_size > self.overlap:
                    removed = window.popleft()
                    current_size -= len(removed.text.split())
            
            window.append(sentence)
            current_size += sentence_size

        if window:
            chunks.append(" ".join([s.text for s in window]))

        return chunks

class SemanticChunker(TextChunker):
    def __init__(self, max_chunk_size: int = 1000, similarity_threshold: float = 0.5):
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold

    def chunk_text(self, text: str) -> List[str]:
        doc = nlp(text)
        sentences = list(doc.sents)
        chunks = []
        current_chunk = []
        current_size = 0

        for i, sentence in enumerate(sentences):
            sentence_size = len(sentence.text.split())
            if current_size + sentence_size > self.max_chunk_size:
                if current_chunk:
                    chunks.append(" ".join([s.text for s in current_chunk]))
                    current_chunk = []
                    current_size = 0

            current_chunk.append(sentence)
            current_size += sentence_size

            if i < len(sentences) - 1:
                next_sentence = sentences[i + 1]
                similarity = sentence.similarity(next_sentence)
                if similarity < self.similarity_threshold and current_size >= self.max_chunk_size // 2:
                    chunks.append(" ".join([s.text for s in current_chunk]))
                    current_chunk = []
                    current_size = 0

        if current_chunk:
            chunks.append(" ".join([s.text for s in current_chunk]))

        return chunks

def get_chunker(method: str) -> TextChunker:
    if method == "sliding_window":
        return SlidingWindowChunker()
    elif method == "semantic":
        return SemanticChunker()
    else:
        raise ValueError("Invalid chunking method. Choose 'sliding_window' or 'semantic'.")

# Document Processing
def process_pdf(file: BytesIO) -> Tuple[str, Dict[str, Any]]:
    text = ""
    metadata = {"pages": []}

    try:
        pdf = PdfReader(file)
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            text += page_text + "\n"
            metadata["pages"].append({"number": i + 1, "text": page_text})

        metadata["num_pages"] = len(pdf.pages)
        if pdf.metadata:
            for key, value in pdf.metadata.items():
                if key != "pages":
                    metadata[key] = value

    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise

    return text, metadata

def safe_file_read(file, fallback_encoding='latin1'):
    try:
        return file.read().decode('utf-8')
    except UnicodeDecodeError:
        file.seek(0)
        return file.read().decode(fallback_encoding)

def clean_text(text: str) -> str:
    text = " ".join(text.split())
    return ''.join(char for char in text if char.isprintable() or char.isspace())

# Storage Management
def clean_storage():
    global chroma_client, collection

    chroma_client.delete_collection(name=COLLECTION_NAME)
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    )

    st.session_state.document_metadata = {}

    if os.path.exists("history.json"):
        os.remove("history.json")

    st.success("Storage cleaned successfully!")

# Document Upload
def upload_document():
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

    if uploaded_file is not None:
        if uploaded_file.name in st.session_state.document_metadata:
            st.warning(f"Document '{uploaded_file.name}' has already been processed.")
            return

        if uploaded_file.type == "application/pdf":
            text, metadata = process_pdf(BytesIO(uploaded_file.getvalue()))
        elif uploaded_file.type == "text/plain":
            text = safe_file_read(uploaded_file)
            metadata = {"type": "txt"}

        text = clean_text(text)

        try:
            language = detect(text)
        except:
            language = "unknown"

        metadata.update({
            "language": language,
            "file_name": uploaded_file.name,
            "file_size": uploaded_file.size
        })

        if "pages" in metadata:
            del metadata["pages"]

        for key, value in metadata.items():
            if isinstance(value, (list, dict)):
                metadata[key] = json.dumps(value)

        chunking_method = st.selectbox("Choose chunking method:", ["sliding_window", "semantic"])
        chunker = get_chunker(chunking_method)
        passages = chunker.chunk_text(text)

        ids = [f"{uploaded_file.name}_{i}" for i in range(len(passages))]

        collection.add(
            ids=ids,
            documents=passages,
            metadatas=[{"chunk_id": i, "source": uploaded_file.name} for i in range(len(passages))]
        )

        st.session_state.document_metadata[uploaded_file.name] = metadata

        st.success(f"Document '{uploaded_file.name}' uploaded and processed successfully!")
        st.write("Document Details:")
        st.json(metadata)

# Query Processing
def retrieve_relevant_passages(query: str) -> List[str]:
    results = collection.query(
        query_texts=[query],
        n_results=5
    )
    return results['documents'][0]

def make_api_call(messages: List[Dict[str, str]], max_tokens: int, is_final_answer: bool = False) -> str:
    messages_tuple = tuple((msg["role"], msg["content"]) for msg in messages)

    for attempt in range(3):
        try:
            return cached_api_call(messages_tuple, max_tokens)
        except Exception as e:
            logger.error(f"API call attempt {attempt + 1} failed: {str(e)}")
            if attempt == 2:
                error_message = f"Failed to generate {'final answer' if is_final_answer else 'step'} after 3 attempts. Error: {str(e)}"
                logger.error(error_message)
                return json.dumps({"title": "Error", "content": error_message, "next_action": "final_answer" if not is_final_answer else None})
            time.sleep(1)

def generate_response(prompt: str, retrieved_context: str):
    messages = [
        {"role": "system", "content": "You are an expert AI assistant. Use the provided context to generate your answer. Follow these steps: 1) Analyze the question. 2) Identify key information from the context. 3) Reason about the answer. 4) Formulate a clear response. Your response for each step MUST be in valid JSON format with 'step_number', 'title', 'content', and 'next_action' fields. The 'content' field MUST be a string. The 'next_action' field should be 'continue' for steps 1-3 and 'final_answer' for step 4."},
        {"role": "user", "content": f"Context: {retrieved_context}\n\nQuestion: {prompt}\n\nBegin with step 1: Analyze the question. Respond in strict JSON format as instructed."}
    ]

    steps = []
    total_thinking_time = 0

    for step_number in range(1, 5):
        start_time = time.time()
        step_response = make_api_call(messages, 300)
        end_time = time.time()
        thinking_time = end_time - start_time
        total_thinking_time += thinking_time

        try:
            step_data = json.loads(step_response)
            if not isinstance(step_data.get('content'), str):
                raise ValueError("'content' field must be a string")
            if 'next_action' not in step_data or step_data['next_action'] not in ['continue', 'final_answer']:
                raise ValueError("Invalid or missing 'next_action' field")
            if int(step_data.get('step_number', 0)) != step_number:
                raise ValueError(f"Expected step {step_number}, got step {step_data.get('step_number')}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {step_response}")
            step_data = {"step_number": step_number, "title": "Error", "content": f"Failed to parse response: {str(e)}", "next_action": "final_answer"}
        except ValueError as e:
            logger.error(f"Invalid JSON structure: {str(e)}")
            step_data = {"step_number": step_number, "title": "Error", "content": str(e), "next_action": "final_answer"}

        steps.append((f"Step {step_number}: {step_data['title']}", step_data['content'], thinking_time))

        if step_number < 4:
            messages.append({"role": "assistant", "content": json.dumps(step_data)})
            next_step_prompts = [
                "Now, proceed with step 2: Identify key information from the context.",
                "Next, move to step 3: Reason about the answer.",
                "Finally, complete step 4: Formulate a clear response."
            ]
            messages.append({"role": "user", "content": f"{next_step_prompts[step_number - 1]} Respond in strict JSON format as instructed."})
        else:
            return steps, total_thinking_time, step_data['content']

    return steps, total_thinking_time, "Error: Failed to generate a final answer."

# History Management
def save_to_history(query: str, response: str):
    history_entry = {"query": query, "response": response, "timestamp": time.time()}
    try:
        with open("history.json", "r+") as f:
            content = f.read()
            history = json.loads(content) if content else []
            history.append(history_entry)
            f.seek(0)
            json.dump(history, f, indent=2)
            f.truncate()
    except FileNotFoundError:
        with open("history.json", "w") as f:
            json.dump([history_entry], f, indent=2)

def show_history():
    try:
        with open("history.json", "r") as f:
            history = json.load(f)

        history.sort(key=lambda x: x['timestamp'], reverse=True)

        for entry in history[:10]:
            st.write(f"**Query**: {entry['query']}")
            st.write(f"**Response**: {entry['response']}")
            st.write(f"**Time**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(entry['timestamp']))}")
            st.write("---")
    except FileNotFoundError:
        st.write("No history available.")


# Main Application
def main():
    st.set_page_config(page_title="RAG with ChromaDB", page_icon="🧠", layout="wide")

    st.title("RAG with ChromaDB and Groq")

    st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="big-font">Upload PDF or TXT files, create a searchable knowledge base, and query documents using retrieval-augmented generation.</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])

    with col1:
        upload_document()

    with col2:
        if st.button("Clean Storage"):
            clean_storage()

    user_query = st.text_input("Enter your query:", placeholder="e.g., What is the main idea of the document?")
    submit_button = st.button("Submit Query")

    if submit_button and user_query:
        retrieved_passages = retrieve_relevant_passages(user_query)
        context = " ".join(retrieved_passages)

        with st.spinner("Generating response..."):
            try:
                steps, total_thinking_time, final_answer = generate_response(user_query, context)

                for title, content, thinking_time in steps:
                    with st.expander(title, expanded=True):
                        st.markdown(content.replace('\n', '<br>'), unsafe_allow_html=True)
                        st.write(f"Thinking time: {thinking_time:.2f} seconds")

                st.markdown("### Final Answer")
                st.markdown(final_answer.replace('\n', '<br>'), unsafe_allow_html=True)

                save_to_history(user_query, final_answer)
                st.success("Response generated successfully!")
                st.write(f"Total thinking time: {total_thinking_time:.2f} seconds")
            except Exception as e:
                st.error(f"An error occurred while generating the response: {str(e)}")
                logger.error(f"Error in generate_response: {str(e)}", exc_info=True)

    if st.button("Show History"):
        show_history()

if __name__ == "__main__":
    main()