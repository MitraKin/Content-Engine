import logging
import os
from flask import Flask, render_template, request, jsonify, session
import ollama
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from typing import List, Tuple, Dict, Any

# Set protobuf environment variable to avoid error messages
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

app = Flask(__name__)
# Security: Generate a random secret key if not provided via environment variable
# In production, always set SECRET_KEY environment variable
secret_key = os.environ.get('SECRET_KEY')
if not secret_key:
    import secrets
    secret_key = secrets.token_hex(32)
    logger.warning("SECRET_KEY not set in environment. Using randomly generated key. Sessions will not persist across restarts.")
app.secret_key = secret_key

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Global variables
# WARNING: The global `vector_db` variable may be accessed concurrently by multiple threads
# if using a multi-threaded WSGI server (e.g., Gunicorn with threads). While reads are generally safe,
# concurrent writes or re-initialization can cause race conditions. For production, consider using
# per-request context, a thread-safe singleton, or a process-based server. See DEPLOYMENT.md for details.
vector_db = None
available_models = []


def extract_model_names(models_info: Dict[str, List[Dict[str, Any]]]) -> Tuple[str, ...]:
    """Extract model names from Ollama models info"""
    logger.info("Extracting model names from models_info")
    try:
        models = models_info.get("models")
        if not isinstance(models, list):
            logger.error(f"'models' key is missing or not a list in models_info: {models_info}")
            return ()
        model_names = tuple(model.get("name", "") for model in models if isinstance(model, dict) and "name" in model)
        logger.info(f"Extracted model names: {model_names}")
        return model_names
    except Exception as e:
        logger.error(f"Error extracting model names: {e}. models_info: {models_info}")
        return ()


def load_pdf_files_from_directory(directory_path: str) -> Chroma:
    """Load all PDF files from a directory and create a vector database"""
    logger.info(f"Loading all PDF files from directory: {directory_path}")
    pdf_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.pdf')]

    # Get embedding model from environment or use default
    embedding_model = os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    embeddings = OllamaEmbeddings(model=embedding_model)
    
    if not pdf_files:
        logger.warning("Directory does not contain any PDF files. Creating empty vector database.")
        # Return an empty Chroma instance
        persist_dir = os.environ.get("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
        vector_db = Chroma.from_documents(
            documents=[],
            embedding=embeddings,
            collection_name="myRAG",
            persist_directory=persist_dir
        )
        logger.info("Empty Vector DB created (no documents found)")
        return vector_db

    all_chunks = []
    for pdf_file in pdf_files:
        logger.info(f"Processing file: {pdf_file}")
        loader = UnstructuredPDFLoader(file_path=pdf_file)
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
        chunks = text_splitter.split_documents(data)
        all_chunks.extend(chunks)

    persist_dir = os.environ.get("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    vector_db = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        collection_name="myRAG",
        persist_directory=persist_dir
    )
    logger.info("Vector DB created from directory")
    return vector_db


def process_question(question: str, vector_db: Chroma, selected_model: str) -> str:
    """Process a question using RAG and return the answer"""
    logger.info(f"Processing question: {question} using model: {selected_model}")
    
    llm = ChatOllama(model=selected_model)
    
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate 2
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), 
        llm,
        prompt=QUERY_PROMPT
    )

    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(question)
    logger.info("Question processed and response generated")
    return response


def initialize_app():
    """Initialize the application by loading models and PDFs"""
    global vector_db, available_models
    
    # Get available Ollama models
    try:
        models_info = ollama.list()
        available_models = list(extract_model_names(models_info))
        logger.info(f"Available models: {available_models}")
    except Exception as e:
        logger.error(f"Error getting Ollama models: {e}")
        available_models = []
    
    # Load PDFs from Documents directory
    documents_path = os.path.join(os.path.dirname(__file__), "Documents")
    if os.path.exists(documents_path):
        try:
            logger.info("Loading PDF files...")
            vector_db = load_pdf_files_from_directory(documents_path)
            logger.info("PDF files loaded successfully")
        except Exception as e:
            logger.error(f"Error loading PDF files: {e}")
            vector_db = None
    else:
        logger.warning(f"Documents directory not found: {documents_path}")


@app.route('/')
def index():
    """Render the main page"""
    # Initialize messages in session if not present
    if 'messages' not in session:
        session['messages'] = []
    
    return render_template('index.html', 
                         models=available_models,
                         vector_db_ready=vector_db is not None)


@app.route('/api/models', methods=['GET'])
def get_models():
    """API endpoint to get available models"""
    return jsonify({'models': available_models})


@app.route('/api/status', methods=['GET'])
def get_status():
    """API endpoint to get system status"""
    return jsonify({
        'vector_db_ready': vector_db is not None,
        'models_available': len(available_models) > 0
    })


@app.route('/api/ask', methods=['POST'])
def ask_question():
    """API endpoint to process a question"""
    global vector_db
    
    data = request.get_json()
    question = data.get('question', '').strip()
    model = data.get('model', '')
    
    if not question:
        return jsonify({'error': 'Question is required'}), 400
    
    # Validate question length
    MAX_QUESTION_LENGTH = 2000
    if len(question) > MAX_QUESTION_LENGTH:
        return jsonify({'error': f'Question is too long (max {MAX_QUESTION_LENGTH} characters).'}), 400
    
    if not model:
        return jsonify({'error': 'Model is required'}), 400
    
    # Validate model is available
    if model not in available_models:
        return jsonify({'error': f"Selected model '{model}' is not available."}), 400
    
    if vector_db is None:
        return jsonify({'error': 'PDF files not loaded yet. Please wait or check if Documents directory contains PDF files.'}), 400
    
    try:
        # Process the question
        answer = process_question(question, vector_db, model)
        
        # Store in session with limit
        if 'messages' not in session:
            session['messages'] = []
        
        session['messages'].append({'role': 'user', 'content': question})
        session['messages'].append({'role': 'assistant', 'content': answer})
        
        # Limit the number of messages stored in session to 100
        max_messages = 100
        if len(session['messages']) > max_messages:
            # Remove oldest messages to keep only the latest 100
            session['messages'] = session['messages'][-max_messages:]
        
        session.modified = True
        
        return jsonify({
            'success': True,
            'answer': answer
        })
    
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return jsonify({'error': 'An error occurred while processing your question. Please try again.'}), 500


@app.route('/api/messages', methods=['GET'])
def get_messages():
    """API endpoint to get chat history"""
    messages = session.get('messages', [])
    return jsonify({'messages': messages})


@app.route('/api/clear', methods=['POST'])
def clear_history():
    """API endpoint to clear chat history"""
    session['messages'] = []
    session.modified = True
    return jsonify({'success': True})


if __name__ == '__main__':
    # Initialize the application
    initialize_app()
    
    # Run the Flask app
    # Note: Set debug=False in production and use a proper WSGI server like gunicorn
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    # Bind to localhost by default for security; override with FLASK_HOST for production
    host = os.environ.get('FLASK_HOST', '127.0.0.1')
    app.run(debug=debug_mode, host=host, port=5000)
