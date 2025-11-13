<!DOCTYPE html>
<html lang="en">

<body>
  <h1>🤖 AI-Powered PDF Reader and Q&A System</h1>
  <p>
    This project is a Flask web application that allows users to upload PDF documents, process them into a searchable knowledge base using 
    <strong>LangChain</strong>, <strong>Ollama embeddings</strong>, and <strong>Chroma vector databases</strong>. Users can then interact 
    with the app to ask questions and receive contextually accurate responses using a Retrieval-Augmented Generation (RAG) workflow.
  </p>

  <h2>✨ Features</h2>
  <ul>
    <li>Load multiple PDF documents from a specified directory.</li>
    <li>Convert document content into a vector database for efficient search and retrieval.</li>
    <li>Use local AI models to generate context-aware answers.</li>
    <li>Modern, responsive web interface built with Flask.</li>
    <li>Real-time chat interface for asking questions.</li>
    <li>Session-based chat history.</li>
  </ul>

  <h2>🚀 How to Use</h2>
  <ol>
    <li>Clone this repository:
      <pre><code>git clone https://github.com/MitraKin/Content-Engine.git</code></pre>
    </li>
    <li>Navigate to the project directory:
      <pre><code>cd Content-Engine</code></pre>
    </li>
    <li>Install the required dependencies:
      <pre><code>pip install -r requirements.txt</code></pre>
    </li>
    <li>Make sure you have Ollama installed and running with at least one model (e.g., llama2, mistral):
      <pre><code>ollama pull llama2</code></pre>
    </li>
    <li>Place your PDF documents in the <code>Documents</code> directory.</li>
    <li>Run the Flask application:
      <pre><code>python app.py</code></pre>
    </li>
    <li>Open your browser and navigate to: <a href="http://localhost:5000" target="_blank">http://localhost:5000</a></li>
    <li>Select an AI model from the dropdown menu and start asking questions!</li>
  </ol>

  <h2>📂 Directory Structure</h2>
  <pre>
Content-Engine/
├── app.py                 <!-- Main Flask application -->
├── newapp.py             <!-- Legacy Streamlit application (deprecated) -->
├── requirements.txt      <!-- Dependencies for the project -->
├── README.md            <!-- Project documentation -->
├── Documents/           <!-- Directory for PDF files -->
├── templates/           <!-- HTML templates -->
│   └── index.html
└── static/              <!-- Static files (CSS, JS) -->
    ├── css/
    │   └── style.css
    └── js/
        └── app.js
  </pre>

  <h2>💡 Example Usage</h2>
  <p>
    <strong>Step 1:</strong> Place your PDF files in the <code>Documents</code> directory.<br>
    <strong>Step 2:</strong> Launch the app and wait for PDFs to be processed.<br>
    <strong>Step 3:</strong> Select a model from the dropdown.<br>
    <strong>Step 4:</strong> Type a question in the input field and get responses based on the document content.
  </p>

  <h2>🛠️ Technologies Used</h2>
  <ul>
    <li><strong>Flask:</strong> Web framework for the application.</li>
    <li><strong>LangChain:</strong> AI framework for building RAG workflows.</li>
    <li><strong>Chroma:</strong> Vector database for efficient document retrieval.</li>
    <li><strong>Ollama:</strong> Embeddings and local AI models for processing.</li>
  </ul>

  <h2>🔄 Migration from Streamlit</h2>
  <p>
    This application was migrated from Streamlit to Flask to provide more flexibility in UI development and better integration capabilities.
    The legacy Streamlit application is still available in <code>newapp.py</code> but is deprecated.
  </p>

  <h2>📞 Contact</h2>
  <p>
    For questions or support, please reach out to <a href="mailto:kinnaurm249@gmail.com">kinnaurm249@gmail.com</a>.
  </p>
</body>
</html>
