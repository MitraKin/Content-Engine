<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI-Powered PDF Reader</title>
</head>
<body>
  <h1>🤖 AI-Powered PDF Reader and Q&A System</h1>
  <p>
    This project is a Streamlit application that allows users to upload PDF documents, process them into a searchable knowledge base using 
    <strong>LangChain</strong>, <strong>Ollama embeddings</strong>, and <strong>Chroma vector databases</strong>. Users can then interact 
    with the app to ask questions and receive contextually accurate responses using a Retrieval-Augmented Generation (RAG) workflow.
  </p>

  <h2>✨ Features</h2>
  <ul>
    <li>Load multiple PDF documents from a specified directory.</li>
    <li>Convert document content into a vector database for efficient search and retrieval.</li>
    <li>Use local AI models to generate context-aware answers.</li>
    <li>User-friendly Streamlit interface for interaction.</li>
  </ul>

  <h2>🚀 How to Use</h2>
  <ol>
    <li>Clone this repository:
      <pre><code>git clone https://github.com/your-username/ai-pdf-reader.git</code></pre>
    </li>
    <li>Navigate to the project directory:
      <pre><code>cd ai-pdf-reader</code></pre>
    </li>
    <li>Install the required dependencies:
      <pre><code>pip install -r requirements.txt</code></pre>
    </li>
    <li>Specify the directory containing PDF documents by updating the <code>file_upload</code> variable in the code:
      <pre><code>file_upload = "path/to/your/pdf/directory"</code></pre>
    </li>
    <li>Run the Streamlit app:
      <pre><code>streamlit run app.py</code></pre>
    </li>
    <li>Open the application in your browser (default: <a href="http://localhost:8501" target="_blank">http://localhost:8501</a>).</li>
    <li>Select an AI model from the dropdown menu and start asking questions!</li>
  </ol>

  <h2>📂 Directory Structure</h2>
  <pre>
ai-pdf-reader/
├── app.py             <!-- Main Streamlit application -->
├── requirements.txt   <!-- Dependencies for the project -->
└── README.html        <!-- Project documentation -->
  </pre>

  <h2>💡 Example Usage</h2>
  <p>
    <strong>Step 1:</strong> Place your PDF files in the specified directory.<br>
    <strong>Step 2:</strong> Launch the app and process the documents.<br>
    <strong>Step 3:</strong> Type a question in the input field and get responses based on the document content.
  </p>

  <h2>🛠️ Technologies Used</h2>
  <ul>
    <li><strong>Streamlit:</strong> Interactive user interface.</li>
    <li><strong>LangChain:</strong> AI framework for building RAG workflows.</li>
    <li><strong>Chroma:</strong> Vector database for efficient document retrieval.</li>
    <li><strong>Ollama:</strong> Embeddings and local AI models for processing.</li>
  </ul>

 

  <h2>📞 Contact</h2>
  <p>
    For questions or support, please reach out to <a href="kinnaurm249@gmail.com">kinnaurm249@gmail.com</a>.
  </p>
</body>
</html>
