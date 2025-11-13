# Application Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         User Browser                         │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐ │
│  │   HTML     │  │    CSS     │  │      JavaScript        │ │
│  │ (Template) │  │  (Styles)  │  │   (AJAX Requests)      │ │
│  └────────────┘  └────────────┘  └────────────────────────┘ │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP/REST API
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      Flask Web Server                        │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                    Routes/Endpoints                     │ │
│  │  • GET  /                    (Main Page)               │ │
│  │  • GET  /api/models          (List Models)             │ │
│  │  • GET  /api/status          (System Status)           │ │
│  │  • POST /api/ask             (Ask Question)            │ │
│  │  • GET  /api/messages        (Get History)             │ │
│  │  • POST /api/clear           (Clear History)           │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                  Session Management                     │ │
│  │  • User chat history                                   │ │
│  │  • Session-based storage                               │ │
│  └────────────────────────────────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    RAG Processing Layer                      │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              Document Processing Pipeline              │ │
│  │                                                         │ │
│  │  1. PDF Loader (UnstructuredPDFLoader)                │ │
│  │          ↓                                              │ │
│  │  2. Text Splitter (RecursiveCharacterTextSplitter)    │ │
│  │          ↓                                              │ │
│  │  3. Embeddings (OllamaEmbeddings - nomic-embed-text)  │ │
│  │          ↓                                              │ │
│  │  4. Vector Store (Chroma DB)                          │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              Question Answering Pipeline               │ │
│  │                                                         │ │
│  │  1. Multi-Query Retriever (generates query variants)  │ │
│  │          ↓                                              │ │
│  │  2. Vector DB Search (retrieve relevant docs)         │ │
│  │          ↓                                              │ │
│  │  3. Context + Question → Prompt Template              │ │
│  │          ↓                                              │ │
│  │  4. LLM Processing (ChatOllama)                       │ │
│  │          ↓                                              │ │
│  │  5. Response Generation                                │ │
│  └────────────────────────────────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                        Ollama Service                        │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                   Local LLM Models                      │ │
│  │  • Model Management                                    │ │
│  │  • Embeddings Generation                               │ │
│  │  • Text Generation                                     │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      File System                             │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                 Documents Directory                     │ │
│  │  • PDF files for processing                            │ │
│  │  • Loaded on application startup                       │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Startup Sequence
```
Application Start
    ├─→ Initialize Flask app
    ├─→ Load Ollama models list
    ├─→ Scan Documents/ directory
    ├─→ Process PDFs
    │   ├─→ Load documents
    │   ├─→ Split into chunks
    │   ├─→ Generate embeddings
    │   └─→ Store in Chroma vector DB
    └─→ Start web server
```

### 2. Question Processing Flow
```
User Question
    ├─→ Browser sends POST /api/ask
    ├─→ Flask receives request
    ├─→ Validate question and model
    ├─→ Multi-Query Retriever
    │   ├─→ Generate query variants
    │   └─→ Search vector DB for each variant
    ├─→ Retrieve relevant document chunks
    ├─→ Build prompt with context
    ├─→ Send to Ollama LLM
    ├─→ Receive generated answer
    ├─→ Store in session history
    └─→ Return JSON response to browser
```

### 3. Session Management
```
User Session
    ├─→ Flask session cookie
    ├─→ Store messages list
    │   ├─→ User messages
    │   └─→ Assistant responses
    └─→ Persist until cleared or session expires
```

## Component Responsibilities

### Frontend (Browser)
- **HTML Template**: Structure and layout
- **CSS**: Styling and responsive design
- **JavaScript**: 
  - User interaction handling
  - AJAX requests to API
  - DOM manipulation
  - Error display

### Backend (Flask)
- **Routing**: Handle HTTP requests
- **Session Management**: Store chat history
- **API**: RESTful endpoints
- **Integration**: Connect to RAG pipeline

### RAG Pipeline
- **Document Processing**: Load and chunk PDFs
- **Embeddings**: Convert text to vectors
- **Vector Storage**: Chroma database
- **Retrieval**: Multi-query search
- **Generation**: LLM-based answers

### External Services
- **Ollama**: Local LLM and embeddings
- **File System**: PDF storage

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Frontend | HTML5, CSS3, JavaScript | User interface |
| Web Framework | Flask | HTTP server & routing |
| Session | Flask Sessions | State management |
| Document Loading | LangChain UnstructuredPDFLoader | PDF processing |
| Text Splitting | LangChain RecursiveCharacterTextSplitter | Chunking |
| Embeddings | Ollama (nomic-embed-text) | Vector generation |
| Vector DB | Chroma | Document storage & retrieval |
| Retrieval | LangChain MultiQueryRetriever | Enhanced search |
| LLM | Ollama (user-selected model) | Answer generation |
| Prompts | LangChain ChatPromptTemplate | Prompt engineering |

## Security Layers

```
┌─────────────────────────────────────────────────────────────┐
│                     Security Measures                        │
├─────────────────────────────────────────────────────────────┤
│ 1. Session Security                                         │
│    • Secret key for session encryption                      │
│    • Server-side session storage                            │
├─────────────────────────────────────────────────────────────┤
│ 2. Error Handling                                           │
│    • Generic error messages to users                        │
│    • Detailed errors only in logs                           │
├─────────────────────────────────────────────────────────────┤
│ 3. Debug Mode                                               │
│    • Disabled by default                                    │
│    • Environment variable controlled                        │
├─────────────────────────────────────────────────────────────┤
│ 4. Input Validation                                         │
│    • Question and model validation                          │
│    • Empty input rejection                                  │
└─────────────────────────────────────────────────────────────┘
```

## Scalability Considerations

### Current Design
- Single-process Flask server
- In-memory vector database
- Session-based state

### Production Recommendations
1. **WSGI Server**: Use Gunicorn with multiple workers
2. **Reverse Proxy**: Nginx for static files and load balancing
3. **Vector DB**: Consider persistent Chroma storage
4. **Caching**: Add Redis for session storage
5. **Queue**: Use Celery for long-running RAG tasks

## File Structure

```
Content-Engine/
├── app.py                      # Main Flask application
├── newapp.py                  # Legacy Streamlit (deprecated)
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore rules
│
├── Documents/                 # PDF files directory
│   ├── goog-10-k-2023.pdf
│   ├── tsla-20231231-gen.pdf
│   └── uber-10-k-2023.pdf
│
├── templates/                 # Jinja2 templates
│   └── index.html            # Main UI template
│
├── static/                    # Static assets
│   ├── css/
│   │   └── style.css         # Application styles
│   └── js/
│       └── app.js            # Client-side logic
│
└── docs/                      # Documentation
    ├── README.md             # Main documentation
    ├── DEPLOYMENT.md         # Production deployment
    ├── MIGRATION.md          # Streamlit→Flask comparison
    ├── TESTING.md            # Testing guide
    └── ARCHITECTURE.md       # This file
```

## Performance Characteristics

### Typical Response Times
- **Page Load**: 1-2 seconds
- **PDF Processing** (startup): 10-60 seconds
  - Depends on: Number of PDFs, size, complexity
- **Question Processing**: 5-30 seconds
  - Depends on: Model size, question complexity, document size

### Resource Usage
- **Memory**: 2-8 GB (varies with model and documents)
- **CPU**: Moderate during question processing
- **Disk**: Minimal (PDFs + vector DB cache)

### Bottlenecks
1. **LLM Processing**: Most time-consuming
2. **Vector DB Search**: Usually fast
3. **PDF Loading**: One-time cost at startup

## Future Enhancements

### Potential Improvements
- [ ] Real-time streaming responses
- [ ] Multiple document upload via UI
- [ ] User authentication and multi-user support
- [ ] Document management (add/remove PDFs)
- [ ] Export chat history
- [ ] Response rating and feedback
- [ ] Advanced RAG techniques (re-ranking, hybrid search)
- [ ] WebSocket support for live updates
- [ ] Progressive Web App (PWA) capabilities
- [ ] Dark mode theme toggle
