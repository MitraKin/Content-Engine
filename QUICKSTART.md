# Quick Start Guide

Get up and running with the AI PDF Reader in 5 minutes!

## Prerequisites Checklist

- [ ] Python 3.8 or higher installed
- [ ] Ollama installed ([Download here](https://ollama.ai))
- [ ] At least one Ollama model downloaded

## Step 1: Install Ollama and Pull a Model

```bash
# After installing Ollama, pull a model
ollama pull llama2

# Verify it's working
ollama list
```

## Step 2: Clone and Setup

```bash
# Clone the repository
git clone https://github.com/MitraKin/Content-Engine.git
cd Content-Engine

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 3: Add Your PDFs

```bash
# PDFs go in the Documents directory
# Sample PDFs are already included, or add your own:
cp /path/to/your/document.pdf Documents/
```

## Step 4: Run the Application

```bash
# Start the Flask server
python app.py
```

You should see output like:
```
INFO - Extracting model names from models_info
INFO - Available models: ('llama2',)
INFO - Loading PDF files...
INFO - Processing file: Documents/your-document.pdf
INFO - Vector DB created from directory
 * Running on http://0.0.0.0:5000
```

## Step 5: Use the Application

1. Open your browser to: **http://localhost:5000**
2. Select a model from the dropdown
3. Wait for "PDF Documents: ✓ Ready" in the status panel
4. Type a question and click "Send"
5. Get AI-powered answers based on your documents!

## Example Questions

```
"What is this document about?"
"Summarize the main points"
"What are the key findings?"
```

## Troubleshooting

### Issue: No models available
```bash
# Pull a model
ollama pull llama2
# Restart the app
```

### Issue: Port 5000 in use
```bash
# Edit app.py, change line 228:
app.run(debug=debug_mode, host='0.0.0.0', port=5001)
```

### Issue: PDFs not loading
```bash
# Check the Documents directory
ls Documents/
# Should see .pdf files
```

## Next Steps

- 📖 Read [TESTING.md](TESTING.md) for detailed testing
- 🚀 Read [DEPLOYMENT.md](DEPLOYMENT.md) for production setup
- 🏗️ Read [ARCHITECTURE.md](ARCHITECTURE.md) to understand the system
- 🔄 Read [MIGRATION.md](MIGRATION.md) for Streamlit comparison

## Getting Help

- Check the logs in your terminal
- See [TESTING.md](TESTING.md) for detailed troubleshooting
- Review [README.md](README.md) for full documentation

## Success!

If you can ask questions and get relevant answers, you're all set! 🎉

---

**Tip**: Start with simple questions to verify everything works, then try more complex queries.
