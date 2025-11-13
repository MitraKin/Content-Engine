# Testing Guide

## Prerequisites
Before testing, ensure you have:
1. ✅ Python 3.8+ installed
2. ✅ Ollama installed and running ([Download Ollama](https://ollama.ai))
3. ✅ At least one Ollama model pulled (e.g., `ollama pull llama2`)
4. ✅ PDF files in the `Documents/` directory

## Quick Start Testing

### 1. Verify Ollama is Running
```bash
# Check Ollama service
ollama list

# You should see your installed models
# Example output:
# NAME              ID              SIZE      MODIFIED
# llama2:latest     abc123...       3.8 GB    2 days ago
```

### 2. Install Dependencies
```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Run the Application
```bash
# Development mode (default - debug off)
python app.py

# Development mode with debug (for development only)
export FLASK_DEBUG=true
python app.py
```

### 4. Access the Application
Open your browser and navigate to:
```
http://localhost:5000
```

## Testing Checklist

### ✅ Initial Load
- [ ] Page loads without errors
- [ ] Models are listed in the dropdown
- [ ] Status panel shows "PDF Documents: ✓ Ready"
- [ ] Status panel shows correct number of models

### ✅ Chat Functionality
- [ ] Can type in the input field
- [ ] Can select a model from dropdown
- [ ] Clicking "Send" submits the question
- [ ] User message appears in chat
- [ ] Loading indicator appears while processing
- [ ] Bot response appears after processing
- [ ] Responses are relevant to PDF content

### ✅ Error Handling
- [ ] Appropriate error shown if no model selected
- [ ] Appropriate error shown if question is empty
- [ ] Network errors are handled gracefully

### ✅ Clear History
- [ ] "Clear" button shows confirmation dialog
- [ ] Clicking "OK" clears all messages
- [ ] Clicking "Cancel" keeps messages

### ✅ UI/UX
- [ ] Layout is responsive on different screen sizes
- [ ] Messages are properly formatted
- [ ] Scrolling works correctly
- [ ] Buttons provide visual feedback on hover
- [ ] Loading states are clear

## Sample Test Questions

Try these questions based on the sample PDF files (if using provided documents):

### For Financial Documents (Google, Tesla, Uber 10-K):
1. "What was the total revenue for the year?"
2. "What are the main risk factors mentioned?"
3. "Summarize the business operations"
4. "What are the key financial metrics?"

### General Testing:
1. "What is this document about?"
2. "Summarize the main points"
3. "What are the key findings?"

## Troubleshooting

### Problem: "No PDF files found"
**Solution**: Ensure PDF files are in the `Documents/` directory

### Problem: "No models available"
**Solution**: 
1. Check if Ollama is running: `ollama list`
2. Pull a model: `ollama pull llama2`
3. Restart the Flask app

### Problem: Page doesn't load
**Solution**: 
1. Check console for errors
2. Verify Flask is running on port 5000
3. Check firewall settings

### Problem: Questions timeout or fail
**Solution**:
1. Check Ollama is responding: `ollama run llama2 "test"`
2. Verify PDF processing completed successfully (check logs)
3. Try a simpler question first

### Problem: Port 5000 already in use
**Solution**:
```bash
# Use a different port
# Edit app.py line 228 to change port:
app.run(debug=debug_mode, host='0.0.0.0', port=5001)
```

## Performance Testing

### Expected Response Times
- **Page Load**: < 2 seconds
- **PDF Processing** (on startup): 10-60 seconds depending on PDF size
- **Question Processing**: 5-30 seconds depending on:
  - Model size
  - Question complexity
  - Document size

### Resource Usage
Monitor with:
```bash
# Check Python process
top -p $(pgrep -f "python app.py")

# Check Ollama process
top -p $(pgrep -f ollama)
```

## API Testing

You can also test the API directly:

### Get Status
```bash
curl http://localhost:5000/api/status
```

### Get Models
```bash
curl http://localhost:5000/api/models
```

### Ask Question
```bash
curl -X POST http://localhost:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this about?", "model": "llama2"}'
```

### Clear History
```bash
curl -X POST http://localhost:5000/api/clear \
  -H "Content-Type: application/json"
```

## Automated Testing (Future)

While this repository doesn't include automated tests yet, you could add:
- Unit tests for core functions
- Integration tests for API endpoints
- E2E tests for the UI

Example structure:
```
tests/
├── test_api.py
├── test_rag.py
└── test_ui.py
```

## Reporting Issues

If you encounter issues:
1. Check the console logs (browser and terminal)
2. Verify all prerequisites are met
3. Try the troubleshooting steps above
4. Report issues with:
   - Error messages
   - Steps to reproduce
   - System information (OS, Python version, Ollama version)

## Success Criteria

The application is working correctly when:
- ✅ PDFs load successfully on startup
- ✅ Models are detected and listed
- ✅ Questions receive relevant answers
- ✅ Chat history persists during session
- ✅ UI is responsive and user-friendly
- ✅ No security warnings from CodeQL
- ✅ Logs show successful processing
