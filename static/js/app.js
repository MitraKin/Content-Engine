// DOM elements
const messagesContainer = document.getElementById('messages');
const questionInput = document.getElementById('questionInput');
const sendBtn = document.getElementById('sendBtn');
const clearBtn = document.getElementById('clearBtn');
const modelSelect = document.getElementById('modelSelect');

// Load messages on page load
document.addEventListener('DOMContentLoaded', function() {
    loadMessages();
    
    // Add event listeners
    sendBtn.addEventListener('click', sendQuestion);
    clearBtn.addEventListener('click', clearHistory);
    
    questionInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendQuestion();
        }
    });
});

// Load existing messages from server
async function loadMessages() {
    try {
        const response = await fetch('/api/messages');
        const data = await response.json();
        
        if (data.messages && data.messages.length > 0) {
            data.messages.forEach(message => {
                displayMessage(message.content, message.role);
            });
        }
    } catch (error) {
        console.error('Error loading messages:', error);
    }
}

// Display a message in the chat
function displayMessage(content, role) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'user' ? '😎' : '🤖';
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    // IMPORTANT: Only plain text should be passed as 'content'.
    // If rich formatting (e.g., HTML/Markdown) is ever supported, sanitize 'content' before inserting.
    messageContent.textContent = content;
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(messageContent);
    
    messagesContainer.appendChild(messageDiv);
    
    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Display error message
function displayError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = message;
    
    messagesContainer.appendChild(errorDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    // Remove error after 5 seconds
    setTimeout(() => {
        errorDiv.remove();
    }, 5000);
}

// Send question to the server
async function sendQuestion() {
    const question = questionInput.value.trim();
    const model = modelSelect.value;
    
    if (!question) {
        return;
    }
    
    if (!model) {
        displayError('Please select a model first');
        return;
    }
    
    // Display user message
    displayMessage(question, 'user');
    
    // Clear input
    questionInput.value = '';
    
    // Disable send button and show loading state
    sendBtn.disabled = true;
    const originalText = sendBtn.textContent;
    // Remove all children from sendBtn
    while (sendBtn.firstChild) {
        sendBtn.removeChild(sendBtn.firstChild);
    }
    // Add loading spinner
    const loadingSpan = document.createElement('span');
    loadingSpan.className = 'loading';
    sendBtn.appendChild(loadingSpan);
    
    try {
        const response = await fetch('/api/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: question,
                model: model
            })
        });
        
        const data = await response.json();
        
        if (response.ok && data.success) {
            // Display assistant response
            displayMessage(data.answer, 'assistant');
        } else {
            // Display error
            displayError(data.error || 'An error occurred while processing your question');
        }
    } catch (error) {
        console.error('Error sending question:', error);
        displayError('Failed to communicate with the server. Please try again.');
    } finally {
        // Re-enable send button
        sendBtn.disabled = false;
        sendBtn.textContent = originalText;
    }
}

// Clear chat history
async function clearHistory() {
    if (!confirm('Are you sure you want to clear the chat history?')) {
        return;
    }
    
    try {
        const response = await fetch('/api/clear', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        if (response.ok) {
            // Clear messages from UI
            messagesContainer.innerHTML = '';
        } else {
            displayError('Failed to clear chat history');
        }
    } catch (error) {
        console.error('Error clearing history:', error);
        displayError('Failed to clear chat history');
    }
}
