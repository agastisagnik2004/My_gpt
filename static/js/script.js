// Configuration
const API_URL = 'http://127.0.0.1:60922';

// DOM Elements
const chatMessages = document.getElementById('chatMessages');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');

// State
let isLoading = false;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    userInput.focus();
});

// Auto-resize textarea
function autoResize(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
}

// Handle keyboard input
function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

// Send message
async function sendMessage() {
    const message = userInput.value.trim();

    if (!message || isLoading) return;

    // Clear welcome message if present
    const welcomeMsg = document.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.remove();
    }

    // Add user message
    addMessage(message, 'user');

    // Clear input
    userInput.value = '';
    userInput.style.height = 'auto';

    // Check if this is a learn command
    if (message.toLowerCase().startsWith('/learn ')) {
        await handleLearnCommand(message);
        return;
    }

    // Show typing indicator
    const typingId = showTypingIndicator();

    // Disable input
    setLoading(true);

    try {
        console.log('Sending request to:', `${API_URL}/query`);
        const response = await fetch(`${API_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: message,
                top_k: 3
            })
        });

        console.log('Response status:', response.status);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log('Response data:', data);

        // Remove typing indicator
        removeTypingIndicator(typingId);

        // Add bot response
        addBotMessage(data);

    } catch (error) {
        console.error('Error:', error);
        removeTypingIndicator(typingId);
        addErrorMessage('Failed to connect to the server. Make sure the API is running on ' + API_URL);
    } finally {
        setLoading(false);
        userInput.focus();
    }
}

// Handle /learn command
async function handleLearnCommand(message) {
    const typingId = showTypingIndicator();
    setLoading(true);

    try {
        // Parse: /learn topic: description
        const learnContent = message.substring(7).trim(); // Remove '/learn '
        const colonIndex = learnContent.indexOf(':');

        if (colonIndex === -1) {
            removeTypingIndicator(typingId);
            addBotMessage({
                answer: "**Invalid Format** ❌\n\nPlease use the format:\n`/learn [topic]: [description]`\n\nExample: `/learn Python: Python is a programming language known for its simplicity.`",
                sources: [],
                latency_ms: 0
            });
            return;
        }

        const topic = learnContent.substring(0, colonIndex).trim();
        const description = learnContent.substring(colonIndex + 1).trim();

        if (!topic || !description || description.length < 10) {
            removeTypingIndicator(typingId);
            addBotMessage({
                answer: "**Please provide more detail** 📝\n\nThe topic and description are required. Description should be at least 10 characters.\n\nExample: `/learn Python: Python is a high-level programming language used for web development, data science, and automation.`",
                sources: [],
                latency_ms: 0
            });
            return;
        }

        console.log('Sending learn request to:', `${API_URL}/learn`);
        const response = await fetch(`${API_URL}/learn`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ topic, description })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        removeTypingIndicator(typingId);
        addBotMessage({
            answer: data.answer,
            sources: [],
            latency_ms: 0
        });

    } catch (error) {
        console.error('Error:', error);
        removeTypingIndicator(typingId);
        addErrorMessage('Failed to save the topic. Please try again.');
    } finally {
        setLoading(false);
        userInput.focus();
    }
}

// Ask predefined question
function askQuestion(question) {
    userInput.value = question;
    sendMessage();
}

// Add user message
function addMessage(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;

    const avatar = sender === 'user' ? '👤' : '🤖';
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    messageDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            <div class="message-bubble">
                ${escapeHtml(text)}
            </div>
            <div class="message-meta">
                <span>${time}</span>
            </div>
        </div>
    `;

    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

// Add bot message with sources
function addBotMessage(data) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot';

    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    // Format the answer with markdown-like formatting
    const formattedAnswer = formatAnswer(data.answer);

    // Build sources HTML
    let sourcesHtml = '';
    if (data.sources && data.sources.length > 0) {
        sourcesHtml = `
            <div class="sources">
                <div class="sources-title">📚 Sources</div>
                <div class="sources-list">
                    ${data.sources.map((s, i) => `<div class="source-item">${i + 1}. ${escapeHtml(s)}</div>`).join('')}
                </div>
            </div>
        `;
    }

    // Latency badge
    const latencyHtml = data.latency_ms !== undefined ?
        `<span class="latency-badge">⚡ ${data.latency_ms}ms</span>` : '';

    messageDiv.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content">
            <div class="message-bubble">
                ${formattedAnswer}
                ${sourcesHtml}
            </div>
            <div class="message-meta">
                <span>${time}</span>
                ${latencyHtml}
            </div>
        </div>
    `;

    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

// Format answer text
function formatAnswer(text) {
    // Convert **text** to bold
    let formatted = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

    // Convert newlines to paragraphs
    formatted = formatted.split('\n\n').map(p => `<p>${p}</p>`).join('');

    // Convert single newlines to breaks within paragraphs
    formatted = formatted.replace(/\n/g, '<br>');

    return formatted;
}

// Add error message
function addErrorMessage(text) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = text;
    chatMessages.appendChild(errorDiv);
    scrollToBottom();
}

// Show typing indicator
function showTypingIndicator() {
    const id = 'typing-' + Date.now();
    const typingDiv = document.createElement('div');
    typingDiv.id = id;
    typingDiv.className = 'message bot';
    typingDiv.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content">
            <div class="message-bubble">
                <div class="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        </div>
    `;
    chatMessages.appendChild(typingDiv);
    scrollToBottom();
    return id;
}

// Remove typing indicator
function removeTypingIndicator(id) {
    const typingDiv = document.getElementById(id);
    if (typingDiv) {
        typingDiv.remove();
    }
}

// Set loading state
function setLoading(loading) {
    isLoading = loading;
    sendBtn.disabled = loading;
    userInput.disabled = loading;
}

// Scroll to bottom
function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Clear chat
function clearChat() {
    chatMessages.innerHTML = `
        <div class="welcome-message">
            <div class="welcome-icon">
                <img src="static/images/favicon.jpg" alt="AI Assistant Icon">
            </div>
            <h2>Welcome to AI Assistant</h2>
            <p>Ask me anything about Machine Learning, Neural Networks, Deep Learning, and more!</p>
            <div class="suggestion-chips">
                <button class="chip" onclick="askQuestion('What is machine learning?')">What is machine learning?</button>
                <button class="chip" onclick="askQuestion('Explain neural networks')">Explain neural networks</button>
                <button class="chip" onclick="askQuestion('What is overfitting?')">What is overfitting?</button>
            </div>
        </div>
    `;
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}