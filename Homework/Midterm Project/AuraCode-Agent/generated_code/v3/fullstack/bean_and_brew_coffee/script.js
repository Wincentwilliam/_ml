document.addEventListener('DOMContentLoaded', () => {
    // Navigation Active State
    const sections = document.querySelectorAll('section');
    const navLinks = document.querySelectorAll('.nav-link');

    window.addEventListener('scroll', () => {
        let current = '';
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            if (pageYOffset >= sectionTop - 100) {
                current = section.getAttribute('id');
            }
        });

        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href').includes(current)) {
                link.classList.add('active');
            }
        });
    });

    // Mobile Menu Toggle
    const hamburger = document.querySelector('.hamburger');
    const navLinksContainer = document.querySelector('.nav-links');

    hamburger.addEventListener('click', () => {
        navLinksContainer.style.display = navLinksContainer.style.display === 'flex' ? 'none' : 'flex';
        navLinksContainer.style.flexDirection = 'column';
        navLinksContainer.style.position = 'absolute';
        navLinksContainer.style.top = '70px';
        navLinksContainer.style.left = '0';
        navLinksContainer.style.width = '100%';
        navLinksContainer.style.background = 'white';
        navLinksContainer.style.padding = '2rem';
    });
});

// Chat Widget Logic
function toggleChat() {
    const panel = document.getElementById('chatPanel');
    panel.classList.toggle('active');
}

async function handleChatKey(e) {
    if (e.key === 'Enter') sendMessage();
}

async function sendMessage() {
    const input = document.getElementById('chatInput');
    const messagesContainer = document.getElementById('chatMessages');
    const typingIndicator = document.getElementById('typingIndicator');
    const text = input.value.trim();

    if (!text) return;

    // User Message
    appendMessage('user', text);
    input.value = '';

    // Show Typing
    typingIndicator.classList.remove('hidden');
    messagesContainer.scrollTop = messagesContainer.scrollHeight;

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                message: text,
                history: [] // Simplified for this demo
            })
        });

        const data = await response.json();
        typingIndicator.classList.add('hidden');
        
        // Typing effect for bot response
        appendMessage('bot', data.response);
    } catch (error) {
        typingIndicator.classList.add('hidden');
        appendMessage('bot', "Sorry, I'm having trouble connecting to the roaster. Please try again later!");
    }
}

function appendMessage(sender, text) {
    const container = document.getElementById('chatMessages');
    const div = document.createElement('div');
    div.className = `message ${sender}`;
    
    const now = new Date();
    const time = now.getHours() + ":" + now.getMinutes().toString().padStart(2, '0');

    div.innerHTML = `
        <div class="msg-content">${text}</div>
        <span class="timestamp">${time}</span>
    `;
    
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

// Payment Modal Logic
function openPayment(name, price) {
    const modal = document.getElementById('paymentModal');
    document.getElementById('modalItemName').innerText = name;
    document.getElementById('modalItemPrice').innerText = `$${price.toFixed(2)}`;
    
    modal.style.display = 'flex';
    document.getElementById('paymentStep1').classList.remove('hidden');
    document.getElementById('paymentStep2').classList.add('hidden');
}

function closePayment() {
    document.getElementById('paymentModal').style.display = 'none';
}

// Card Formatting
document.getElementById('cardNumber').addEventListener('input', (e) => {
    let value = e.target.value.replace(/\s+/g, '').replace(/[^0-9]/gi, '');
    let formatted = value.match(/.{1,4}/g)?.join(' ') || value;
    e.target.value = formatted;
});

document.getElementById('expiryDate').addEventListener('input', (e) => {
    let value = e.target.value.replace(/[^0-9]/gi, '');
    if (value.length >= 2) {
        e.target.value = value.slice(0, 2) + '/' + value.slice(2, 4);
    }
});

async function processPayment(e) {
    e.preventDefault();
    const btn = e.target.querySelector('.pay-button');
    const originalText = btn.innerText;
    
    btn.disabled = true;
    btn.innerText = 'Processing...';

    // Fake payment delay
    await new Promise(resolve => setTimeout(resolve, 2000));

    document.getElementById('paymentStep1').classList.add('hidden');
    document.getElementById('paymentStep2').classList.remove('hidden');
    
    const orderId = Math.floor(10000 + Math.random() * 90000);
    document.getElementById('orderConfirmation').innerText = `Order #${orderId} confirmed!`;
    
    btn.disabled = false;
    btn.innerText = originalText;
}