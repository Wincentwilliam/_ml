'use strict';

let chatHistory = [];

document.addEventListener('DOMContentLoaded', () => {
    initNavbar();
    initTabs();
    initScrollAnimations();
    initCounters();
});

function initNavbar() {
    const navbar = document.querySelector('.navbar');
    const navLinks = document.querySelectorAll('.nav-link');

    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }

        let currentSection = '';
        document.querySelectorAll('section').forEach(section => {
            const sectionTop = section.offsetTop;
            if (pageYOffset >= sectionTop - 100) {
                currentSection = section.getAttribute('id');
            }
        });

        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href').substring(1) === currentSection) {
                link.classList.add('active');
            }
        });
    });
}

function initTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const menuCards = document.querySelectorAll('.menu-card');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.dataset.tabContent;

            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            menuCards.forEach(card => {
                if (card.dataset.tab === targetTab) {
                    card.style.display = 'block';
                } else {
                    card.style.display = 'none';
                }
            });
        });
    });
}

function initScrollAnimations() {
    const observerOptions = { threshold: 0.1 };
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, observerOptions);

    document.querySelectorAll('.fade-in').forEach(el => observer.observe(el));
}

function initCounters() {
    document.querySelectorAll('.stat-number').forEach(el => animateCounter(el));
}

function animateCounter(el) {
    const target = parseInt(el.dataset.target);
    let count = 0;
    const speed = target / 50;

    const updateCount = () => {
        if (count < target) {
            count += speed;
            el.innerText = Math.ceil(count);
            setTimeout(updateCount, 30);
        } else {
            el.innerText = target;
        }
    };
    updateCount();
}

function toggleChat() {
    const panel = document.querySelector('.chat-panel');
    const chatIcon = document.getElementById('chat-icon');
    const closeIcon = document.getElementById('close-icon');
    
    panel.classList.toggle('open');
    chatIcon.style.display = panel.classList.contains('open') ? 'none' : 'block';
    closeIcon.style.display = panel.classList.contains('open') ? 'block' : 'none';
}

async function sendMessage() {
    const input = document.getElementById('chatInput');
    const text = input.value.trim();
    if (!text) return;

    addMessage('user', text);
    input.value = '';
    
    const typingIndicator = document.getElementById('chatTyping');
    typingIndicator.style.display = 'block';

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text, history: chatHistory })
        });

        if (!response.ok) throw new Error('Network error');
        
        const data = await response.json();
        addMessage('bot', data.response);
    } catch (error) {
        addMessage('bot', 'Sorry, I am having trouble connecting. Please try again later!');
    } finally {
        typingIndicator.style.display = 'none';
    }
}

function addMessage(role, text) {
    const chatBox = document.getElementById('chatBox');
    const msgDiv = document.createElement('div');
    msgDiv.className = `chat-msg ${role}`;
    
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    
    msgDiv.innerHTML = `
        <div class="msg-bubble">
            ${text}
            <span class="msg-time">${time}</span>
        </div>
    `;
    
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
    
    chatHistory.push({ role, content: text });
}

function openPayment(itemName, price) {
    const modal = document.getElementById('paymentModal');
    const summary = document.getElementById('orderSummary');
    summary.innerText = `${itemName} - ${price}`;
    modal.style.display = 'flex';
}

function closePayment() {
    document.getElementById('paymentModal').style.display = 'none';
}

function processPayment(event) {
    event.preventDefault();
    const btn = event.target;
    const originalText = btn.innerText;
    
    btn.disabled = true;
    btn.innerText = 'Processing...';

    setTimeout(() => {
        btn.disabled = false;
        btn.innerText = originalText;
        document.getElementById('paymentModal').style.display = 'none';
        
        const successModal = document.getElementById('modalSuccess');
        const orderNum = 'BNB-' + Math.floor(Math.random() * 90000 + 10000);
        successModal.querySelector('.order-id').innerText = orderNum;
        successModal.style.display = 'flex';
        
        spawnConfetti();
    }, 2000);
}

function spawnConfetti() {
    const container = document.getElementById('confetti');
    const colors = ['#f39c12', '#e74c3c', '#9b59b6', '#3498db', '#2ecc71'];
    
    for (let i = 0; i < 30; i++) {
        const confetti = document.createElement('div');
        confetti.className = 'confetti-piece';
        confetti.style.backgroundColor = colors[Math.floor(Math.random() * colors.length)];
        confetti.style.left = Math.random() * 100 + 'vw';
        confetti.style.animationDelay = Math.random() * 2 + 's';
        confetti.style.animationDuration = (Math.random() * 3 + 2) + 's';
        container.appendChild(confetti);
        
        setTimeout(() => confetti.remove(), 5000);
    }
}

function formatCard(input) {
    let value = input.value.replace(/\D/g, '');
    let formatted = '';
    for (let i = 0; i < value.length; i++) {
        if (i > 0 && i % 4 === 0) formatted += ' ';
        formatted += value[i];
    }
    input.value = formatted.substring(0, 19);
}

function formatExpiry(input) {
    let value = input.value.replace(/\D/g, '');
    if (value.length > 2) {
        input.value = value.substring(0, 2) + '/' + value.substring(2, 4);
    } else {
        input.value = value;
    }
}

// Modal overlay click handler
window.onclick = function(event) {
    const modal = document.getElementById('paymentModal');
    if (event.target == modal) {
        closePayment();
    }
};