// Configuration
const BACKEND_URL = 'http://localhost:5000';
const VIDEO_FEED_URL = `${BACKEND_URL}/video_feed`;

// DOM Elements
const videoStream = document.getElementById('videoStream');
const overlay = document.getElementById('overlay');
const status = document.getElementById('status');
const statusText = status.querySelector('.status-text');
const fpsDisplay = document.getElementById('fps');
const reconnectBtn = document.getElementById('reconnectBtn');
const toggleQualityBtn = document.getElementById('toggleQualityBtn');

// State
let isConnected = false;
let frameCount = 0;
let lastTime = Date.now();
let fpsInterval;

// Initialize
function init() {
    console.log('Initializing camera stream...');
    connectToStream();
    setupEventListeners();
    startFPSCounter();
}

// Connect to video stream
function connectToStream() {
    console.log('Connecting to:', VIDEO_FEED_URL);

    updateStatus('connecting');

    // Set video source
    videoStream.src = VIDEO_FEED_URL;

    // Handle successful load
    videoStream.onload = () => {
        console.log('Stream connected successfully');
        isConnected = true;
        hideOverlay();
        updateStatus('connected');
    };

    // Handle errors
    videoStream.onerror = (error) => {
        console.error('Stream error:', error);
        isConnected = false;
        showOverlay('Lỗi kết nối. Vui lòng thử lại...');
        updateStatus('disconnected');
    };
}

// Update connection status
function updateStatus(state) {
    status.classList.remove('disconnected', 'connecting');

    switch (state) {
        case 'connected':
            statusText.textContent = 'Đã kết nối';
            break;
        case 'disconnected':
            statusText.textContent = 'Mất kết nối';
            status.classList.add('disconnected');
            break;
        case 'connecting':
            statusText.textContent = 'Đang kết nối...';
            status.classList.add('connecting');
            break;
    }
}

// Show/hide overlay
function hideOverlay() {
    overlay.classList.add('hidden');
}

function showOverlay(message) {
    overlay.classList.remove('hidden');
    const overlayText = overlay.querySelector('p');
    if (overlayText && message) {
        overlayText.textContent = message;
    }
}

// Reconnect to stream
function reconnect() {
    console.log('Reconnecting...');
    showOverlay('Đang kết nối lại...');

    // Force reload by adding timestamp
    const timestamp = new Date().getTime();
    videoStream.src = `${VIDEO_FEED_URL}?t=${timestamp}`;
}

// FPS Counter
function startFPSCounter() {
    fpsInterval = setInterval(() => {
        const now = Date.now();
        const elapsed = (now - lastTime) / 1000;
        const fps = Math.round(frameCount / elapsed);

        if (isConnected && fps > 0) {
            fpsDisplay.textContent = `${fps} fps`;
        } else {
            fpsDisplay.textContent = '--';
        }

        frameCount = 0;
        lastTime = now;
    }, 1000);
}

// Event Listeners
function setupEventListeners() {
    // Reconnect button
    reconnectBtn.addEventListener('click', () => {
        reconnect();
    });

    // Toggle quality button (placeholder - would need backend support)
    toggleQualityBtn.addEventListener('click', () => {
        alert('Chức năng chuyển chất lượng sẽ được cập nhật trong phiên bản sau');
    });

    // Track frames for FPS
    videoStream.addEventListener('load', () => {
        frameCount++;
    });

    // Check backend health
    checkBackendHealth();
    setInterval(checkBackendHealth, 30000); // Check every 30 seconds
}

// Check if backend is running
async function checkBackendHealth() {
    try {
        const response = await fetch(`${BACKEND_URL}/health`);
        const data = await response.json();

        if (data.status === 'ok') {
            console.log('Backend is healthy:', data);
        }
    } catch (error) {
        console.warn('Backend health check failed:', error);
        updateStatus('disconnected');
    }
}

// Handle page visibility
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        console.log('Page hidden - pausing updates');
    } else {
        console.log('Page visible - resuming');
        reconnect();
    }
});

// Handle window unload
window.addEventListener('beforeunload', () => {
    if (fpsInterval) {
        clearInterval(fpsInterval);
    }
});

// Start the application
init();
