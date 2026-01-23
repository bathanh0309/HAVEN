/**
 * HAVEN Frontend - WebSocket Client
 * ==================================
 * Canvas-based video rendering with WebSocket streaming.
 * 
 * Features:
 * - Real-time frame rendering via WebSocket
 * - Auto-reconnect with exponential backoff
 * - Stream switching (HD/SD)
 * - FPS monitoring
 * - Connection status indicators
 */

class HAVENClient {
    constructor() {
        // WebSocket
        this.ws = null;
        this.wsUrl = this.getWebSocketURL();
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 10;
        this.reconnectDelay = 1000; // Start with 1 second
        this.reconnectTimer = null;

        // Canvas
        this.canvas = document.getElementById('videoCanvas');
        this.ctx = this.canvas.getContext('2d');

        // UI Elements
        this.statusDot = document.getElementById('statusDot');
        this.statusText = document.getElementById('statusText');
        this.videoOverlay = document.getElementById('videoOverlay');
        this.errorMessage = document.getElementById('errorMessage');

        // Stats elements
        this.fpsValue = document.getElementById('fpsValue');
        this.streamValue = document.getElementById('streamValue');
        this.uptimeValue = document.getElementById('uptimeValue');
        this.frameValue = document.getElementById('frameValue');

        // Buttons
        this.switchBtn = document.getElementById('switchBtn');
        this.reconnectBtn = document.getElementById('reconnectBtn');

        // State
        this.currentStream = 'SD';
        this.connected = false;
        this.frameCount = 0;
        this.lastFrameTime = 0;
        this.fps = 0;

        // Bind event handlers
        this.setupEventListeners();

        // Start connection
        this.connect();

        // Start health check polling
        this.startHealthCheck();
    }

    /**
     * Get WebSocket URL based on current location
     */
    getWebSocketURL() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host;
        return `${protocol}//${host}/ws/stream`;
    }

    /**
     * Setup UI event listeners
     */
    setupEventListeners() {
        // Stream switch button
        this.switchBtn.addEventListener('click', () => this.handleStreamSwitch());

        // Reconnect button
        this.reconnectBtn.addEventListener('click', () => this.handleReconnect());

        // Radio button change
        document.querySelectorAll('input[name="stream"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.currentStream = e.target.value;
            });
        });
    }

    /**
     * Connect to WebSocket
     */
    connect() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            console.log('Already connected');
            return;
        }

        console.log(`Connecting to ${this.wsUrl}...`);
        this.updateStatus('connecting', 'Connecting...');
        this.showOverlay(true);

        try {
            this.ws = new WebSocket(this.wsUrl);

            this.ws.onopen = () => this.handleOpen();
            this.ws.onmessage = (event) => this.handleMessage(event);
            this.ws.onerror = (error) => this.handleError(error);
            this.ws.onclose = (event) => this.handleClose(event);
        } catch (error) {
            console.error('WebSocket connection error:', error);
            this.scheduleReconnect();
        }
    }

    /**
     * Handle WebSocket open
     */
    handleOpen() {
        console.log('✅ WebSocket connected');
        this.connected = true;
        this.reconnectAttempts = 0;
        this.reconnectDelay = 1000;
        this.updateStatus('connected', 'Connected');
        this.showOverlay(false);
        this.hideError();

        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
    }

    /**
     * Handle incoming WebSocket messages
     */
    handleMessage(event) {
        try {
            const message = JSON.parse(event.data);

            if (message.type === 'frame') {
                this.renderFrame(message.data, message.metadata);
            } else if (message.type === 'error') {
                console.error('Server error:', message.message);
                this.showError(message.message);
            }
        } catch (error) {
            console.error('Failed to parse message:', error);
        }
    }

    /**
     * Render frame on canvas
     */
    renderFrame(base64Data, metadata) {
        const img = new Image();

        img.onload = () => {
            // Set canvas size to match image (only on first frame or size change)
            if (this.canvas.width !== img.width || this.canvas.height !== img.height) {
                this.canvas.width = img.width;
                this.canvas.height = img.height;
            }

            // Draw image
            this.ctx.drawImage(img, 0, 0);

            // Update stats
            this.frameCount++;
            this.updateStats(metadata);
        };

        img.onerror = (error) => {
            console.error('Failed to load frame:', error);
        };

        // Decode base64 to image
        img.src = 'data:image/jpeg;base64,' + base64Data;
    }

    /**
     * Update statistics display
     */
    updateStats(metadata) {
        // FPS from backend
        if (metadata.fps !== undefined) {
            this.fpsValue.textContent = metadata.fps.toFixed(1);
        }

        // Stream type
        if (metadata.stream_type) {
            this.streamValue.textContent = metadata.stream_type;
        }

        // Frame count
        this.frameValue.textContent = this.frameCount;
    }

    /**
     * Handle WebSocket error
     */
    handleError(error) {
        console.error('WebSocket error:', error);
        this.connected = false;
        this.updateStatus('disconnected', 'Connection Error');
        this.showError('Connection error. Reconnecting...');
    }

    /**
     * Handle WebSocket close
     */
    handleClose(event) {
        console.log('WebSocket closed:', event.code, event.reason);
        this.connected = false;
        this.ws = null;
        this.updateStatus('disconnected', 'Disconnected');
        this.showOverlay(true);

        // Auto-reconnect
        this.scheduleReconnect();
    }

    /**
     * Schedule reconnection with exponential backoff
     */
    scheduleReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.error('Max reconnect attempts reached');
            this.showError(`Failed to connect after ${this.maxReconnectAttempts} attempts. Click Reconnect to try again.`);
            return;
        }

        this.reconnectAttempts++;
        const delay = Math.min(this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1), 30000);

        console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
        this.updateStatus('disconnected', `Reconnecting in ${Math.round(delay / 1000)}s...`);

        this.reconnectTimer = setTimeout(() => {
            this.connect();
        }, delay);
    }

    /**
     * Handle stream switch button click
     */
    async handleStreamSwitch() {
        const selectedStream = document.querySelector('input[name="stream"]:checked').value;

        if (selectedStream === this.currentStream) {
            console.log('Already on this stream');
            return;
        }

        this.switchBtn.disabled = true;
        this.switchBtn.textContent = 'Switching...';
        this.showOverlay(true);

        try {
            const response = await fetch(`/api/stream/switch?stream=${selectedStream}`, {
                method: 'POST'
            });

            const data = await response.json();

            if (data.success) {
                console.log(`✅ Switched to ${selectedStream} stream`);
                this.currentStream = selectedStream;

                // Reconnect WebSocket to get new stream
                if (this.ws) {
                    this.ws.close();
                }

                setTimeout(() => {
                    this.connect();
                }, 500);
            } else {
                console.error('Stream switch failed:', data.error);
                this.showError('Failed to switch stream: ' + (data.error || 'Unknown error'));
            }
        } catch (error) {
            console.error('Stream switch error:', error);
            this.showError('Failed to switch stream: ' + error.message);
        } finally {
            this.switchBtn.disabled = false;
            this.switchBtn.textContent = 'Switch Stream';
        }
    }

    /**
     * Handle reconnect button click
     */
    handleReconnect() {
        console.log('Manual reconnect triggered');
        this.reconnectAttempts = 0;

        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }

        if (this.ws) {
            this.ws.close();
        }

        this.connect();
    }

    /**
     * Update connection status UI
     */
    updateStatus(state, text) {
        this.statusDot.className = 'status-dot ' + state;
        this.statusText.textContent = text;
    }

    /**
     * Show/hide video overlay
     */
    showOverlay(show) {
        if (show) {
            this.videoOverlay.classList.add('show');
        } else {
            this.videoOverlay.classList.remove('show');
        }
    }

    /**
     * Show error message
     */
    showError(message) {
        this.errorMessage.textContent = message;
        this.errorMessage.classList.add('show');
    }

    /**
     * Hide error message
     */
    hideError() {
        this.errorMessage.classList.remove('show');
    }

    /**
     * Start health check polling
     */
    startHealthCheck() {
        setInterval(async () => {
            try {
                const response = await fetch('/api/health');
                const data = await response.json();

                if (data.status === 'ok' && data.camera) {
                    // Update uptime
                    this.uptimeValue.textContent = this.formatUptime(data.camera.uptime_seconds);

                    // Update stream if backend changed it
                    if (data.camera.current_stream !== this.currentStream) {
                        this.currentStream = data.camera.current_stream;
                        document.getElementById(`stream${data.camera.current_stream}`).checked = true;
                    }
                }
            } catch (error) {
                console.error('Health check failed:', error);
            }
        }, 5000); // Poll every 5 seconds
    }

    /**
     * Format uptime seconds to human-readable string
     */
    formatUptime(seconds) {
        if (seconds < 60) {
            return Math.round(seconds) + 's';
        } else if (seconds < 3600) {
            return Math.round(seconds / 60) + 'm';
        } else {
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.round((seconds % 3600) / 60);
            return `${hours}h ${minutes}m`;
        }
    }

    /**
     * Cleanup on page unload
     */
    destroy() {
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
        }

        if (this.ws) {
            this.ws.close();
        }
    }
}

// Initialize client when DOM is ready
let client;

document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Initializing HAVEN Client...');
    client = new HAVENClient();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (client) {
        client.destroy();
    }
});