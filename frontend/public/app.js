// ==========================================
// HAVEN Frontend - WebSocket + Canvas
// ==========================================

const BACKEND_URL = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000/ws/stream';
const AUTO_RECONNECT_INTERVAL = 3000;
const EVENT_LOG_COOLDOWN = 20000; // 20 giây - thời gian chờ giữa các lần log cùng 1 sự kiện
const STORAGE_KEY = 'haven_event_logs';

// DOM Elements
const canvas = document.getElementById('videoCanvas');
const ctx = canvas ? canvas.getContext('2d') : null;
const overlay = document.getElementById('overlay');
const status = document.getElementById('status');
const statusText = status ? status.querySelector('.status-text') : null;
const fpsDisplay = document.getElementById('fps');
const screenshotBtn = document.getElementById('screenshotBtn');
const exportCsvBtn = document.getElementById('exportCsvBtn');
const resetCsvBtn = document.getElementById('resetCsvBtn');
const logList = document.getElementById('logList');
const eventCounter = document.getElementById('eventCounter');

// State
let ws = null;
let isConnected = false;
let frameCount = 0;
let lastTime = Date.now();
let fpsInterval = null;
let autoReconnectTimer = null;
let reconnectAttempts = 0;
let eventCount = 0;
let lastFrame = null;
let loggedEvents = new Map();
let eventLogs = [];

// Skeleton connections (COCO format)
const SKELETON = [
    [5, 7], [7, 9], [6, 8], [8, 10],
    [5, 6], [5, 11], [6, 12],
    [11, 12], [11, 13], [13, 15],
    [12, 14], [14, 16]
];

// Mảng màu sắc để vẽ cho các Track ID khác nhau
const TRACK_COLORS = [
    '#00ff00', '#ff0000', '#0000ff', '#ffff00', '#ff00ff',
    '#00ffff', '#ffa500', '#800080', '#008000', '#ffc0cb'
];

function getTrackColor(trackId) {
    // Lấy màu dựa trên ID (xoay vòng)
    return TRACK_COLORS[trackId % TRACK_COLORS.length];
}

// Initialize - Khởi tạo ứng dụng
function init() {
    console.log('🚀 Initializing HAVEN...');
    if (!canvas || !ctx) {
        console.error('❌ Canvas not found');
        showOverlay('Lỗi: Không tìm thấy Canvas');
        return;
    }
    loadLogsFromStorage();
    connectToWebSocket();
    setupEventListeners();
    startFPSCounter();
}

// Load logs from localStorage - Tải nhật ký từ bộ nhớ trình duyệt
function loadLogsFromStorage() {
    try {
        const stored = localStorage.getItem(STORAGE_KEY);
        if (stored) {
            eventLogs = JSON.parse(stored);
            eventCount = eventLogs.length;
            if (eventCounter) eventCounter.textContent = eventCount;

            // Render stored logs
            if (logList && eventLogs.length > 0) {
                logList.innerHTML = '';
                // Show last 100 logs (newest first)
                const logsToShow = eventLogs.slice(-100).reverse();
                logsToShow.forEach(log => {
                    renderLogItem(log.time, log.label, log.confidence, log.trackId);
                });
            }
            console.log(`📋 Loaded ${eventLogs.length} events from storage`);
        }
    } catch (e) {
        console.error('Error loading logs:', e);
    }
}

// Save logs to localStorage - Lưu nhật ký vào bộ nhớ trình duyệt
function saveLogsToStorage() {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(eventLogs));
    } catch (e) {
        console.error('Error saving logs:', e);
    }
}

// WebSocket Connection - Kết nối WebSocket
function connectToWebSocket() {
    console.log('🔌 Connecting to:', WS_URL);
    updateStatus('connecting');
    showOverlay('Đang kết nối...');

    try {
        ws = new WebSocket(WS_URL);

        ws.onopen = () => {
            console.log('✅ WebSocket connected');
            isConnected = true;
            reconnectAttempts = 0;
            stopAutoReconnect();
            hideOverlay();
            updateStatus('connected');
        };

        ws.onmessage = (event) => {
            handleFrame(event.data);
        };

        ws.onerror = (error) => {
            console.error('❌ WebSocket error:', error);
            isConnected = false;
            updateStatus('disconnected');
        };

        ws.onclose = () => {
            console.warn('🔌 WebSocket closed');
            isConnected = false;
            updateStatus('disconnected');
            showOverlay('Mất kết nối...');
            startAutoReconnect();
        };

    } catch (error) {
        console.error('❌ WS create error:', error);
        startAutoReconnect();
    }
}

// Frame Processing - Xử lý khung hình nhận được
function handleFrame(data) {
    try {
        const payload = JSON.parse(data);
        if (payload.type !== 'frame') return;

        frameCount++;
        const img = new Image();

        img.onload = () => {
            if (canvas.width !== img.width || canvas.height !== img.height) {
                canvas.width = img.width;
                canvas.height = img.height;
            }

            ctx.drawImage(img, 0, 0);
            lastFrame = img;

            // Handle detections from different formats
            let detections = [];
            if (payload.metadata) {
                if (payload.metadata.detections) {
                    detections = payload.metadata.detections;
                } else if (payload.detections) {
                    detections = payload.detections;
                }
            } else if (payload.detections) {
                detections = payload.detections;
            }

            if (detections && detections.length > 0) {
                drawDetections(detections);
                logDetections(detections);
            }
        };

        img.src = 'data:image/jpeg;base64,' + payload.data;

    } catch (error) {
        console.error('❌ Frame error:', error);
    }
}

// Draw Detections - Vẽ box và nhãn
function drawDetections(detections) {
    if (!detections || detections.length === 0) return;

    detections.forEach(det => {
        const trackId = det.track_id || det.id || 1;
        const confidence = det.confidence || det.conf || 0;
        const label = det.class || det.label || 'person';
        const color = getTrackColor(trackId);

        // Handle different bbox formats
        let x1, y1, w, h;
        if (det.bbox && Array.isArray(det.bbox)) {
            const [x, y, bw, bh] = det.bbox;
            x1 = x - bw / 2;
            y1 = y - bh / 2;
            w = bw;
            h = bh;
        } else if (det.x !== undefined) {
            // x, y, width, height format
            x1 = det.x - (det.width || 0) / 2;
            y1 = det.y - (det.height || 0) / 2;
            w = det.width || 0;
            h = det.height || 0;
        } else {
            return;
        }

        // Draw bbox
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, w, h);

        // Draw label
        const labelText = `${label} ${Math.round(confidence * 100)}%`;
        ctx.font = 'bold 12px Arial';
        const textWidth = ctx.measureText(labelText).width;
        ctx.fillStyle = 'rgba(0,0,0,0.7)';
        ctx.fillRect(x1, y1 - 18, textWidth + 8, 16);
        ctx.fillStyle = color;
        ctx.fillText(labelText, x1 + 4, y1 - 5);

        // Draw skeleton if available
        if (det.keypoints && det.keypoints.length > 0) {
            drawSkeleton(det.keypoints, color);
        }
    });
}

// Draw Skeleton - Vẽ khung xương (Pose)
function drawSkeleton(keypoints, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;

    SKELETON.forEach(([i, j]) => {
        const kp1 = keypoints[i];
        const kp2 = keypoints[j];
        if (kp1 && kp2 && (kp1.confidence || kp1.c || 0) > 0.5 && (kp2.confidence || kp2.c || 0) > 0.5) {
            ctx.beginPath();
            ctx.moveTo(kp1.x, kp1.y);
            ctx.lineTo(kp2.x, kp2.y);
            ctx.stroke();
        }
    });

    keypoints.forEach((kp) => {
        const conf = kp.confidence || kp.c || 0;
        if (conf > 0.5) {
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(kp.x, kp.y, 4, 0, 2 * Math.PI);
            ctx.fill();
            ctx.strokeStyle = 'white';
            ctx.lineWidth = 1;
            ctx.stroke();
        }
    });
}

// Log Detections (20s cooldown) - Ghi nhật ký phát hiện (có thời gian chờ)
function logDetections(detections) {
    if (!detections || detections.length === 0) return;

    const now = Date.now();

    detections.forEach(det => {
        const label = det.class || det.label || 'person';
        const conf = Math.round((det.confidence || det.conf || 0) * 100);
        const trackId = det.track_id || det.id || 1;

        // Skip if confidence is 0
        if (conf === 0) return;

        // Check cooldown
        const lastLogTime = loggedEvents.get(trackId) || 0;

        if (now - lastLogTime >= EVENT_LOG_COOLDOWN) {
            loggedEvents.set(trackId, now);
            const timeStr = formatTime(new Date());
            addEventLog(timeStr, label, conf, trackId);
        }
    });
}

// Format time - Định dạng thời gian hiển thị
function formatTime(date) {
    const h = String(date.getHours()).padStart(2, '0');
    const m = String(date.getMinutes()).padStart(2, '0');
    const d = String(date.getDate()).padStart(2, '0');
    const mo = String(date.getMonth() + 1).padStart(2, '0');
    const y = date.getFullYear();
    return `${h}:${m} ${d}/${mo}/${y}`;
}

// Render a log item - Hiển thị 1 dòng log lên UI
function renderLogItem(time, label, confidence, trackId) {
    const logItem = document.createElement('div');
    logItem.className = 'log-item';
    logItem.style.cssText = 'display: grid; grid-template-columns: 1fr 1fr 0.5fr; gap: 0.5rem; padding: 0.6rem 0.5rem; border-bottom: 1px solid var(--border); align-items: center;';
    logItem.innerHTML = `
        <span style="color: var(--text-dim); font-size: 0.8rem;">${time}</span>
        <span style="font-weight: 600;">${label} #${trackId}</span>
        <span style="color: var(--primary); text-align: right;">${confidence}%</span>
    `;
    logList.insertBefore(logItem, logList.firstChild);
}

// Add Event to Log - Thêm sự kiện mới vào danh sách
function addEventLog(time, label, confidence, trackId) {
    if (!logList) return;

    // Clear placeholder on first log
    if (eventCount === 0 && logList.querySelector('div[style*="text-align: center"]')) {
        logList.innerHTML = '';
    }

    eventCount++;
    if (eventCounter) eventCounter.textContent = eventCount;

    // Store
    eventLogs.push({ time, label, trackId, confidence });
    saveLogsToStorage();

    // Render
    renderLogItem(time, label, confidence, trackId);

    // Keep max 100 in DOM
    while (logList.children.length > 100) {
        logList.removeChild(logList.lastChild);
    }
}

// Export CSV - Xuất nhật ký ra file CSV
function exportCsv() {
    if (eventLogs.length === 0) {
        alert('Không có dữ liệu để xuất');
        return;
    }

    let csv = 'Thời gian,Nhãn,Track ID,Confidence (%)\n';
    eventLogs.forEach(log => {
        csv += `"${log.time}","${log.label}",${log.trackId},${log.confidence}\n`;
    });

    const blob = new Blob(['\ufeff' + csv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const now = new Date();
    const filename = `HAVEN_Events_${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, '0')}${String(now.getDate()).padStart(2, '0')}_${String(now.getHours()).padStart(2, '0')}${String(now.getMinutes()).padStart(2, '0')}.csv`;

    link.download = filename;
    link.href = URL.createObjectURL(blob);
    link.click();

    console.log('📥 CSV exported:', filename, `(${eventLogs.length} events)`);
}

// Reset logs - Xóa toàn bộ nhật ký
function resetLogs() {
    if (!confirm('Bạn có chắc muốn xóa toàn bộ nhật ký sự kiện?')) return;

    eventLogs = [];
    eventCount = 0;
    loggedEvents.clear();
    localStorage.removeItem(STORAGE_KEY);

    if (eventCounter) eventCounter.textContent = '0';
    if (logList) {
        logList.innerHTML = '<div style="text-align: center; color: var(--text-dim); padding: 2rem;">Đang chờ hoạt động...</div>';
    }

    console.log('🗑️ Logs reset');
}

// Screenshot - Chụp ảnh màn hình canvas hiện tại
function takeScreenshot() {
    if (!canvas || !lastFrame) {
        alert('Không có frame để chụp');
        return;
    }

    const link = document.createElement('a');
    const now = new Date();
    const filename = `HAVEN_${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, '0')}${String(now.getDate()).padStart(2, '0')}_${String(now.getHours()).padStart(2, '0')}${String(now.getMinutes()).padStart(2, '0')}${String(now.getSeconds()).padStart(2, '0')}.jpg`;

    link.download = filename;
    link.href = canvas.toDataURL('image/jpeg', 0.9);
    link.click();

    console.log('📷 Screenshot saved:', filename);
}

// UI Updates - Cập nhật trạng thái giao diện
function updateStatus(state) {
    if (!status) return;
    status.classList.remove('disconnected', 'connected');

    switch (state) {
        case 'connected':
            if (statusText) statusText.textContent = 'Đã kết nối';
            status.classList.add('connected');
            break;
        case 'disconnected':
            if (statusText) statusText.textContent = 'Mất kết nối';
            status.classList.add('disconnected');
            break;
        case 'connecting':
            if (statusText) statusText.textContent = 'Đang kết nối...';
            status.classList.add('disconnected');
            break;
    }
}

function hideOverlay() {
    if (overlay) overlay.classList.add('hidden');
}

function showOverlay(message) {
    if (!overlay) return;
    overlay.classList.remove('hidden');
    const p = overlay.querySelector('p');
    if (p && message) p.textContent = message;
}

// Reconnection - Hàm kết nối lại thủ công
function reconnect() {
    if (ws) {
        ws.close();
        ws = null;
    }
    reconnectAttempts++;
    connectToWebSocket();
}

// Bắt đầu timer tự động kết nối lại
function startAutoReconnect() {
    if (autoReconnectTimer) clearTimeout(autoReconnectTimer);
    const delay = Math.min(AUTO_RECONNECT_INTERVAL * Math.pow(1.5, reconnectAttempts), 30000);
    autoReconnectTimer = setTimeout(() => reconnect(), delay);
}

// Dừng timer tự động kết nối lại
function stopAutoReconnect() {
    if (autoReconnectTimer) {
        clearTimeout(autoReconnectTimer);
        autoReconnectTimer = null;
    }
}

// FPS Counter - Bộ đếm FPS
function startFPSCounter() {
    fpsInterval = setInterval(() => {
        const now = Date.now();
        const elapsed = (now - lastTime) / 1000;
        const fps = Math.round(frameCount / elapsed);

        if (fpsDisplay) {
            fpsDisplay.textContent = isConnected ? `📊 ${fps} fps` : '📊 -- fps';
        }

        frameCount = 0;
        lastTime = now;
    }, 1000);
}

// Event Listeners - Thiết lập các sự kiện DOM
function setupEventListeners() {
    if (screenshotBtn) {
        screenshotBtn.addEventListener('click', takeScreenshot);
    }

    if (exportCsvBtn) {
        exportCsvBtn.addEventListener('click', exportCsv);
    }

    if (resetCsvBtn) {
        resetCsvBtn.addEventListener('click', resetLogs);
    }

    document.addEventListener('visibilitychange', () => {
        if (!document.hidden && !isConnected) {
            reconnect();
        }
    });
}

// Cleanup
window.addEventListener('beforeunload', () => {
    if (fpsInterval) clearInterval(fpsInterval);
    if (ws) ws.close();
});

// Start
init();
