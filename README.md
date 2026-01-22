# HAVEN - Home Activity Vision & Event Notification

```
Thực hiện: Nguyễn Bá Thành
Giám sát: Lê Phong Phú
```
---

## 🎯 Tổng quan dự án (Overview)
HAVEN là hệ thống giám sát hoạt động tại gia đình sử dụng trí tuệ nhân tạo để phát hiện các hành vi (ADL - Activities of Daily Living) và gửi thông báo khẩn cấp.

![Architecture Pipeline](pipeline\pipeline.png)

Hệ thống cung cấp giải pháp streaming RTSP đơn giản và hiệu quả cho camera Tapo C210 với Backend Python FastAPI và Frontend HTML/CSS/JS thuần.

---

## 🌐 Cấu hình Mạng & Cổng (Network Ports)

Để vận hành hệ thống, vui lòng lưu ý các cổng (port) quan trọng sau:

### 1. Port 8090 - Web Dashboard (Giao diện người dùng)
*   **Mô tả**: Đây là cổng truy cập chính cho giao diện Web của hệ thống HAVEN.
*   **Chức năng**: Hiển thị luồng camera trực tiếp, các thông số trạng thái hệ thống và cảnh báo.
*   **Cách dùng**: Truy cập `http://localhost:8090` trên trình duyệt sau khi khởi chạy frontend.

### 2. Port 554 - RTSP Stream (Kết nối Camera)
*   **Mô tả**: Cổng chuẩn giao thức Real Time Streaming Protocol (RTSP) của camera Tapo C210.
*   **Chức năng**: Truyền tải dữ liệu video thô từ camera về server xử lý.
*   **Cấu hình**: `rtsp://<username>:<password>@<ip_address>:554/stream1`
    *   `stream1`: Luồng HD (1080p)
    *   `stream2`: Luồng SD (640x480) - Khuyên dùng để giảm độ trễ.

---

## 🛠️ Cài đặt & Hướng dẫn (Setup Guide)

### Cấu hình Camera (Tapo C210)
*   **IP Address**: `10.0.14.14`
*   **Username / Password**: `bathanh0309` / `bathanh0309`
*   **ONVIF Service**: `http://10.0.14.14:2020/onvif/device_service`

### 🚀 Khởi chạy nhanh (Quick Start)

#### Bước 1: Cài đặt thư viện
```bash
# Kích hoạt môi trường ảo
.venv\Scripts\activate.bat

# Cài đặt các gói cần thiết
pip install -r backend\requirements.txt
```

#### Bước 2: Khởi động Backend Server
```bash
# Chạy file batch tự động
run_camera.bat

# Hoặc chạy lệnh thủ công
python backend\stream_server.py
```

#### Bước 3: Khởi chạy Frontend (Port 8090)
Mở terminal tại thư mục `frontend` và chạy lệnh sau để khởi tạo server tại cổng 8090:

```bash
cd frontend
python -m http.server 8090
```
Sau đó truy cập: **http://localhost:8090**

---

## 📁 Cấu trúc dự án
```
HAVEN/
├── backend/
│   ├── stream_server.py      # Server xử lý luồng RTSP & WebSocket
│   └── requirements.txt      # Thư viện Python yêu cầu
├── frontend/
│   ├── index.html            # Giao diện chính
│   ├── style.css             # Giao diện Dark Mode hiện đại
│   └── app.js                # Logic xử lý Frontend
├── camera-tapo-C210/         # Tài liệu tham khảo camera
└── run_camera.bat            # Script khởi chạy nhanh
```

## 🔧 Tính năng nổi bật

### Backend
*   ✅ Thu thập luồng RTSP qua OpenCV
*   ✅ Streaming thời gian thực qua WebSockets
*   ✅ Tự động kết nối lại (Auto-reconnect)
*   ✅ Endpoint kiểm tra trạng thái (`/health`)

### Frontend
*   ✅ Giao diện Dark Mode hiện đại (Glassmorphism)
*   ✅ Hiển thị FPS và trạng thái kết nối
*   ✅ Responsive (Tương thích máy tính & điện thoại)

## 🔍 Khắc phục sự cố (Troubleshooting)

1.  **Camera không kết nối**:
    *   Kiểm tra kết nối mạng: `ping 10.0.14.14`
    *   Đảm bảo username/password đúng.
2.  **Video bị trễ (Lag)**:
    *   Chuyển sang dùng `stream2` (SD) thay vì `stream1` (HD).
    *   Kiểm tra băng thông WiFi.

---
**Created for HAVEN Project**
