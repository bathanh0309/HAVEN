# HAVEN - Smart Surveillance System

He thong giam sat camera su dung AI Pose Detection thoi gian thuc.

## Tinh nang
- Real-time Streaming RTSP (Tapo C210)
- AI Pose Detection (YOLOv8)
- Ho tro luong HD va SD
- WebSocket streaming (Low latency)
- Tu dong ket noi lai
- Toi uu cho Laptop khong GPU

## Yeu cau
- Camera ho tro RTSP (VD: Tapo C210)
- Python 3.10+
- May tinh cung mang WiFi voi camera

## Cai dat

1. Clone repository
   ```bash
   git clone https://github.com/yourusername/HAVEN.git
   cd HAVEN
   ```

2. Tao moi truong ao
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Cai dat thu vien
   ```bash
   pip install -r requirements.txt
   ```

## Cau hinh

1. Tao file .env
   ```bash
   copy .env.example .env
   ```

2. Sua file .env voi thong tin camera cua ban:
   - IP, Username, Password (Account Camera, khong phai Cloud)
   - Chinh RTSP Stream URL neu can

## Chay he thong

Su dung script tu dong:
```bash
.\run.bat
```

Hoac chay thu cong:
1. Backend: `python backend/src/main.py`
2. Frontend: `python -m http.server 8090` (tai thu muc frontend)

Truy cap: http://localhost:8090

## Bao mat (Quan trong)

- KHONG BAO GIO commit file .env len Git.
- Su dung script `.\.github\push.bat` de push code an toan. Script se tu dong kiem tra cac file nhay cam truoc khi push.

## Troubleshooting

- Loi module: Chay python tu thu muc goc hoac set PYTHONPATH.
- Loi RTSP: Kiem tra IP, User/Pass, va dam bao da bat ONVIF/RTSP tren camera app.
- Lag: Chuyen sang stream SD va giam FPS trong .env.