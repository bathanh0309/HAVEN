HAVEN — Home Activity Vision & Event Notification

```
Thành viên: Nguyễn Bá Thành
Giám sát  : Lê Phong Phú
```

## 🎯 Tổng quan dự án (Overview)
HAVEN là hệ thống giám sát hoạt động tại gia đình sử dụng trí tuệ nhân tạo để phát hiện các hành vi (ADL - Activities of Daily Living) và gửi thông báo khẩn cấp.

## 🏗️ Nguyên tắc tổ chức (Architecture Principles)

### 1. Phân tách rõ rệt (Separation of Concerns)
- **Backend**: FastAPI + Logic nghiệp vụ + Xử lý Computer Vision.
- **Frontend**: Streamlit dashboard (Phase 1) và Flutter (Phase 2).
- **Models**: Trọng số AI (`.pt`, `.onnx`) được tách biệt hoàn toàn với mã nguồn.
- **Data**: Dữ liệu runtime (logs, snapshots) không đưa lên Git.

### 2. Kiến trúc phân lớp (Backend Layers)
- `api/`: Presentation layer (REST & WebSocket).
- `core/`: Business logic engine (Capture, CV, ADL, Alerts).
- `models/`: Database model & Pydantic schemas.
- `services/`: Lớp điều phối (Orchestration).

### 3. Khả năng mở rộng (Scalability)
- Sử dụng mô hình Queue cho luồng Capture → Processing.
- Cấu hình linh hoạt qua các file YAML trong thư mục `config/`.

### 4. Dễ dàng bảo trì (Maintainability)
- Tài liệu chi tiết trong `docs/`.
- Hệ thống test riêng biệt theo cấp độ (Unit, Integration, E2E).
- Kịch bản tiện ích trong `scripts/`.

---
*Lấy cảm hứng và tiêu chuẩn tổ chức từ repository SMAC.*
