# HAVEN: Home Activity Vision & Event Notification

Hệ thống ADL (Activity of Daily Living) giúp nhận diện hành vi con người qua Camera/Video sử dụng AI Pose Estimation.

## 🎥 Kết quả Demo (ADL + Pose)

![ADL Demo](pose-adl-ver2.gif)

> **Màu sắc Skeleton (Bộ xương):**
> - 🔴 **Head** (Đầu) - Red
> - 💗 **Torso** (Thân) - Pink
> - 🟢 **Upper Arm** (Cánh tay trên: Vai → Khuỷu tay) - Green
> - 🟩 **Lower Arm** (Cánh tay dưới/Bàn tay: Khuỷu → Cổ tay) - Dark Green
> - 🟠 **Upper Leg** (Chân trên: Hông → Đầu gối) - Orange
> - 🟡 **Lower Leg** (Chân dưới/Bàn chân: Đầu gối → Mắt cá) - Yellow

> **Màu sắc BBox (Tư thế):**
> - 🟢 **Standing** (Đứng) - Green
> - 🔵 **Walking** (Đi bộ) - Cyan
> - 🟡 **Sitting** (Ngồi) - Yellow
> - 🔴 **Laying** (Nằm) - Red
> - ⚪ **Unknown** (Không xác định) - Gray

## 🚀 Tính năng chính
1. **Pose Classification**: Phân loại hành vi dựa trên góc xương và chuyển động.
2. **Event Detection**: Phát hiện Ngã (Fall Down), Kêu cứu (Hand Up) - *đang phát triển*.
3. **Tracking**: DeepSORT/IOU Tracking giữ ID đối tượng ổn định.
4. **Tối ưu**: Chạy mượt trên Laptop CPU (YOLO11s) và Jetson Nano (YOLO11n).

## 🛠️ Cách chạy Demo

Chạy file batch để xem kết quả test trên video mẫu:

```bash
.\pose_adl.bat
```

Sau khi chạy, nhấn phím **G** để quay màn hình (GIF), nhấn lần nữa để lưu.

---
**Bảo mật**: Sử dụng `.\.github\push.bat` để đẩy code an toàn lên GitHub.