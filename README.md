# HAVEN: Home Activity Vision & Event Notification

Hệ thống ADL (Activity of Daily Living) giúp nhận diện hành vi con người qua Camera/Video sử dụng AI Pose Estimation.

## 🎥 Kết quả Demo (ADL + Pose)

![ADL Demo](adl_output.gif)

> **Màu sắc Skeleton:**
> - 🟢 Standing (Đứng)
> - 🟡 Walking (Đi bộ)
> - 🟨 Sitting (Ngồi)
> - 🔴 Laying (Nằm)

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