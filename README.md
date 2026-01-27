# HAVEN: Home Activity Vision & Event Notification

Hệ thống giám sát thông minh ADL (Activity of Daily Living) sử dụng AI để nhận diện hành vi, phát hiện sự cố và xâm nhập vùng cấm.

## Demo Tính Năng Mới (Ver 6)
*Tích hợp: Pose + ADL + Zone Intrusion + Object Detection*

![Demo HAVEN](backend/outputs/pose-adl-ver6.gif)

## Tính Năng Chính
1. **Pose Detection**: Nhận diện tư thế (Đứng, Đi, Ngồi, Nằm).
2. **Event Detection**: Phát hiện sự kiện (Ngã, Giơ tay cầu cứu, Ngồi xuống, Đứng lên).
3. **Zone Intrusion**: Cảnh báo khi người đi vào vùng cấm (Ví dụ: Bếp, Khu vực nguy hiểm).
4. **Object Detection**: Phát hiện vật dụng nguy hiểm (Dao, Kéo, Điện thoại).

---

## Quy Định Màu Sắc (Color Coding)

Hệ thống sử dụng mã màu để người dùng dễ dàng nhận biết trạng thái:

### 1. Trạng Thái Người (Bounding Box)
| Màu Sắc | Ý Nghĩa | Trạng Thái |
| :--- | :--- | :--- |
| 🟢 **Xanh Lá** | **BÌNH THƯỜNG** | Đứng (Standing) |
| 🔵 **Cyan** | **HOẠT ĐỘNG** | Đi lại (Walking) |
| 🟠 **Cam** | **TĨNH TẠI** | Ngồi (Sitting) |
| 🔴 **Đỏ** | **NGUY HIỂM** | Nằm (Laying), Ngã (Fall Down) |
| ⚪ **Xám** | **KHÔNG RÕ** | Chưa xác định (Unknown) |

### 2. Cảnh Báo (Alerts)
- **Vùng Cấm (Zone)**: Khung 🔴 **Đỏ** + Nền đỏ nhạt.
- **Vật Nguy Hiểm**: Khung 🔴 **Đỏ Đậm** kèm nhãn cảnh báo.

### 3. Bộ Xương (Skeleton)
Để hỗ trợ chẩn đoán tư thế chính xác:
- 🔴 **Đầu**: Đỏ (Red)
- 🟣 **Thân**: Tím (Magenta)
- 🔵 **Tay**: Cyan (Trên) & Xanh dương (Dưới)
- 🟠 **Chân**: Cam (Trên) & Xanh lá mạ (Dưới)

---

## Hướng Dẫn Sử Dụng

### 1. Chạy với Video File
Dùng để kiểm thử tính năng với video có sẵn.
```bash
.\video-pose-adl.bat
```
*Để thay đổi video:* Chỉnh sửa file `.env` dòng `TEST_VIDEO_PATH`.

### 2. Chạy với Camera RTSP
Dùng cho camera giám sát thực tế (IP Camera).
```bash
.\rtsp_pose_adl.bat
```
*Cấu hình Camera:* Chỉnh sửa file `.env` (IP, Port, User, Pass).

### Phím Tắt Điều Khiển
| Phím | Chức Năng |
| :---: | :--- |
| **Q** | Thoát chương trình |
| **Space** | Tạm dừng / Tiếp tục |
| **L** | Bật / Tắt chế độ lặp lại video |
| **G** | **Ghi hình (GIF)** - Nhấn lần 1 để bắt đầu, lần 2 để lưu |
| **H / S** | Chuyển luồng HD / SD (chỉ dùng cho RTSP) |

---
**Bảo mật**: Sử dụng `.\.github\push.bat` để đẩy code an toàn lên GitHub.