# HAVEN: Home Activity Vision & Event Network

Hệ thống giám sát thông minh hỗ trợ nhận diện hành vi ADL (Activity of Daily Living), phát hiện sự cố và cảnh báo an ninh bằng AI.

### Demo Hoạt Động
![Demo HAVEN](backend/outputs/demo-AnhPhu.gif)

### Bảng Phân Tích Dữ Liệu
![Dashboard Analytics](backend/outputs/output1.png)

## Tính Năng Chính

1.  **Nhận diện tư thế (Pose Detection)**: Phân loại các trạng thái cơ thể như Đứng, Đi, Ngồi, Nằm.
2.  **Phát hiện sự kiện (Event Detection)**: Nhận biết các hành động như Ngã, Giơ tay cầu cứu, Ngồi xuống, Đứng lên.
3.  **Giám sát khu vực (Zone Intrusion)**: Cảnh báo khi có người xâm nhập vào các vùng đã thiết lập (Vùng nguy hiểm).
4.  **Phát hiện vật thể (Object Detection)**: Nhận diện các vật dụng nguy hiểm trong khung hình như Dao, Kéo, Điện thoại.
5.  **Định danh người (ReID)**: Theo dõi và duy tri ID của từng người qua nhiều camera khác nhau.

## Công Nghệ Sử Dụng

*   **Lõi AI (AI Core)**: Ultralytics YOLOv8 (Pose Classification & Object Detection).
*   **Tracking**: DeepSORT / ByteTrack (theo dõi đa đối tượng).
*   **Xử lý hình ảnh**: OpenCV, Numpy.
*   **Phân tích dữ liệu**: Pandas, Matplotlib, Seaborn.
*   **Backend & Cơ sở dữ liệu**: Python, SQLite (Lưu vết sự kiện).
*   **Công cụ khác**: FFmpeg (xử lý video), Jupyter Notebook (báo cáo).

## Hướng Dẫn Sử Dụng

### Chạy nguồn video
Sử dụng để kiểm thử với file video có sẵn.
```bash
.\sequential.bat
```

### Chạy Camera RTSP: chưa setup
Sử dụng cho hệ thống camera IP thực tế.
```bash
.\rtsp_pose_adl.bat
```

### Phím Tắt
- **Q**: Thoát chương trình
- **Space**: Tạm dừng / Tiếp tục
- **L**: Bật / Tắt chế độ lặp lại video
- **G**: Ghi hình GIF
- **H / S**: Chuyển đổi độ phân giải HD / SD