# Task: Triển khai Hệ thống ADL (Activity of Daily Living)

## Phase 1: Core ADL Pipeline (Video Test)
- [x] Thiết kế kiến trúc module (Pose -> Features -> Rules -> State Machine)
- [x] Chuyển [ai_engine.py](file:///d:/HAVEN/backend/src/core/ai_engine.py) sang hỗ trợ trả về keypoints đầy đủ (hiện tại đang tối ưu hóa bounding box)
- [x] Implement [PostureClassifier](file:///d:/HAVEN/backend/src/adl/inference.py#12-160): Classify Standing, Sitting, Laying dựa trên Keypoints
- [x] Implement `FeatureExtractor`: Velocity, Torso Angle, Zone Membership check
- [x] Implement [RuleEngine](file:///d:/HAVEN/backend/src/adl/rules.py#14-104): State Machine cho Fall Down, Bed Exit, v.v.
- [x] Tạo script [test_adl_video.py](file:///d:/HAVEN/backend/test_adl_video.py) để chạy pipeline trên file mp4 với debug overlay chi tiết

## Phase 2: Interactivity & Advanced Logic
- [ ] Phone Detection integration (logic giả lập hoặc model detection)
- [ ] Hand Up Detection logic (Wrist vs Shoulder relative position)
- [ ] Zone Configuration (Load bed/chair polymer từ JSON configs)
- [ ] Light Sensor Integration (Giả lập input Light ON/OFF)

## Phase 3: Integration & Optimization
- [ ] Tích hợp vào [StreamManager](file:///d:/HAVEN/backend/src/core/stream_manager.py#33-34) của Backend chính (cho RTSP)
- [ ] Debouncing & Cooldown system (tránh spam events)
- [ ] Websocket Protocol Update (Gửi metadata chi tiết về Client)
- [ ] Export CSV Event Logs

## Phase 4: Frontend UI (Visual Feedback)
- [ ] Vẽ Skeleton với màu quy định (Cam, Vàng, Hồng, Xanh, Nâu, Đỏ)
- [ ] Hiển thị State/Warning/Danger Overlay
- [ ] Dashboard theo dõi sự kiện
