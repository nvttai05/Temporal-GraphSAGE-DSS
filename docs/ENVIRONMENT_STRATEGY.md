# Chiến lược môi trường huấn luyện

## Quyết định

Sử dụng mô hình hybrid trong quá trình phát triển, nhưng chuẩn hóa mọi kết quả chính thức trên Google Colab.

### Laptop

- Viết và rà soát code.
- Chạy GĐ6.0.
- Unit test và integration test.
- Smoke test Popularity, CBF và XGBoost trên dữ liệu nhỏ.
- Không dùng runtime laptop làm số so sánh chính thức.

### Google Colab

- Tái kiểm tra checksum và unit test trước mỗi batch train.
- Chạy toàn bộ baseline chính thức, kể cả các baseline nhẹ.
- Train Static GraphSAGE ở GĐ6 và Temporal GraphSAGE ở GĐ7.
- Chạy evaluation engine chung và xuất bảng kết quả cuối.

### Kaggle

Chỉ dùng khi Colab không khả dụng. Run từ Kaggle phải lưu environment snapshot riêng. Metric chất lượng có thể ghép vào bảng nếu protocol và artifact hoàn toàn khớp; runtime không được so trực tiếp với Colab.

## Artifact bắt buộc của mỗi run

- Dataset version và checksum verification result.
- Git commit hoặc source archive SHA-256.
- Config đầy đủ và config hash.
- Data seed, model seed và scenario.
- Python/package versions.
- Loại CPU/GPU và dung lượng RAM/VRAM.
- Checkpoint/model file.
- Prediction file trước khi tổng hợp metric.
- Metric JSON và execution log.

Nhờ lưu prediction file, evaluation engine có thể được chạy lại mà không cần train lại mô hình.

