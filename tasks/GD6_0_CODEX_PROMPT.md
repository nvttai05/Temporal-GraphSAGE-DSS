# Nhiệm vụ Codex — GĐ6.0 Data & Evaluation Preflight

Đọc toàn bộ `AGENTS.md`, `configs/stage6_protocol.yaml`, `configs/experiments.yaml`, `configs/paths.local.yaml` và data contract của GĐ5 trước khi hành động.

## Mục tiêu

Triển khai và chạy cổng GĐ6.0 trực tiếp trên package frozen `medajob-temporal-v0.3`. Không sửa hoặc tái sinh dữ liệu GĐ5.

## Sáu check bắt buộc

1. Manifest và toàn bộ SHA-256 bàn giao.
2. Loader đọc đủ seed/scenario/split và graph snapshot không chứa tương lai.
3. Thống kê positive/negative/censored và Top-K eligible cho các nhóm chính.
4. Unit test Recall@10/NDCG@10 bằng fixture tính tay.
5. Xác nhận implementation tuân thủ exposed-set ranking và rolling as-of-t.
6. Xác nhận model list, feature set, loss, seed và tuning budget không còn TBD.

## Yêu cầu triển khai

- Tạo code có thể chạy lại, không chỉ notebook hoặc phân tích một lần.
- Mỗi check có `PASS`, `FAIL` hoặc `BLOCKED` và evidence cụ thể.
- Kiểm thử leakage gồm boundary cases, deterministic sample và fixture cố ý có tương lai.
- Không đọc test predictions hoặc train model ở bước này.
- Không suy ra PASS từ báo cáo GĐ5; phải chạy kiểm tra trên file thực tế.
- Nếu schema thực tế khác data contract, dừng check liên quan và báo `BLOCKED` hoặc `FAIL`; không tự đoán cột.

## Output chính thức

- `reports/DSS_STAGE6_DATA_PREFLIGHT.md`
- `artifacts/preflight/preflight_results.json`

Ngoài ra lưu log và bảng thống kê cần thiết dưới `artifacts/preflight/`. Cuối cùng chạy toàn bộ test và báo rõ có đủ điều kiện chuyển sang GĐ6.1 hay không.

