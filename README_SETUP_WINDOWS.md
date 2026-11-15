# DSS Stage 6 — Thiết lập laptop Windows

## 1. Mục đích

Laptop được dùng để phát triển, chạy data preflight, kiểm thử loader/metric và smoke test các baseline nhẹ. Colab được dùng cho các lượt train và đánh giá chính thức.

Khuyến nghị đặt dự án tại đường dẫn ngắn, không đồng bộ OneDrive, ví dụ:

```text
D:\HocTap\DSS\DSS_Stage6
```

## 2. Yêu cầu

- Windows 11.
- Python 3.11 x64.
- VS Code và extension Python.
- Git nếu muốn lưu phiên bản mã nguồn.
- Dataset `medajob-temporal-v0.3` và loader bàn giao từ GĐ5.

Không cần cài CUDA/PyTorch/PyG trên laptop ở bước đầu.

## 3. Tạo môi trường

Mở PowerShell tại thư mục dự án:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
```

Script sẽ:

1. Tạo `.venv` bằng Python 3.11.
2. Cài dependency local.
3. Chạy unit test metric.
4. Ghi snapshot môi trường local.

## 4. Khai báo đường dẫn dataset

Tạo config local:

```powershell
Copy-Item .\configs\paths.example.yaml .\configs\paths.local.yaml
```

Mở `configs/paths.local.yaml` và sửa `dataset_root` tới package frozen. File này bị Git bỏ qua để không công khai đường dẫn máy cá nhân.

Không chép đè, đổi tên hay sửa các file bên trong dataset frozen.

## 5. Kiểm tra môi trường

```powershell
.\scripts\check_environment.ps1
```

Kết quả hợp lệ ban đầu:

- Unit test metric PASS.
- Sinh được `artifacts/environment/local_environment.json`.
- Chưa sinh báo cáo 6.0 vì chưa triển khai kiểm tra dataset.

## 6. Chạy Codex cho bước 6.0

Mở toàn bộ thư mục dự án trong VS Code, sau đó giao nội dung trong:

```text
tasks/GD6_0_CODEX_PROMPT.md
```

Codex phải đọc `AGENTS.md`, hai config đã khóa và data contract của GĐ5 trước khi viết code.

## 7. Phân chia laptop và Colab

| Công việc | Laptop | Colab |
|---|---:|---:|
| Preflight dữ liệu | Chính | Có thể tái kiểm tra |
| Unit/integration test | Chính | Tái kiểm tra trước train |
| Popularity/CBF smoke | Có | Có |
| XGBoost smoke | Có | Có |
| Static GraphSAGE smoke | Dataset nhỏ nếu cần | Chính |
| Mọi kết quả chính thức | Không dùng để chốt | Chính |
| Temporal GraphSAGE GĐ7 | Không | Chính |

Kaggle chỉ là môi trường dự phòng. Nếu buộc dùng Kaggle cho một run chính thức, phải ghi riêng phần cứng, package và runtime; không so sánh tốc độ trực tiếp với Colab.

