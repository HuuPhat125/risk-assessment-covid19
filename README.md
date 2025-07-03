# Covid-19 Risk Assessment & Information Platform
[Link website demo](https://risk-assessment-covid19-web-ifry.vercel.app/)
(đã tắt model dự đoán 🦫)

Dự án gồm 2 thành phần chính:

- **1. risk-assessment-covid19**: Hệ thống train, đánh giá, triển khai model AI dự đoán nguy cơ Covid-19.
- **2. web**: Website cung cấp thông tin, bản đồ, tin tức và giao tiếp với backend/model.

---

## 1. risk-assessment-covid19 (Model & Backend AI)

### Chức năng

- Tiền xử lý dữ liệu, huấn luyện nhiều mô hình ML (Logistic Regression, Decision Tree, KNN, MLP, Naive Bayes, Random Forest, ...)
- Tự động hóa train, log, lưu version model lên S3.
- Triển khai model lên AWS SageMaker.
- Lưu trữ log chi tiết, kết quả train, versioning model.

### Cấu trúc thư mục

```
risk-assessment-covid19/
├── data/                # Dữ liệu train/val/test
├── new_data/            # Dữ liệu mới, chưa xử lý
├── src/                 # Code train, inference, requirements.txt
├── train_job.py         # Script train + upload model
├── deploy.py            # Script deploy model
├── .github/workflows/   # CI/CD pipeline
├── Dockerfile           # Docker cho train/inference
└── ...
```

### Quy trình Train & Deploy Model

#### a. Train Model

- **Tự động hóa qua CI/CD (GitHub Actions):**
    - Khi có Pull Request lên branch `train`, workflow `.github/workflows/train.yml` sẽ tự động:
        1. Cài đặt Python, các dependencies.
        2. Thiết lập AWS credentials.
        3. Chạy script train:
            ```sh
            python train_job.py --model_config logistic_regression
            ```
        4. Kết quả train (log chi tiết, metrics) được ghi vào `detail.txt` và comment lên PR.
        5. Model, log, version được upload lên S3.

- **Thủ công:**
    ```sh
    cd risk-assessment-covid19
    python train_job.py --model_config logistic_regression
    ```

#### b. Deploy Model

- **Tự động hóa qua CI/CD (GitHub Actions):**
    - Khi có Pull Request lên branch `main` hoặc trigger workflow `.github/workflows/deploy.yml`:
        1. Cài đặt dependencies, thiết lập AWS credentials.
        2. Chạy script deploy:
            ```sh
            python deploy.py --model_config logistic_regression,decision_tree,...
            ```
        3. Model được deploy lên SageMaker endpoint.

- **Thủ công:**
    ```sh
    cd risk-assessment-covid19
    python deploy.py --model_config logistic_regression
    ```

#### c. CI/CD Pipeline

- **Train:**  
    ![Train Workflow](.github/workflows/train.yml)
    - Trigger: Pull Request lên branch `train`
    - Action: Train model, log kết quả, upload S3, comment report lên PR.

- **Deploy:**  
    ![Deploy Workflow](.github/workflows/deploy.yml)
    - Trigger: Pull Request lên branch `main` hoặc thủ công
    - Action: Deploy model lên SageMaker.

---

## 2. web (Website & API) [repo web](https://github.com/HuuPhat125/risk-assessment-covid19-web/tree/main)


### Chức năng

- **Backend** (`web/backend`):  
    - API FastAPI cung cấp dữ liệu bệnh viện, tin tức, upload, ...  
    - Đọc dữ liệu từ các file JSON, CSV, ...  
    - Cấu hình CORS, RESTful API.

- **Frontend** (`web/frontend`):  
    - Next.js (React) hiển thị bản đồ, danh sách bệnh viện, tin tức Covid-19, giao tiếp API backend.
    - Responsive, tối ưu SEO, sử dụng Leaflet cho bản đồ.

### Cài đặt & Chạy thử

#### Backend

```sh
cd web/backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

#### Frontend

```sh
cd web/frontend
npm install
npm run dev
```

---

## 3. Tổng quan quy trình DevOps (CI/CD)

1. **Push code lên GitHub**  
   ⬇️  
2. **GitHub Actions tự động chạy workflow:**  
   - **Train:** Train model, log kết quả, upload S3, comment report lên PR.
   - **Deploy:** Deploy model lên SageMaker endpoint.
   ⬇️  
3. **Model mới sẵn sàng, website/backend có thể gọi API model mới để inference.**

---
