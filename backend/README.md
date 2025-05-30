# Backend - Covid App

## Yêu cầu

- Python 3.10+
- pip

## Cài đặt môi trường

```bash
python -m venv venv
source venv/bin/activate  # Trên Linux/Mac
venv\Scripts\activate     # Trên Windows
pip install -r requirements.txt
```

Cấu trúc thư mục
data/: Chứa dữ liệu 
router/: Định nghĩa các route API.
utils/: Các hàm tiện ích 

````bash
uvicorn main:app --reload --log-level debug
````