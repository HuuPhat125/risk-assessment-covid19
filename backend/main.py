from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from router import  hospitals, news

app = FastAPI()

# Cấu hình CORS cho phép frontend trên localhost:3000 truy cập
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # cho phép truy cập từ frontend
    allow_credentials=True,
    allow_methods=["*"],          # cho phép tất cả phương thức GET, POST,...
    allow_headers=["*"],          # cho phép tất cả headers
)

# Mount router
app.include_router(hospitals.router, prefix="/hospitals", tags=["hospitals"])
app.include_router(news.router)

@app.get("/")
def read_root():
    return {"message": "Backend API is running."}
