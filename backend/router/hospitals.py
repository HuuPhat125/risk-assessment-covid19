from fastapi import APIRouter, Query, HTTPException
from typing import Optional
import json
import os

from utils.haversine import haversine  # hàm tính khoảng cách km giữa 2 điểm lat/lon

router = APIRouter()

# Đường dẫn tới file dữ liệu bệnh viện
DATA_PATH = 'data/hospital_data.json'
# Load dữ liệu bệnh viện một lần khi khởi động router
try:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        print(f"Loaded {len(raw_data.get('elements', []))} hospitals from {DATA_PATH}")
        hospitals_data = raw_data.get("elements", [])
except Exception as e:
    hospitals_data = []
    print(f"Error loading hospital data: {e}")


@router.get("/nearest")
def get_nearest_hospital(lat: float = Query(..., description="Latitude người dùng"),
                         lon: float = Query(..., description="Longitude người dùng")):
    """
    Tìm bệnh viện gần nhất dựa trên vị trí lat/lon của người dùng.
    Trả về thông tin bệnh viện gần nhất và khoảng cách (km).
    """
    if not hospitals_data:
        raise HTTPException(status_code=500, detail="No hospital data available")

    nearest = None
    min_distance = float("inf")

    for hospital in hospitals_data:
        hosp_lat = hospital.get("lat")
        hosp_lon = hospital.get("lon")
        tags = hospital.get("tags", {})
        # print('tags', tags)
        name = tags.get("name")
        
        if hosp_lat is None or hosp_lon is None or not name:
            continue

        dist = haversine(lat, lon, hosp_lat, hosp_lon)
        if dist < min_distance:
            min_distance = dist
            nearest = {
                "name": name,
                "latitude": hosp_lat,
                "longitude": hosp_lon,
                "raw": tags
            }

    if nearest is None:
        raise HTTPException(status_code=404, detail="No valid hospital found")

    return {
        "hospital": nearest,
        "distance_km": round(min_distance, 2)
    }
    
@router.get("/hospitals_all")
def get_all_hospitals():
    """
    Trả về danh sách tất cả các bệnh viện, loại bỏ các bệnh viện không có tên
    """
    filtered_data = [
        hospital for hospital in hospitals_data
        if hospital.get("tags", {}).get("name") is not None
    ]
    
    if not filtered_data:
        raise HTTPException(status_code=500, detail="No valid hospital data available")
    
    return filtered_data