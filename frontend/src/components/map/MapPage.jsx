"use client";

import React, { useEffect, useState, useRef } from "react";
import dynamic from "next/dynamic";
import Legend from "./Legend"; // Nếu có
import Actions from "./Action"; // Nếu có

// Import MapView với SSR false để tránh lỗi Leaflet chạy trên server
const MapView = dynamic(() => import("./MapView"), { ssr: false });

export default function MapPage() {
  const [mounted, setMounted] = useState(false);
  const [hospitals, setHospitals] = useState([]);
  const [userPosition, setUserPosition] = useState([10.9182976, 106.7483136]); // Vị trí mặc định
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const didFetchRef = useRef(false);

  // Đánh dấu mounted để tránh render SSR
  useEffect(() => {
    setMounted(true);
  }, []);

  // Fetch danh sách bệnh viện 1 lần
  useEffect(() => {
    if (!didFetchRef.current) {
      fetchAllHospitals();
      didFetchRef.current = true;
    }
  }, []);

  // Hàm lấy data bệnh viện từ backend
  const fetchAllHospitals = () => {
    setLoading(true);
    setError(null);

    fetch("http://127.0.0.1:8000/hospitals/hospitals_all")
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        return res.json();
      })
      .then((data) => {
        if (Array.isArray(data)) setHospitals(data);
        else {
          console.error("Dữ liệu trả về không phải mảng:", data);
          setHospitals([]);
        }
      })
      .catch((err) => {
        console.error("Error fetching hospitals:", err);
        setError(
          "Không thể tải dữ liệu bệnh viện. Vui lòng kiểm tra backend đang chạy."
        );
        setHospitals([]);
      })
      .finally(() => setLoading(false));
  };

  // Lấy vị trí người dùng
  useEffect(() => {
    if (typeof window !== "undefined" && navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          const { latitude, longitude } = pos.coords;
          setUserPosition([latitude, longitude]);
          console.log("User position:", latitude, longitude);
        },
        (err) => {
          console.warn("Lỗi lấy vị trí:", err);
          setUserPosition([10.9182976, 106.7483136]); // fallback
        },
        {
          enableHighAccuracy: true,
          timeout: 10000,
          maximumAge: 300000,
        }
      );
    }
  }, []);
  const handleShowAll = () => {
    fetchAllHospitals();
  };
  
  // Hàm lấy bệnh viện gần nhất
  const handleGetNearest = () => {
    if (!userPosition) {
      alert("Vui lòng cho phép truy cập vị trí");
      return;
    }

    const [lat, lon] = userPosition;
    fetch(`http://127.0.0.1:8000/hospitals/nearest?lat=${lat}&lon=${lon}`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        return res.json();
      })
      .then((data) => {
        console.log("Nearest hospital:", data);
        if (data.hospital) {
          setHospitals([data.hospital]);
          alert(
            `Bệnh viện gần nhất: ${data.hospital.name} (${data.distance_km} km)`
          );
        }
      })
      .catch((err) => {
        console.error("Error getting nearest hospital:", err);
        alert("Lỗi khi tìm bệnh viện gần nhất");
      });
  };

  if (!mounted) return null; // Tránh SSR render

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <h1 className="text-3xl font-bold text-blue-900 mb-2">
            COVID-19 Interactive Map
          </h1>
          <p className="text-gray-600">
            Real-time COVID-19 data visualization and nearby healthcare
            facilities
          </p>
        </div>
      </div>

      {/* Error message */}
      {error && (
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="bg-red-50 border border-red-200 rounded-lg p-4">
            <p className="text-red-800">❌ {error}</p>
            <button
              onClick={fetchAllHospitals}
              className="mt-2 text-red-900 underline hover:text-red-700"
            >
              Thử lại
            </button>
          </div>
        </div>
      )}

      {/* Main content */}
      <div className="max-w-7xl mx-auto p-4">
        {/* Status info */}
        <div className="mb-6 flex flex-wrap gap-4 text-sm">
          <div className="bg-white px-4 py-2 rounded-lg shadow-sm border">
            <span className="text-gray-600">Bệnh viện: </span>
            <span className="font-semibold text-blue-600">
              {loading ? "Đang tải..." : hospitals.length}
            </span>
          </div>

          <div className="bg-white px-4 py-2 rounded-lg shadow-sm border">
            <span className="text-gray-600">Vị trí: </span>
            <span className="font-semibold text-green-600">
              {userPosition ? "✅ Đã bật" : "❌ Chưa bật"}
            </span>
          </div>

          <div className="bg-white px-4 py-2 rounded-lg shadow-sm border">
            <span className="text-gray-600">Backend: </span>
            <span className="font-semibold text-blue-600">
              {error ? "❌ Lỗi" : "✅ Kết nối"}
            </span>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Map area */}
          <div className="lg:col-span-2 bg-white rounded-lg shadow-sm p-4 border">
            <div className="mb-4">
              <h2 className="text-xl font-semibold text-gray-900">
                Bản đồ tương tác
              </h2>
              <p className="text-sm text-gray-600">
                Click vào marker để xem thông tin chi tiết
              </p>
            </div>

            {loading ? (
              <div className="h-[500px] bg-gray-100 rounded flex items-center justify-center">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
                  <p className="text-gray-600">Đang tải bản đồ...</p>
                </div>
              </div>
            ) : (
              <MapView hospitals={hospitals} userPosition={userPosition} />
            )}
          </div>

          {/* Legend and Actions */}
          <div className="space-y-6">
            {Legend && <Legend />}
            {Actions && (
              <Actions
                onGetNearest={handleGetNearest}
                onShowAll ={handleShowAll}
                userPosition={userPosition}
                hospitalsCount={hospitals.length}
              />
            )}
          </div>
        </div>

        {/* Debug info chỉ hiện ở dev */}
        {process.env.NODE_ENV === "development" && (
          <div className="mt-8 bg-white rounded-lg shadow-sm p-6 border">
            <h3 className="text-lg font-semibold text-gray-800 mb-3">
              Debug Info
            </h3>
            <pre className="text-xs bg-gray-100 p-4 rounded overflow-auto max-h-96">
              {JSON.stringify({ userPosition, hospitals }, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
}
