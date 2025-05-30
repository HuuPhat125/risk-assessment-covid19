"use client";

import React, { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import L from "leaflet";
import { useMap } from "react-leaflet";

// Dynamic import các component Leaflet tránh SSR lỗi (Next.js không hỗ trợ SSR cho Leaflet)
const MapContainer = dynamic(
  () => import("react-leaflet").then((mod) => mod.MapContainer),
  { ssr: false }
);
const TileLayer = dynamic(
  () => import("react-leaflet").then((mod) => mod.TileLayer),
  { ssr: false }
);
const Marker = dynamic(
  () => import("react-leaflet").then((mod) => mod.Marker),
  { ssr: false }
);
const Popup = dynamic(() => import("react-leaflet").then((mod) => mod.Popup), {
  ssr: false,
});

// Component tự động set vị trí bản đồ khi userPosition thay đổi
function SetViewToUser({ position }) {
  const map = useMap();

  useEffect(() => {
    if (map && position) {
      map.setView(position, 14);
    }
  }, [map, position]);

  return null;
}

export default function MapView({ hospitals = [], userPosition }) {
  const defaultCenter = [10.9182976, 106.7483136];
  const zoom = 12;

  // Tạo icon cho marker user, chỉ tạo 1 lần trên client
  const [userIcon, setUserIcon] = useState(null);
  useEffect(() => {
    const icon = new L.Icon({
      iconUrl: "https://cdn-icons-png.flaticon.com/512/149/149059.png",
      iconSize: [30, 30],
      iconAnchor: [15, 30],
      popupAnchor: [0, -30],
    });
    setUserIcon(icon);
  }, []);

  // Chưa có icon thì không render bản đồ
  if (!userIcon) return null;

  return (
    <div className="h-[500px] w-full rounded overflow-hidden">
      <MapContainer
        center={userPosition || defaultCenter}
        zoom={zoom}
        style={{ height: "100%", width: "100%" }}
        className="rounded"
      >
        <TileLayer
          attribution='&copy; <a href="https://osm.org/copyright">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {/* Auto center map khi vị trí user thay đổi */}
        {userPosition && <SetViewToUser position={userPosition} />}

        {/* Marker vị trí user */}
        {userPosition && (
          <Marker position={userPosition} icon={userIcon}>
            <Popup>
              <div className="text-center">
                <strong>📍 Bạn đang ở đây</strong>
                <br />
                <small>
                  {userPosition[0].toFixed(4)}, {userPosition[1].toFixed(4)}
                </small>
              </div>
            </Popup>
          </Marker>
        )}

        {/* Marker các bệnh viện */}
        {hospitals.map((hospital, index) => {
          const lat = hospital.lat ?? hospital.latitude ?? hospital.center?.lat;
          const lon =
            hospital.lon ?? hospital.longitude ?? hospital.center?.lon;
          if (!lat || !lon) return null;

          const name =
            hospital.name ?? hospital.tags?.name ?? "Bệnh viện không rõ tên";
          const hospitalType =
            hospital.hospital_type ?? hospital.tags?.amenity ?? "hospital";
          const hospitalKey = hospital.id || hospital.osm_id || index;

          return (
            <Marker key={hospitalKey} position={[lat, lon]}>
              <Popup>
                <div className="max-w-xs">
                  <strong className="text-blue-900">{name}</strong>
                  <br />
                  <span className="text-gray-600 capitalize">
                    {hospitalType.replace("_", " ")}
                  </span>
                  {hospital.addr_street && (
                    <>
                      <br />
                      <small className="text-gray-500">
                        📍 {hospital.addr_street}
                      </small>
                    </>
                  )}
                  {hospital.distance && (
                    <>
                      <br />
                      <small className="text-green-600 font-medium">
                        🚗 {hospital.distance} km
                      </small>
                    </>
                  )}
                  <br />
                  <small className="text-gray-400">
                    {lat.toFixed(4)}, {lon.toFixed(4)}
                  </small>
                </div>
              </Popup>
            </Marker>
          );
        })}
      </MapContainer>

      {/* Thông tin bản đồ dưới */}
      <div className="mt-2 text-sm text-gray-600 flex justify-between">
        <span>🏥 {hospitals.length} bệnh viện được hiển thị</span>
        {userPosition && (
          <span>
            📍 {userPosition[0].toFixed(4)}, {userPosition[1].toFixed(4)}
          </span>
        )}
      </div>
    </div>
  );
}
