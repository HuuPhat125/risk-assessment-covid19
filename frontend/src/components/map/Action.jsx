"use client";
export default function Actions({
  onGetNearest,
  onShowAll,
  userPosition,
  hospitalsCount = 0,
}) {
  return (
    <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-200">
      <h3 className="text-lg font-semibold mb-4">Quick Actions</h3>

      <div className="space-y-3">
        <button
          onClick={onGetNearest}
          disabled={!userPosition}
          className={`w-full text-left px-4 py-3 rounded-lg transition ${
            userPosition
              ? "bg-blue-50 hover:bg-blue-100 text-blue-900"
              : "bg-gray-100 text-gray-500 cursor-not-allowed"
          }`}
        >
          <div className="font-medium">🏥 Find Nearby Hospitals</div>
          <div className="text-sm mt-1">
            {userPosition
              ? "Tìm bệnh viện trong bán kính 10km"
              : "Cần bật định vị trước"}
          </div>
        </button>

        <button
          onClick={onShowAll}
          className="w-full text-left px-4 py-3 bg-indigo-50 hover:bg-indigo-100 rounded-lg transition"
        >
          <div className="font-medium text-indigo-900">
            🏥 Xem toàn bộ bệnh viện
          </div>
          <div className="text-sm text-indigo-600 mt-1">
            Xem danh sách toàn bộ bệnh viện
          </div>
        </button>
      </div>

      <div className="mt-4 pt-4 border-t border-gray-200 text-xs text-gray-500 flex justify-between">
        <span>Backend: ✅ Connected</span>
        <span>Maps: ✅ Loaded</span>
      </div>
    </div>
  );
}
