"use client";
export default function Legend() {
  return (
    <div className="bg-white p-4 rounded shadow border border-gray-200">
      <h3 className="text-lg font-semibold mb-3">Map Legend</h3>
      <ul className="space-y-2 text-gray-700 text-sm">
        <li>
          <span className="inline-block w-4 h-4 bg-red-500 rounded-full mr-2"></span>
          High Risk Areas
        </li>
        <li>
          <span className="inline-block w-4 h-4 bg-yellow-400 rounded-full mr-2"></span>
          Medium Risk Areas
        </li>
        <li>
          <span className="inline-block w-4 h-4 bg-green-400 rounded-full mr-2"></span>
          Low Risk Areas
        </li>
        <li>
          <span className="inline-block w-4 h-4 bg-blue-500 rounded-full mr-2"></span>
          Healthcare Facilities
        </li>
      </ul>
    </div>
  );
}
