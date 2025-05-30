"use client";

import { useState } from "react";
import DatasetVisualization from "./DatasetVisualization";
import DataUpload from "./DataUpload";

export default function DatasetPage() {
  const [activeTab, setActiveTab] = useState("visualization"); // "visualization" or "upload"

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <h1 className="text-3xl font-bold text-blue-900 mb-2">
            Dataset Management
          </h1>
          <p className="text-gray-600">
            Visualize COVID-19 dataset and upload new data for training
          </p>
        </div>
      </div>

      {/* Main content */}
      <div className="max-w-7xl mx-auto p-4">
        {/* Tab navigation */}
        <div className="mb-6">
          <div className="border-b border-gray-200">
            <nav className="-mb-px flex space-x-8">
              <button
                onClick={() => setActiveTab("visualization")}
                className={`py-4 px-1 border-b-2 font-medium text-sm ${
                  activeTab === "visualization"
                    ? "border-indigo-500 text-indigo-600"
                    : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
                }`}
              >
                Dataset Visualization
              </button>
              <button
                onClick={() => setActiveTab("upload")}
                className={`py-4 px-1 border-b-2 font-medium text-sm ${
                  activeTab === "upload"
                    ? "border-indigo-500 text-indigo-600"
                    : "border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300"
                }`}
              >
                Upload New Data
              </button>
            </nav>
          </div>
        </div>

        {/* Tab content */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          {activeTab === "visualization" ? (
            <DatasetVisualization />
          ) : (
            <DataUpload />
          )}
        </div>
      </div>
    </div>
  );
}
