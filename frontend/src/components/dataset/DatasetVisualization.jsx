"use client";

import { useState } from "react";

export default function DatasetVisualization() {
  const [selectedChart, setSelectedChart] = useState("distribution");

  const chartOptions = [
    { id: "distribution", label: "Age Distribution" },
    { id: "symptoms", label: "Symptoms Analysis" },
    { id: "outcomes", label: "Outcome Analysis" },
    { id: "risk", label: "Risk Factors" },
  ];

  return (
    <div className="space-y-6">
      {/* Chart selector */}
      <div className="flex flex-wrap gap-2">
        {chartOptions.map((option) => (
          <button
            key={option.id}
            onClick={() => setSelectedChart(option.id)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              selectedChart === option.id
                ? "bg-indigo-100 text-indigo-900"
                : "bg-gray-100 text-gray-600 hover:bg-gray-200"
            }`}
          >
            {option.label}
          </button>
        ))}
      </div>

      {/* Dataset statistics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-blue-50 p-4 rounded-lg">
          <h3 className="text-sm font-medium text-blue-900">Total Records</h3>
          <p className="text-2xl font-bold text-blue-700 mt-1">1,234,567</p>
        </div>
        <div className="bg-green-50 p-4 rounded-lg">
          <h3 className="text-sm font-medium text-green-900">Positive Cases</h3>
          <p className="text-2xl font-bold text-green-700 mt-1">456,789</p>
        </div>
        <div className="bg-yellow-50 p-4 rounded-lg">
          <h3 className="text-sm font-medium text-yellow-900">Recovered</h3>
          <p className="text-2xl font-bold text-yellow-700 mt-1">345,678</p>
        </div>
        <div className="bg-red-50 p-4 rounded-lg">
          <h3 className="text-sm font-medium text-red-900">Deceased</h3>
          <p className="text-2xl font-bold text-red-700 mt-1">12,345</p>
        </div>
      </div>

      {/* Chart area */}
      <div className="bg-gray-50 rounded-lg p-6 min-h-[400px] flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 bg-indigo-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg
              className="w-8 h-8 text-indigo-600"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
              />
            </svg>
          </div>
          <h3 className="text-lg font-medium text-gray-900 mb-1">
            {chartOptions.find((opt) => opt.id === selectedChart)?.label}
          </h3>
          <p className="text-gray-500">
            Chart visualization will be implemented here
          </p>
        </div>
      </div>

      {/* Dataset information */}
      <div className="bg-white rounded-lg border border-gray-200 p-4">
        <h3 className="text-lg font-medium text-gray-900 mb-4">
          Dataset Information
        </h3>
        <div className="space-y-3">
          <div className="flex justify-between">
            <span className="text-gray-600">Last Updated</span>
            <span className="font-medium">March 15, 2024</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Data Source</span>
            <span className="font-medium">Mexican Government</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Features</span>
            <span className="font-medium">45 columns</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Missing Values</span>
            <span className="font-medium">2.3%</span>
          </div>
        </div>
      </div>
    </div>
  );
}
