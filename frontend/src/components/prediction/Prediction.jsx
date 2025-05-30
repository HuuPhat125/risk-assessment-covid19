"use client";

import { useState } from "react";
import {
  Activity,
  Brain,
  AlertTriangle,
  CheckCircle,
  Loader2,
} from "lucide-react";

export default function PredictionPage() {
  const [formData, setFormData] = useState({
    sex: "",
    age: "",
    patient_type: "",
    pneumonia: "",
    pregnancy: "",
    diabetes: "",
    copd: "",
    asthma: "",
    inmsupr: "",
    hypertension: "",
    cardiovascular: "",
    renal_chronic: "",
    other_disease: "",
    obesity: "",
    tobacco: "",
    usmr: "",
    medical_unit: "",
    intubed: "",
    icu: "",
  });

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const formFields = [
    {
      section: "Basic Information",
      fields: [
        {
          name: "Name",
          label: "Name",
          type: "text",
          required: false,
        },
        {
          name: "sex",
          label: "Sex",
          type: "select",
          options: [
            { value: "1", label: "Female" },
            { value: "2", label: "Male" },
          ],
          required: true,
        },
        {
          name: "age",
          label: "Age",
          type: "number",
          min: 0,
          max: 120,
          required: true,
        },
        {
          name: "patient_type",
          label: "Patient Type",
          type: "select",
          options: [
            { value: "1", label: "Returned Home" },
            { value: "2", label: "Hospitalization" },
          ],
          required: true,
        },
      ],
    },
    {
      section: "Medical Conditions",
      fields: [
        {
          name: "pneumonia",
          label: "Pneumonia (Air sacs inflammation)",
          type: "select",
          options: [
            { value: "1", label: "Yes" },
            { value: "2", label: "No" },
            { value: "97", label: "Unknown" },
          ],
        },
        {
          name: "pregnancy",
          label: "Pregnancy",
          type: "select",
          options: [
            { value: "1", label: "Yes" },
            { value: "2", label: "No" },
            { value: "97", label: "Unknown" },
          ],
        },
        {
          name: "diabetes",
          label: "Diabetes",
          type: "select",
          options: [
            { value: "1", label: "Yes" },
            { value: "2", label: "No" },
            { value: "97", label: "Unknown" },
          ],
        },
        {
          name: "copd",
          label: "COPD (Chronic Obstructive Pulmonary Disease)",
          type: "select",
          options: [
            { value: "1", label: "Yes" },
            { value: "2", label: "No" },
            { value: "97", label: "Unknown" },
          ],
        },
        {
          name: "asthma",
          label: "Asthma",
          type: "select",
          options: [
            { value: "1", label: "Yes" },
            { value: "2", label: "No" },
            { value: "97", label: "Unknown" },
          ],
        },
        {
          name: "inmsupr",
          label: "Immunosuppressed",
          type: "select",
          options: [
            { value: "1", label: "Yes" },
            { value: "2", label: "No" },
            { value: "97", label: "Unknown" },
          ],
        },
        {
          name: "hypertension",
          label: "Hypertension",
          type: "select",
          options: [
            { value: "1", label: "Yes" },
            { value: "2", label: "No" },
            { value: "97", label: "Unknown" },
          ],
        },
        {
          name: "cardiovascular",
          label: "Cardiovascular Disease",
          type: "select",
          options: [
            { value: "1", label: "Yes" },
            { value: "2", label: "No" },
            { value: "97", label: "Unknown" },
          ],
        },
        {
          name: "renal_chronic",
          label: "Chronic Renal Disease",
          type: "select",
          options: [
            { value: "1", label: "Yes" },
            { value: "2", label: "No" },
            { value: "97", label: "Unknown" },
          ],
        },
        {
          name: "other_disease",
          label: "Other Disease",
          type: "select",
          options: [
            { value: "1", label: "Yes" },
            { value: "2", label: "No" },
            { value: "97", label: "Unknown" },
          ],
        },
        {
          name: "obesity",
          label: "Obesity",
          type: "select",
          options: [
            { value: "1", label: "Yes" },
            { value: "2", label: "No" },
            { value: "97", label: "Unknown" },
          ],
        },
        {
          name: "tobacco",
          label: "Tobacco User",
          type: "select",
          options: [
            { value: "1", label: "Yes" },
            { value: "2", label: "No" },
            { value: "97", label: "Unknown" },
          ],
        },
      ],
    },
    {
      section: "Medical Care Information",
      fields: [
        {
          name: "usmr",
          label: "Medical Unit Level",
          type: "select",
          options: [
            { value: "1", label: "First Level" },
            { value: "2", label: "Second Level" },
            { value: "3", label: "Third Level" },
            { value: "97", label: "Unknown" },
          ],
        },
        {
          name: "medical_unit",
          label: "Type of Medical Institution",
          type: "select",
          options: [
            { value: "1", label: "IMSS" },
            { value: "2", label: "ISSSTE" },
            { value: "3", label: "PEMEX" },
            { value: "4", label: "SEDENA" },
            { value: "5", label: "SEMAR" },
            { value: "6", label: "SSA" },
            { value: "7", label: "Private" },
            { value: "8", label: "Other" },
            { value: "97", label: "Unknown" },
          ],
        },
        {
          name: "intubed",
          label: "Connected to Ventilator",
          type: "select",
          options: [
            { value: "1", label: "Yes" },
            { value: "2", label: "No" },
            { value: "97", label: "Unknown" },
          ],
        },
        {
          name: "icu",
          label: "Admitted to ICU",
          type: "select",
          options: [
            { value: "1", label: "Yes" },
            { value: "2", label: "No" },
            { value: "97", label: "Unknown" },
          ],
        },
      ],
    },
  ];

  const handleInputChange = (name, value) => {
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
    setError("");
  };

  const validateForm = () => {
    const requiredFields = ["sex", "age",  "patient_type"];
    for (let field of requiredFields) {
      if (!formData[field]) {
        setError(`Please fill in the required field: ${field}`);
        return false;
      }
    }
    return true;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!validateForm()) return;

    setLoading(true);
    setError("");

    try {
      // Django API call
      const response = await fetch("/api/predict/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-CSRFToken": getCookie("csrftoken"), // CSRF token for Django
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setPrediction(result);
    } catch (err) {
      setError("Failed to get prediction. Please try again.");
      console.error("Prediction error:", err);
    } finally {
      setLoading(false);
    }
  };

  // Helper function to get CSRF token
  const getCookie = (name) => {
    let cookieValue = null;
    if (document.cookie && document.cookie !== "") {
      const cookies = document.cookie.split(";");
      for (let i = 0; i < cookies.length; i++) {
        const cookie = cookies[i].trim();
        if (cookie.substring(0, name.length + 1) === name + "=") {
          cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
          break;
        }
      }
    }
    return cookieValue;
  };

  const resetForm = () => {
    setFormData({
      sex: "",
      age: "",
      patient_type: "",
      pneumonia: "",
      pregnancy: "",
      diabetes: "",
      copd: "",
      asthma: "",
      inmsupr: "",
      hypertension: "",
      cardiovascular: "",
      renal_chronic: "",
      other_disease: "",
      obesity: "",
      tobacco: "",
      usmr: "",
      medical_unit: "",
      intubed: "",
      icu: "",
    });
    setPrediction(null);
    setError("");
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Form Section */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-6">
                Patient Information
              </h2>

              <form onSubmit={handleSubmit} className="space-y-8">
                {formFields.map((section, sectionIndex) => (
                  <div key={sectionIndex} className="space-y-4">
                    <h3 className="text-lg font-medium text-gray-800 border-b border-gray-200 pb-2">
                      {section.section}
                    </h3>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {section.fields.map((field) => (
                        <div key={field.name} className="space-y-2">
                          <label className="block text-sm font-medium text-gray-700">
                            {field.label}
                            {field.required && (
                              <span className="text-red-500 ml-1">*</span>
                            )}
                          </label>

                          {field.type === "select" ? (
                            <select
                              value={formData[field.name]}
                              onChange={(e) =>
                                handleInputChange(field.name, e.target.value)
                              }
                              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                              required={field.required}
                            >
                              <option value="">Select...</option>
                              {field.options.map((option) => (
                                <option key={option.value} value={option.value}>
                                  {option.label}
                                </option>
                              ))}
                            </select>
                          ) : (
                            <input
                              type={field.type}
                              value={formData[field.name]}
                              onChange={(e) =>
                                handleInputChange(field.name, e.target.value)
                              }
                              min={field.min}
                              max={field.max}
                              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                              required={field.required}
                            />
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                ))}

                {error && (
                  <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                    <div className="flex items-center gap-2">
                      <AlertTriangle className="w-5 h-5 text-red-500" />
                      <span className="text-red-700">{error}</span>
                    </div>
                  </div>
                )}

                <div className="flex gap-4">
                  <button
                    type="submit"
                    disabled={loading}
                    className="flex-1 bg-blue-600 text-white py-3 px-6 rounded-lg font-semibold hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-200 flex items-center justify-center gap-2"
                  >
                    {loading ? (
                      <>
                        <Loader2 className="w-5 h-5 animate-spin" />
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <Brain className="w-5 h-5" />
                        Get Risk Prediction
                      </>
                    )}
                  </button>

                  <button
                    type="button"
                    onClick={resetForm}
                    className="px-6 py-3 border border-gray-300 text-gray-700 rounded-lg font-semibold hover:bg-gray-50 transition-colors duration-200"
                  >
                    Reset Form
                  </button>
                </div>
              </form>
            </div>
          </div>

          {/* Results Section */}
          <div className="space-y-6">
            {/* Dataset Info */}
            <div className="bg-blue-50 rounded-lg border border-blue-200 p-6">
              <h3 className="text-lg font-semibold text-blue-900 mb-3">
                About This Model
              </h3>
              <div className="space-y-2 text-sm text-blue-800">
                <p>• Trained on 1,048,576 anonymized patients</p>
                <p>• 21 unique medical features</p>
                <p>• Mexican government dataset</p>
                <p>• Boolean features: 1=Yes, 2=No</p>
                <p>• Missing data: 97, 99</p>
              </div>
            </div>

            {/* Prediction Results */}
            {prediction && (
              <div
                className={`rounded-lg border p-6 ${
                  prediction.risk_level === "HIGH"
                    ? "bg-red-50 border-red-200"
                    : "bg-green-50 border-green-200"
                }`}
              >
                <div className="flex items-center gap-3 mb-4">
                  {prediction.risk_level === "HIGH" ? (
                    <AlertTriangle className="w-8 h-8 text-red-500" />
                  ) : (
                    <CheckCircle className="w-8 h-8 text-green-500" />
                  )}
                  <div>
                    <h3
                      className={`text-xl font-bold ${
                        prediction.risk_level === "HIGH"
                          ? "text-red-900"
                          : "text-green-900"
                      }`}
                    >
                      {prediction.risk_level} RISK
                    </h3>
                    <p
                      className={`text-sm ${
                        prediction.risk_level === "HIGH"
                          ? "text-red-700"
                          : "text-green-700"
                      }`}
                    >
                      Confidence: {(prediction.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>

                <div className="space-y-3">
                  <div
                    className={`text-sm ${
                      prediction.risk_level === "HIGH"
                        ? "text-red-800"
                        : "text-green-800"
                    }`}
                  >
                    <p className="font-medium mb-2">Recommendations:</p>
                    <ul className="space-y-1">
                      {prediction.recommendations?.map((rec, index) => (
                        <li key={index}>• {rec}</li>
                      ))}
                    </ul>
                  </div>

                  {prediction.risk_factors && (
                    <div className="mt-4 pt-4 border-t border-gray-200">
                      <p className="font-medium text-gray-700 mb-2">
                        Key Risk Factors:
                      </p>
                      <div className="flex flex-wrap gap-2">
                        {prediction.risk_factors.map((factor, index) => (
                          <span
                            key={index}
                            className="px-2 py-1 bg-gray-200 rounded text-xs"
                          >
                            {factor}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Disclaimer */}
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
              <div className="flex items-start gap-2">
                <AlertTriangle className="w-5 h-5 text-yellow-600 mt-0.5" />
                <div className="text-sm text-yellow-800">
                  <p className="font-medium mb-1">Medical Disclaimer</p>
                  <p>
                    This tool is for informational purposes only and should not
                    replace professional medical advice, diagnosis, or
                    treatment.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
