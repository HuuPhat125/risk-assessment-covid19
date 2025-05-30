import Link from "next/link";

export default function HowItHelpsSection() {
  return (
    <div className="flex flex-col xl:flex-row justify-center items-center gap-8 xl:gap-16 w-full px-4 sm:px-6 lg:px-8 py-8 lg:py-12">
      {/* Left Content - Feature Cards */}
      <div className="w-full max-w-2xl xl:max-w-none flex flex-col sm:flex-row xl:flex-row justify-between items-stretch gap-4 sm:gap-6">
        {/* Risk Triage Card */}
        <div className="flex-1 min-h-72 sm:min-h-80 flex flex-col justify-center items-center gap-6 px-4 py-6 rounded-2xl border border-gray-200 hover:border-gray-300 hover:shadow-sm transition-all duration-200">
          <div className="flex flex-col items-center gap-3">
            {/* Icon placeholder */}
            <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
              <svg
                className="w-6 h-6 text-blue-600"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
              </svg>
            </div>
            <h3 className="font-bold text-base text-blue-900 text-center">
              Risk Triage
            </h3>
          </div>
          <p className="font-medium text-sm text-center text-gray-400 leading-relaxed">
            Quickly identify patients at risk of severe COVID-19 based on key
            symptoms and conditions.
          </p>
        </div>

        {/* Medical Recommendation Card */}
        <div className="flex-1 min-h-72 sm:min-h-80 flex flex-col justify-center items-center gap-6 px-4 py-6 rounded-2xl border border-gray-200 hover:border-gray-300 hover:shadow-sm transition-all duration-200">
          <div className="flex flex-col items-center gap-3">
            {/* Icon placeholder */}
            <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center">
              <svg
                className="w-6 h-6 text-green-600"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                />
              </svg>
            </div>
            <h3 className="font-bold text-base text-center text-blue-900">
              Medical Recommendation
            </h3>
          </div>
          <p className="font-medium text-sm text-center text-gray-400 leading-relaxed">
            Receive AI-backed suggestions to prioritize care and optimize
            resource allocation.
          </p>
        </div>

        {/* Nearby Facilities Card */}
        <div className="flex-1 min-h-72 sm:min-h-80 flex flex-col justify-center items-center gap-6 px-4 py-6 rounded-2xl border border-gray-200 hover:border-gray-300 hover:shadow-sm transition-all duration-200">
          <div className="flex flex-col items-center gap-3">
            {/* Icon placeholder */}
            <div className="w-12 h-12 bg-red-100 rounded-lg flex items-center justify-center">
              <svg
                className="w-6 h-6 text-red-600"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"
                />
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"
                />
              </svg>
            </div>
            <h3 className="font-bold text-base text-center text-blue-900">
              Nearby Facilities
            </h3>
          </div>
          <p className="font-medium text-sm text-center text-gray-400 leading-relaxed">
            Find and connect to the closest COVID-19 medical support centers.
          </p>
        </div>
      </div>

      {/* Right Content - Description */}
      <div className="flex-shrink-0 w-full max-w-lg xl:max-w-md flex flex-col justify-center gap-6 lg:gap-8 text-center xl:text-left">
        <div className="flex flex-col gap-4 lg:gap-6">
          <div className="flex flex-col gap-3 lg:gap-4">
            <h2 className="font-bold text-2xl lg:text-3xl text-blue-900 leading-tight">
              How it helps people
            </h2>
            <p className="font-medium text-sm lg:text-base text-gray-700 leading-relaxed">
              Our machine learning model helps identify high-risk COVID-19 cases
              early, improving triage and treatment planning for better
              outcomes.
            </p>
          </div>
          <p className="font-bold text-sm lg:text-base text-gray-400">
            Supports frontline healthcare workers with real-time risk analysis.
          </p>
        </div>


        <div className="flex justify-center xl:justify-start">
          <Link href="/prediction">
            <button className="inline-flex justify-center items-center gap-2 px-4 py-2.5 rounded-xl border border-gray-300 hover:border-gray-400 hover:bg-gray-50 transition-all duration-200">
              <span className="font-bold text-base text-blue-900">
                Get Started
              </span>
            </button>
          </Link>
        </div>
      </div>
    </div>
  );
}
