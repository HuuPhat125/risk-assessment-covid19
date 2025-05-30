import Link from "next/link";

export default function ExploreDataSection() {
    const dataCategories = [
      {
        title: "Patient Overview",
        bgColor: "bg-blue-100",
        description: "Comprehensive patient demographics and basic information"
      },
      {
        title: "Chronic Conditions",
        bgColor: "bg-blue-200",
        description: "Pre-existing medical conditions and health history"
      },
      {
        title: "Risk Factors",
        bgColor: "bg-blue-300",
        description: "Key indicators that influence COVID-19 severity"
      }
    ];

    return (
      <div className="flex flex-col xl:flex-row justify-between items-center gap-8 xl:gap-12 w-full px-4 sm:px-6 lg:px-8 py-8 lg:py-12">
        {/* Left Content - Description */}
        <div className="flex-shrink-0 w-full max-w-lg xl:max-w-sm flex flex-col justify-center gap-6 lg:gap-10 text-center xl:text-left">
          {/* Text Content */}
          <div className="flex flex-col gap-4 lg:gap-5">
            <h2 className="font-bold text-2xl lg:text-3xl text-blue-900 leading-tight">
              Explore the Data
            </h2>
            <p className="font-medium text-sm lg:text-base text-slate-600 leading-relaxed">
              Learn how patient symptoms, conditions, and test results help predict COVID-19 severity risk
            </p>
          </div>

          {/* CTA Button */}
          <div className="flex justify-center xl:justify-start">
            <Link href="/dataset">
              <button className="inline-flex justify-center items-center gap-2 px-4 lg:px-6 py-3 lg:py-4 rounded-xl border border-gray-300 hover:border-gray-400 hover:bg-gray-50 transition-all duration-200 group">
                <span className="font-bold text-base lg:text-lg text-blue-900">
                  Learn more
                </span>
                <svg
                  className="w-4 h-4 lg:w-5 lg:h-5 text-blue-900 group-hover:translate-x-1 transition-transform duration-200"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 5l7 7-7 7"
                  />
                </svg>
              </button>
            </Link>
          </div>
        </div>

        {/* Right Content - Data Visualization Cards */}
        <div className="flex-1 flex flex-col sm:flex-row items-center justify-center gap-6 sm:gap-8 lg:gap-12 xl:gap-16">
          {dataCategories.map((category, index) => (
            <div
              key={index}
              className="w-full max-w-xs sm:max-w-none sm:w-48 lg:w-56 flex flex-col justify-center items-center gap-4 lg:gap-6 group cursor-pointer"
            >
              {/* Data Visualization Placeholder */}
              <div className={`w-full h-64 sm:h-72 lg:h-80 xl:h-88 ${category.bgColor} rounded-lg group-hover:shadow-lg transition-all duration-300 relative overflow-hidden`}>
                {/* Optional: Add some visual elements to make it look like data */}
                <div className="absolute inset-0 flex flex-col justify-end p-4 opacity-30">
                  <div className="space-y-2">
                    <div className="h-2 bg-white rounded w-3/4"></div>
                    <div className="h-2 bg-white rounded w-1/2"></div>
                    <div className="h-2 bg-white rounded w-2/3"></div>
                  </div>
                </div>

                {/* Chart-like decoration */}
                <div className="absolute top-4 right-4 w-8 h-8 bg-white bg-opacity-40 rounded-full"></div>
                <div className="absolute top-1/3 left-4 w-6 h-6 bg-white bg-opacity-30 rounded-full"></div>
              </div>

              {/* Category Label */}
              <div className="text-center">
                <h3 className="font-semibold text-sm lg:text-base text-gray-400 group-hover:text-gray-600 transition-colors duration-200">
                  {category.title}
                </h3>
                {/* Optional description on hover */}
                <p className="text-xs text-gray-300 mt-1 opacity-0 group-hover:opacity-100 transition-opacity duration-200 hidden sm:block">
                  {category.description}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }
