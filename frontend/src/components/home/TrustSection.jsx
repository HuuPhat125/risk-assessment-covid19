export default function TrustSection() {
  const trustFactors = [
    {
      title: "Based on large-scale real-world data",
      description:
        "Trained on anonymized records from over 1 million COVID-19 patients provided by the Mexican government.",
      icon: (
        <svg
          className="w-8 h-8 text-slate-500"
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
      ),
    },
    {
      title: "Verified with clinical guidelines",
      description:
        "Updated using latest guidelines from WHO and CDC, validated by medical experts to ensure safety and reliability.",
      icon: (
        <svg
          className="w-8 h-8 text-slate-500"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
          />
        </svg>
      ),
    },
  ];

  return (
    <div className="flex flex-col justify-center items-center gap-8 lg:gap-15 w-full px-4 sm:px-6 lg:px-8 py-12 lg:py-16 bg-slate-100">
      {/* Header */}
      <h2 className="font-bold text-2xl sm:text-3xl lg:text-4xl text-center text-blue-900 leading-tight max-w-2xl">
        Why you can trust this tool
      </h2>

      {/* Trust Factors */}
      <div className="flex flex-col lg:flex-row justify-center items-center lg:items-stretch gap-8 lg:gap-16 xl:gap-24 w-full max-w-6xl">
        {trustFactors.map((factor, index) => (
          <div
            key={index}
            className="flex-1 max-w-md lg:max-w-none flex flex-col justify-center gap-4 lg:gap-6 text-center lg:text-left group"
          >
            {/* Icon (visible on larger screens) */}
            <div className="hidden lg:flex justify-center lg:justify-start mb-2">
              {factor.icon}
            </div>

            {/* Title */}
            <h3 className="font-bold text-lg sm:text-xl lg:text-2xl text-slate-500 group-hover:text-slate-600 transition-colors duration-200 leading-tight">
              {factor.title}
            </h3>

            {/* Description */}
            <p className="font-semibold text-sm sm:text-base text-slate-400 leading-relaxed">
              {factor.description}
            </p>

            {/* Visual separator for mobile */}
            {index === 0 && (
              <div className="lg:hidden w-16 h-px bg-slate-300 mx-auto mt-4"></div>
            )}
          </div>
        ))}
      </div>

      {/* Optional: Trust badges/stats */}
      <div className="flex flex-col sm:flex-row items-center justify-center gap-6 sm:gap-12 mt-4 lg:mt-8">
        <div className="flex items-center gap-2 text-slate-400">
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M6.267 3.455a3.066 3.066 0 001.745-.723 3.066 3.066 0 013.976 0 3.066 3.066 0 001.745.723 3.066 3.066 0 012.812 2.812c.051.643.304 1.254.723 1.745a3.066 3.066 0 010 3.976 3.066 3.066 0 00-.723 1.745 3.066 3.066 0 01-2.812 2.812 3.066 3.066 0 00-1.745.723 3.066 3.066 0 01-3.976 0 3.066 3.066 0 00-1.745-.723 3.066 3.066 0 01-2.812-2.812 3.066 3.066 0 00-.723-1.745 3.066 3.066 0 010-3.976 3.066 3.066 0 00.723-1.745 3.066 3.066 0 012.812-2.812zm7.44 5.252a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
              clipRule="evenodd"
            />
          </svg>
          <span className="text-sm font-medium">1M+ Patient Records</span>
        </div>

        <div className="flex items-center gap-2 text-slate-400">
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M2.166 4.999A11.954 11.954 0 0010 1.944 11.954 11.954 0 0017.834 5c.11.65.166 1.32.166 2.001 0 5.225-3.34 9.67-8 11.317C5.34 16.67 2 12.225 2 7c0-.682.057-1.35.166-2.001zm11.541 3.708a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
              clipRule="evenodd"
            />
          </svg>
          <span className="text-sm font-medium">WHO/CDC Verified</span>
        </div>

        <div className="flex items-center gap-2 text-slate-400">
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
            <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span className="text-sm font-medium">Medical Expert Validated</span>
        </div>
      </div>
    </div>
  );
}
