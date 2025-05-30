import Link from 'next/link';

export default function HeroSection() {
  return (
    <div className="flex flex-col xl:flex-row justify-between items-center  px-4 sm:px-6 lg:px-8 py-8 sm:py-12 lg:py-16 bg-slate-200 rounded-3xl">
      {/* Left Content */}
      <div className="flex-1 max-w-xl flex flex-col gap-6 lg:gap-12 text-center xl:text-left m-4">
        {/* Text Content */}
        <div className="flex flex-col gap-4 lg:gap-6">
          <h1 className="font-bold text-3xl sm:text-4xl md:text-5xl xl:text-6xl text-blue-900 leading-tight">
            COVID-19 Risk Assessment Tool
          </h1>
          <p className="font-medium text-sm sm:text-base lg:text-lg text-gray-500 leading-relaxed ">
            A set of solution designed to help quickly identify coronavirus
            symptoms and get reliable information regarding COVID-19 concerns
          </p>
        </div>

        {/* CTA Button */}
        <div className="flex justify-center xl:justify-start">
          <Link href="#howithelp">
            <button className="inline-flex justify-center items-center gap-3 px-4 sm:px-6 py-3 sm:py-4 bg-red-400 hover:bg-red-500 rounded-xl transition-colors duration-200 group">
              <span className="font-extrabold text-base sm:text-lg text-white whitespace-nowrap">
                See how to use it
              </span>
              <svg
                className="w-4 h-4 sm:w-5 sm:h-5 text-white group-hover:translate-x-1 transition-transform duration-200"
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

      {/* Right Content - Image Grid */}
      <div className="m-4 flex-shrink-0 flex gap-1.5 sm:gap-2">
        {/* Left Grid - 2x3 small images */}
        <div className="flex flex-col gap-1.5 sm:gap-2">
          <div className="flex gap-1.5 sm:gap-2">
            <img src="/hero-section-grid-image-1.jpg" alt="COVID-19" className="w-16 h-16 sm:w-20 sm:h-20 md:w-24 md:h-24 lg:w-28 lg:h-28 bg-gray-300 rounded-lg" />
            <img src="/hero-section-grid-image-2.jpg" alt="COVID-19" className="w-16 h-16 sm:w-20 sm:h-20 md:w-24 md:h-24 lg:w-28 lg:h-28 bg-gray-300 rounded-lg" />
          </div>
          <div className="flex gap-1.5 sm:gap-2">
            <img src="/hero-section-grid-image-3.jpg" alt="COVID-19" className="w-16 h-16 sm:w-20 sm:h-20 md:w-24 md:h-24 lg:w-28 lg:h-28 bg-gray-300 rounded-lg" />
            <img src="/hero-section-grid-image-4.jpg" alt="COVID-19" className="w-16 h-16 sm:w-20 sm:h-20 md:w-24 md:h-24 lg:w-28 lg:h-28 bg-gray-300 rounded-lg" />
          </div>
          <div className="flex gap-1.5 sm:gap-2">
            <img src="/hero-section-grid-image-5.jpg" alt="COVID-19" className="w-16 h-16 sm:w-20 sm:h-20 md:w-24 md:h-24 lg:w-28 lg:h-28 bg-gray-300 rounded-lg" />
            <img src="/hero-section-grid-image-6.jpg" alt="COVID-19" className="w-16 h-16 sm:w-20 sm:h-20 md:w-24 md:h-24 lg:w-28 lg:h-28 bg-gray-300 rounded-lg" />
          </div>
        </div>

        {/* Right Tall Image */}
        <img src="/hero-section-grid-image-7.jpg" alt="COVID-19" className="w-16 sm:w-20 md:w-24 lg:w-28 h-64 sm:h-42 md:h-52 lg:h-60 rounded-lg object-cover" />
      </div>
    </div>
  );
}
