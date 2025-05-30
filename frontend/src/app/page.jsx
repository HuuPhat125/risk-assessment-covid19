import HeroSection from "../components/home/HeroSection";
import HowItHelpsSection from "../components/home/HowItHelps";
import ExploreDataSection from "../components/home/ExploreData";
import TrustSection from "../components/home/TrustSection";

export default function Home() {
  return (
    <main>
      {/* Hero Section - COVID-19 Risk Assessment (About content) */}
      <section className="max-w-7xl mx-auto px-4 py-8">
        <HeroSection />
      </section>

      {/* How It Helps Section */}
      <section className="max-w-7xl mx-auto px-4 py-8" id="howithelp">
        <HowItHelpsSection />
      </section>

      {/* Explore Data Section */}
      <section className="max-w-7xl mx-auto px-4 py-8">
        <ExploreDataSection />
      </section>

      {/* Trust Section - Full width background với content centered */}
      <section className="w-full bg-slate-100">
        <div className="max-w-7xl mx-auto">
          <TrustSection />
        </div>
      </section>
    </main>
  );
}
