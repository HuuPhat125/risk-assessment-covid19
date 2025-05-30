import Link from 'next/link';
import NewsCard from '@/components/news/NewsCard';

// Dữ liệu tin tức tĩnh (có thể di chuyển ra file riêng sau)
const newsArticles = [
  {
    slug: 'vaccine-variants-promise',
    title: 'Vaccine Variants Promise',
    description: 'Scientists are optimistic about vaccines adapting to new variants.',
    image: '/placeholder-news-1.jpg', // Thay bằng đường dẫn ảnh thực tế trong public
    date: 'October 26, 2023',
    source: 'Health Today',
    content: '<p>Nội dung chi tiết bài báo 1...</p>'
  },
  {
    slug: 'covid-19-restpara-10',
    title: 'Healtl Coalasehoal COVID-19 Restpare 10 Fostent',
    description: 'Expert discussion on the latest COVID-19 recovery data.',
    image: '/placeholder-news-2.jpg', // Thay bằng đường dẫn ảnh thực tế trong public
    date: 'October 25, 2023',
    source: 'Global Health News',
    content: '<p>Nội dung chi tiết bài báo 2...</p>'
  },
  {
    slug: 'latest-policy-updates',
    title: 'Hlore Covereys Polirvee',
    description: 'Overview of recent health policy changes.',
    image: '/placeholder-news-3.jpg', // Thay bằng đường dẫn ảnh thực tế trong public
    date: 'October 24, 2023',
    source: 'Policy Watch',
    content: '<p>Nội dung chi tiết bài báo 3...</p>'
  },
    {
    slug: 'testing-day-outcome',
    title: 'Oodle One Ifate poatiele be bolretom Pessio COrD Da yat ibotier 200',
    description: 'Analysis of the outcomes from the latest COVID testing day.',
    image: '/placeholder-news-4.jpg', // Thay bằng đường dẫn ảnh thực tế trong public
    date: 'October 23, 2023',
    source: 'Data Insights',
    content: '<p>Nội dung chi tiết bài báo 4...</p>'
  },
      {
    slug: 'new-treatment-approaches',
    title: 'Miort Ofttee Dutòckecutt Dithe Selte Ulamg Rolw\'b bdlewoe',
    description: 'Examining promising new methods for treating COVID-19.',
    image: '/placeholder-news-5.jpg', // Thay bằng đường dẫn ảnh thực tế trong public
    date: 'October 22, 2023',
    source: 'Research Daily',
    content: '<p>Nội dung chi tiết bài báo 5...</p>'
  },
        {
    slug: 'global-response-efforts',
    title: 'Miort Ofttee Dutòckecutt Dithe Selte Ulamg Rolw\'b bdlewoe',
    description: 'A look at coordinated international efforts against the virus.',
    image: '/placeholder-news-6.jpg', // Thay bằng đường dẫn ảnh thực tế trong public
    date: 'October 21, 2023',
    source: 'World News',
    content: '<p>Nội dung chi tiết bài báo 6...</p>'
  }
];

export default function NewsPage() {
  return (
    <div className="container mx-auto p-4">
      <h1 className="text-4xl font-bold text-center my-8">Latest COVID-19 Recent Coronavirus News</h1>
      <p className="text-center text-gray-600 mb-12">Stay updated with accurate and recent coronavirus information.</p>

      {/* Grid tin tức */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
        {newsArticles.map(article => (
          <NewsCard key={article.slug} article={article} />
        ))}
      </div>
    </div>
  );
}
