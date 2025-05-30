import Image from 'next/image';
import { notFound } from 'next/navigation';

// Dữ liệu tin tức tĩnh (phải được import hoặc định nghĩa lại ở đây)
// Tạm thời copy lại từ news/page.jsx để dễ làm việc
const newsArticles = [
  {
    slug: 'vaccine-variants-promise',
    title: 'Vaccine Variants Promise',
    description: 'Scientists are optimistic about vaccines adapting to new variants.',
    image: '/placeholder-news-1.jpg', // Thay bằng đường dẫn ảnh thực tế
    date: 'October 26, 2023',
    source: 'Health Today',
    content: '<p>Nội dung chi tiết bài báo 1. Đây là nơi bạn sẽ đặt toàn bộ nội dung của bài viết. Bạn có thể sử dụng HTML tags thông thường trong chuỗi content này. Ví dụ: <strong>đoạn in đậm</strong>, <em>đoạn in nghiêng</em>, <a href="#">liên kết</a>, v.v. Hãy tưởng tượng đây là nội dung đầy đủ mà người đọc sẽ thấy khi bấm vào tiêu đề bài báo.</p><p>Bạn có thể thêm nhiều đoạn văn bản, hình ảnh, hoặc bất kỳ nội dung HTML nào khác vào đây.</p>'
  },
  {
    slug: 'covid-19-restpara-10',
    title: 'Healtl Coalasehoal COVID-19 Restpare 10 Fostent',
    description: 'Expert discussion on the latest COVID-19 recovery data.',
    image: '/placeholder-news-2.jpg', // Thay bằng đường dẫn ảnh thực tế
    date: 'October 25, 2023',
    source: 'Global Health News',
    content: '<p>Nội dung chi tiết bài báo 2...</p>'
  },
  {
    slug: 'latest-policy-updates',
    title: 'Hlore Covereys Polirvee',
    description: 'Overview of recent health policy changes.',
    image: '/placeholder-news-3.jpg', // Thay bằng đường dẫn ảnh thực tế
    date: 'October 24, 2023',
    source: 'Policy Watch',
    content: '<p>Nội dung chi tiết bài báo 3...</p>'
  },
    {
    slug: 'testing-day-outcome',
    title: 'Oodle One Ifate poatiele be bolretom Pessio COrD Da yat ibotier 200',
    description: 'Analysis of the outcomes from the latest COVID testing day.',
    image: '/placeholder-news-4.jpg', // Thay bằng đường dẫn ảnh thực tế
    date: 'October 23, 2023',
    source: 'Data Insights',
    content: '<p>Nội dung chi tiết bài báo 4...</p>'
  },
      {
    slug: 'new-treatment-approaches',
    title: 'Miort Ofttee Dutòckecutt Dithe Selte Ulamg Rolw\'b bdlewoe',
    description: 'Examining promising new methods for treating COVID-19.',
    image: '/placeholder-news-5.jpg', // Thay bằng đường dẫn ảnh thực tế
    date: 'October 22, 2023',
    source: 'Research Daily',
    content: '<p>Nội dung chi tiết bài báo 5...</p>'
  },
        {
    slug: 'global-response-efforts',
    title: 'Miort Ofttee Dutòckecutt Dithe Selte Ulamg Rolw\'b bdlewoe',
    description: 'A look at coordinated international efforts against the virus.',
    image: '/placeholder-news-6.jpg', // Thay bằng đường dẫn ảnh thực tế
    date: 'October 21, 2023',
    source: 'World News',
    content: '<p>Nội dung chi tiết bài báo 6...</p>'
  }
];

// Hàm để tạo ra các đường dẫn tĩnh cho các bài báo
export async function generateStaticParams() {
  return newsArticles.map(article => ({
    slug: article.slug,
  }));
}

export default function NewsArticlePage({ params }) {
  const { slug } = params;

  // Tìm bài báo dựa trên slug
  const article = newsArticles.find(item => item.slug === slug);

  // Nếu không tìm thấy bài báo, hiển thị trang 404
  if (!article) {
    notFound();
  }

  return (
    <div className="container mx-auto p-4 max-w-4xl">
      <h1 className="text-3xl md:text-4xl font-bold text-gray-800 mb-4 leading-tight">{article.title}</h1>
      <div className="text-sm text-gray-500 mb-6">
        By {article.source} - {article.date}
      </div>

      {/* Hiển thị ảnh bài viết nếu có */}
      {article.image && (
        <div className="relative w-full h-64 md:h-80 mb-6 rounded-lg overflow-hidden">
           <Image
            src={article.image}
            alt={article.title}
            layout="fill"
            objectFit="cover"
          />
        </div>
      )}

      {/* Nội dung bài viết (sử dụng dangerouslySetInnerHTML vì content là chuỗi HTML) */}
      <div
        className="prose max-w-none" // Sử dụng class prose của Tailwind typography để định dạng nội dung HTML
        dangerouslySetInnerHTML={{ __html: article.content }}
      />
    </div>
  );
}
