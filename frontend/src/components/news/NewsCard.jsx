import Link from 'next/link';
import Image from 'next/image'; // Sử dụng component Image của Next.js để tối ưu ảnh

export default function NewsCard({ article }) {
  return (
    <Link href={`/news/${article.slug}`} className="block bg-white rounded-lg shadow-md hover:shadow-xl transition-shadow duration-300 overflow-hidden">
      <div className="relative w-full h-48">
        {/* Sử dụng component Image của Next.js */}
        <Image
          src={article.image}
          alt={article.title}
          layout="fill" // Giúp ảnh lấp đầy container
          objectFit="cover" // Đảm bảo tỷ lệ ảnh không bị méo
        />
      </div>
      <div className="p-4">
        {/* Thêm tag nguồn hoặc ngày tháng ở đây nếu cần */}
        <div className="text-sm text-gray-500 mb-2">
          {article.source} - {article.date}
        </div>
        <h3 className="text-xl font-semibold text-gray-800 mb-2 leading-tight">
          {article.title}
        </h3>
        <p className="text-gray-600 text-sm mb-4">
          {article.description}
        </p>
        {/* Nút hoặc liên kết "Read more" */}
        <div className="flex justify-end">
           <span className="text-blue-600 hover:underline font-medium text-sm">Read more &rarr;</span>
        </div>
      </div>
    </Link>
  );
}
