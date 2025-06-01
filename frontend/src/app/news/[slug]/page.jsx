// app/news/[slug]/page.jsx

"use client";

import { useEffect, useState } from "react";
import { notFound } from "next/navigation";

export default function NewsArticlePage({ params }) {
  const { slug } = params;
  const [article, setArticle] = useState(null);

  useEffect(() => {
    async function fetchArticle() {
      const res = await fetch("http://localhost:8000/news"); // fetch toàn bộ
      const data = await res.json();
      const found = data.find((item) =>
        encodeURIComponent(item.link).includes(slug)
      );
      if (!found) {
        notFound();
      }
      setArticle(found);
    }
    fetchArticle();
  }, [slug]);

  if (!article) return <div>Loading...</div>;

  return (
    <div className="container mx-auto p-4 max-w-4xl">
      <h1 className="text-3xl md:text-4xl font-bold text-gray-800 mb-4 leading-tight">
        {article.title}
      </h1>
      <div className="text-sm text-gray-500 mb-6">
        By {article.source} - {new Date(article.published).toLocaleDateString()}
      </div>

      {article.image && (
        <div className="relative w-full h-64 md:h-80 mb-6 rounded-lg overflow-hidden">
          <img
            src={article.image}
            alt={article.title}
            className="w-full h-full object-cover"
          />
        </div>
      )}

      <div
        className="prose max-w-none"
        dangerouslySetInnerHTML={{ __html: article.description }}
      />
      <p className="mt-8 text-blue-600">
        <a href={article.link} target="_blank" rel="noopener noreferrer">
          Đọc bản gốc
        </a>
      </p>
    </div>
  );
}
