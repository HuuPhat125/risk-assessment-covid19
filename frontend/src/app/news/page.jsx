// app/news/page.jsx
"use client";

import { useEffect, useState } from "react";
import NewsCard from "@/components/news/NewsCard";

export default function NewsPage() {
  const [articles, setArticles] = useState([]);

  useEffect(() => {
    async function fetchNews() {
      const res = await fetch("http://localhost:8000/news"); // thay bằng URL thực tế
      const data = await res.json();
      setArticles(data);
    }
    fetchNews();
  }, []);
  console.log(articles.data);
  return (
    <div className="container mx-auto p-4">
      <h1 className="text-4xl font-bold text-center my-8">
        Latest COVID-19 News
      </h1>
      <p className="text-center text-gray-600 mb-12">
        Stay updated with accurate and recent coronavirus information.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
        {articles.data?.map((article) => (
          <NewsCard key={article.link} article={article} />
        ))}
      </div>
    </div>
  );
}
