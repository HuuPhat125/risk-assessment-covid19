"use client";

import { useState } from "react";
import { Menu, X } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

export default function Navbar() {
  const [menuOpen, setMenuOpen] = useState(false);
  const pathname = usePathname();

  const navLinks = [
    { href: "/", label: "Home" },
    { href: "/dataset", label: "Dataset" },
    { href: "/prediction", label: "Prediction" },
    { href: "/map", label: "Map" },
    { href: "/news", label: "News" },
  ];

  return (
    <header className="w-full px-4 sm:px-6 lg:px-8 py-3">
      {/* Container chính với max-width và auto layout */}
      <div className="max-w-7xl mx-auto">
        {/* Thanh điều hướng */}
        <div className="flex justify-between items-center w-full">
          {/* Logo section */}
          <Link href="/" className="flex items-center flex-shrink-0">
            <div className="w-16 h-16 flex justify-center items-center px-3 py-3">
              <svg
                className="w-16 h-16 text-red-400"
                fill="currentColor"
                viewBox="0 0 24 24"
              >
                <path d="M10.5,13H8v-3h2.5V7.5h3V10H16v3h-2.5v2.5h-3V13z M12,2L4,5v6.09c0,5.05,3.41,9.76,8,10.91c4.59-1.15,8-5.86,8-10.91V5L12,2 z" />
              </svg>
            </div>
            <span className="font-semibold text-xl text-indigo-900">
              HealthGuys
            </span>
          </Link>

          {/* Menu cho desktop - hidden trên mobile */}
          <nav className="hidden lg:flex items-center space-x-8 xl:space-x-12">
            {navLinks.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className={`font-semibold text-lg transition-colors duration-200 ${
                  pathname === link.href
                    ? "text-indigo-900"
                    : "text-gray-600 hover:text-indigo-900"
                }`}
              >
                {link.label}
              </Link>
            ))}
          </nav>

          {/* Nút gọi hành động - responsive */}
          <div className="hidden md:flex">
            <button className="flex justify-center items-center gap-2.5 px-4 py-2.5 rounded-xl border border-solid border-gray-300 hover:border-gray-400 hover:bg-gray-100 transition-all duration-200">
              <span className="font-bold text-lg text-blue-900">
                Get started
              </span>
            </button>
          </div>

          {/* Nút menu mobile */}
          <button
            className="lg:hidden p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-all duration-200"
            onClick={() => setMenuOpen(!menuOpen)}
            aria-label="Toggle menu"
          >
            {menuOpen ? <X size={28} /> : <Menu size={28} />}
          </button>
        </div>

        {/* Đường gạch chân */}
        <div className="h-px bg-gray-500 mt-3" />

        {/* Menu cho mobile - với animation */}
        <div
          className={`lg:hidden overflow-hidden transition-all duration-300 ease-in-out ${
            menuOpen ? "max-h-96 opacity-100" : "max-h-0 opacity-0"
          }`}
        >
          <nav className="py-2 space-y-3">
            {navLinks.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className={`block font-semibold text-lg py-2 px-2 rounded-lg transition-colors duration-200 ${
                  pathname === link.href
                    ? "text-indigo-900 bg-indigo-100"
                    : "text-gray-600 hover:bg-gray-100 hover:text-gray-900"
                }`}
                onClick={() => setMenuOpen(false)}
              >
                {link.label}
              </Link>
            ))}

            {/* CTA button for mobile */}
            <div className="pt-3 mt-3 border-t border-gray-300">
              <button
                className="w-full flex justify-center items-center gap-2.5 px-4 py-2.5 rounded-xl border border-solid border-gray-300 hover:border-gray-400 hover:bg-gray-100 transition-all duration-200"
                onClick={() => setMenuOpen(false)}
              >
                <span className="font-bold text-lg text-blue-900">
                  Get started
                </span>
              </button>
            </div>
          </nav>
        </div>
      </div>
    </header>
  );
}
