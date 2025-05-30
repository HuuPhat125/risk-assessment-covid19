import {
  Heart,
  Facebook,
  Twitter,
  Linkedin,
  Instagram,
  Github,
} from "lucide-react";

export default function Footer() {
  const socialLinks = [
    {
      name: "Facebook",
      icon: <Facebook className="w-5 h-5" />,
      href: "#",
      color: "hover:text-blue-600",
    },
    {
      name: "Twitter",
      icon: <Twitter className="w-5 h-5" />,
      href: "#",
      color: "hover:text-blue-400",
    },
    {
      name: "LinkedIn",
      icon: <Linkedin className="w-5 h-5" />,
      href: "#",
      color: "hover:text-blue-700",
    },
    {
      name: "Instagram",
      icon: <Instagram className="w-5 h-5" />,
      href: "#",
      color: "hover:text-pink-500",
    },
    {
      name: "GitHub",
      icon: <Github className="w-5 h-5" />,
      href: "#",
      color: "hover:text-gray-800",
    },
  ];

  const footerLinks = [
    {
      title: "Product",
      links: [
        { name: "Explore Data", href: "/dataset" },
        { name: "Map View", href: "/map" },
        { name: "News Updates", href: "/news" },
        { name: "Predictions", href: "/prediction" },
      ],
    },
    {
      title: "Support",
      links: [
        { name: "Help Center", href: "#" },
        { name: "Contact Us", href: "#" },
        { name: "Privacy Policy", href: "#" },
        { name: "Terms of Service", href: "#" },
      ],
    },
  ];

  return (
    <footer className="w-full bg-gray-200 border-t border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 lg:py-6">
        {/* Main Footer Content */}
        <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4 lg:gap-8">
          {/* Left Section - Brand & Disclaimer */}
          <div className="flex-1 max-w-md flex flex-col gap-2 lg:gap-3">
            {/* Brand */}
            <div className="flex flex-col gap-1">
              <div className="flex items-center gap-2">
                <div className="w-10 h-10 flex items-center justify-center bg-red-50 rounded-lg">
                  <Heart className="w-5 h-5 text-red-400" />
                </div>
                <span className="font-semibold text-base text-indigo-800">
                  HealthGuys
                </span>
              </div>
            </div>
          </div>

          {/* Center Section - Links */}
          <div className="flex-1 grid grid-cols-1 sm:grid-cols-3 gap-4 lg:gap-6 w-full lg:max-w-2xl">
            {footerLinks.map((section, index) => (
              <div key={index} className="flex flex-col gap-2">
                <h3 className="font-semibold text-xs text-gray-900 uppercase tracking-wide">
                  {section.title}
                </h3>
                <ul className="flex flex-col gap-1">
                  {section.links.map((link, linkIndex) => (
                    <li key={linkIndex}>
                      <a
                        href={link.href}
                        className="text-xs text-gray-600 hover:text-blue-600 transition-colors duration-200"
                      >
                        {link.name}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>

          {/* Right Section - Social Links */}
          <div className="flex-shrink-0 flex flex-col items-center lg:items-end gap-2">
            <h3 className="font-semibold text-xs text-gray-900 uppercase tracking-wide">
              Follow Us
            </h3>
            <div className="flex items-center gap-2">
              {socialLinks.map((social, index) => (
                <a
                  key={index}
                  href={social.href}
                  className={`w-8 h-8 flex items-center justify-center rounded-full bg-gray-100 text-gray-600 transition-all duration-200 hover:bg-gray-200 ${social.color}`}
                  aria-label={social.name}
                >
                  {social.icon}
                </a>
              ))}
            </div>

            {/* Newsletter Signup */}
            <div className="flex flex-col gap-1 mt-2">
              <span className="text-xs text-gray-500 text-center lg:text-right">
                Get health updates
              </span>
              <div className="flex gap-1">
                <input
                  type="email"
                  placeholder="Enter email"
                  className="px-2 py-1 text-xs border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
                <button className="px-3 py-1 bg-blue-600 text-white text-xs rounded-lg hover:bg-blue-700 transition-colors duration-200">
                  Subscribe
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="mt-4 pt-3 border-t border-gray-200">
            <div className="flex items-center gap-3 text-xs text-gray-500">
              <a href="#" className="hover:text-gray-700 transition-colors">
                Privacy
              </a>
              <a href="#" className="hover:text-gray-700 transition-colors">
                Terms
              </a>
              <a href="#" className="hover:text-gray-700 transition-colors">
                Cookies
              </a>
            </div>
        </div>
      </div>
    </footer>
  );
}
