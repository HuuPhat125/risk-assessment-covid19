import "../styles/globals.css";
import Navbar from "../components/home/Navbar";
import Footer from "../components/home/Footer";

export default function RootLayout({ children }) {
  return (
    <html lang="en" className="mdl-js">
      <body className="bg-gray-50 text-gray-800">
        <Navbar />
        <main className="mx-auto p-4">{children}</main>
        <Footer />
      </body>
    </html>
  );
}
