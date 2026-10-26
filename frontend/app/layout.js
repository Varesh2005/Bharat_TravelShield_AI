import "./globals.css";

export const metadata = {
  title: "Bharat_Travelshield_Ai | Spatiotemporal Risk Intelligence",
  description:
    "India-First Spatiotemporal Travel Health & Natural Disaster Risk Intelligence Platform. Multi-hazard risk prediction for Malaria, Heatwave, Flood, and Dengue across Indian districts.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-100 text-slate-900 antialiased">
        {children}
      </body>
    </html>
  );
}
