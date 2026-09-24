import type { ReactNode } from "react";
import "./globals.css";
import Sidebar from "./components/Sidebar";
import AuthGate from "./components/AuthGate";

export const metadata = { title: "AirSentinel", description: "Air Quality Intelligence" };

export default function RootLayout({ children }: { children: ReactNode })  {
  return (
    <html lang="en">
      <body className="bg-[#0D0F12] text-[#E8EDEE] min-h-screen">
        <AuthGate>
          <div className="flex min-h-screen">
            <Sidebar />
            <main className="flex-1 p-8">{children}</main>
          </div>
        </AuthGate>
      </body>
    </html>
  );
}
