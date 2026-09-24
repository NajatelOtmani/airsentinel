"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { LayoutGrid, Activity, MessageSquare, FileText, Wind } from "lucide-react";

const nav = [
  { href: "/", label: "Overview", icon: LayoutGrid },
  { href: "/anomalies", label: "Anomalies", icon: Activity },
  { href: "/analyst", label: "AI Analyst", icon: MessageSquare },
  { href: "/report", label: "Daily Report", icon: FileText },
];

export default function Sidebar() {
  const pathname = usePathname();
  return (
    <aside className="w-64 min-h-screen bg-[#0F1D22] text-slate-200 flex flex-col p-4 border-r border-white/5">
      <div className="flex items-center gap-2 px-2 py-3 mb-6">
        <div className="bg-teal-900/40 p-2 rounded-lg">
          <Wind className="w-5 h-5 text-teal-300" />
        </div>
        <div>
          <div className="font-semibold text-white text-sm leading-tight">AirSentinel</div>
          <div className="text-[10px] text-slate-400 tracking-wide">AIR INTELLIGENCE</div>
        </div>
      </div>

      <nav className="flex flex-col gap-1">
        {nav.map(({ href, label, icon: Icon }) => {
          const active = pathname === href;
          return (
            <Link
              key={href}
              href={href}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                active ? "bg-teal-800/40 text-white" : "text-slate-400 hover:bg-white/5 hover:text-white"
              }`}
            >
              <Icon className="w-4 h-4" />
              {label}
            </Link>
          );
        })}
      </nav>

      <div className="mt-auto bg-white/5 rounded-xl p-4 text-xs">
        <div className="flex items-center gap-2 text-emerald-400 font-medium mb-1">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
          LIVE NETWORK
        </div>
        <div className="text-slate-400">Streaming · updated live</div>
      </div>
    </aside>
  );
}