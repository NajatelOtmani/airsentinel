"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";

type SensorReading = {
  location_id: string;
  pm25: number;
  epa_aqi: number;
  aqi_category: string;
  latitude: number;
  longitude: number;
};

function aqiBadge(aqi: number) {
  const color = aqi <= 50 ? "bg-emerald-100 text-emerald-700"
    : aqi <= 100 ? "bg-amber-100 text-amber-700"
    : "bg-red-100 text-red-700";
  return <span className={`px-2.5 py-1 rounded-full text-xs font-semibold ${color}`}>AQI {aqi}</span>;
}

export default function OverviewPage() {
  const [readings, setReadings] = useState<SensorReading[]>([]);
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  useEffect(() => {
    // 1. Verify access token presence
    const token = localStorage.getItem("token");
    if (!token) {
      router.push("/login");
      return;
    }

    // 2. Fetch sensor data
    (async () => {
      try {
        const { sensors } = await api.getSensors();
        const results = await Promise.all(
          sensors.map((id: string) => api.getSensorLatest(id).catch(() => null))
        );
        setReadings(results.filter((r) => r && !r.error));
      } catch (err) {
        console.error("Failed to load overview telemetry:", err);
      } finally {
        setLoading(false);
      }
    })();
  }, [router]);

  if (loading) return <div className="text-slate-500 p-8">Loading network data...</div>;

  const avgPm25 = readings.reduce((s, r) => s + r.pm25, 0) / (readings.length || 1);
  const avgAqi = Math.round(readings.reduce((s, r) => s + r.epa_aqi, 0) / (readings.length || 1));
  const worst = readings.reduce((a, b) => (a.epa_aqi > b.epa_aqi ? a : b), readings[0]);

  return (
    <div className="p-8">
      <div className="flex items-center gap-2 text-xs text-slate-500 mb-2">
        <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" /> Live data
      </div>
      <h1 className="text-4xl font-bold text-slate-900 mb-1">Air Quality Overview</h1>
      <p className="text-slate-500 mb-8">Last 24 hours · London monitoring network</p>

      <div className="grid grid-cols-4 gap-4 mb-8">
        <div className="bg-white rounded-2xl p-5 shadow-sm border border-slate-100">
          <div className="text-xs text-slate-500 mb-1">Network Avg PM2.5</div>
          <div className="text-2xl font-bold text-slate-900">{avgPm25.toFixed(1)} µg/m³</div>
          <div className="mt-2">{aqiBadge(avgAqi)}</div>
        </div>
        <div className="bg-white rounded-2xl p-5 shadow-sm border border-slate-100">
          <div className="text-xs text-slate-500 mb-1">Sensors Online</div>
          <div className="text-2xl font-bold text-slate-900">{readings.length}</div>
          <div className="text-xs text-slate-400 mt-2">Across London zones</div>
        </div>
        <div className="bg-white rounded-2xl p-5 shadow-sm border border-slate-100">
          <div className="text-xs text-slate-500 mb-1">Worst Zone</div>
          <div className="text-lg font-bold text-slate-900 truncate">{worst?.location_id}</div>
          <div className="mt-2">{worst && aqiBadge(worst.epa_aqi)}</div>
        </div>
        <div className="bg-white rounded-2xl p-5 shadow-sm border border-slate-100">
          <div className="text-xs text-slate-500 mb-1">Active Anomalies</div>
          <div className="text-2xl font-bold text-slate-900">0</div>
          <div className="text-xs text-slate-400 mt-2">Right now</div>
        </div>
      </div>

      <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-100">
        <h2 className="text-lg font-semibold text-slate-900 mb-4">Pollutant Breakdown by Zone</h2>
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-slate-400 border-b border-slate-100">
              <th className="pb-2">Zone</th>
              <th className="pb-2">PM2.5</th>
              <th className="pb-2">AQI</th>
              <th className="pb-2">Category</th>
            </tr>
          </thead>
          <tbody>
            {readings.map((r) => (
              <tr key={r.location_id} className="border-b border-slate-50">
                <td className="py-2 font-medium text-slate-800">{r.location_id}</td>
                <td className="py-2 text-slate-600">{r.pm25} µg/m³</td>
                <td className="py-2">{aqiBadge(r.epa_aqi)}</td>
                <td className="py-2 text-slate-600">{r.aqi_category}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}