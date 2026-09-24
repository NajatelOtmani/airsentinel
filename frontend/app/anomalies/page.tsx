"use client";

import { useEffect, useState } from "react";

interface AnomalyAlert {
  id: string;
  sensor_id: string;
  location: string;
  pollutant: string;
  value: number;
  threshold: number;
  severity: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  timestamp: string;
}

export default function AnomaliesPage() {
  const [alerts, setAlerts] = useState<AnomalyAlert[]>([]);
  const [wsStatus, setWsStatus] = useState<"connecting" | "connected" | "disconnected">("connecting");

  useEffect(() => {
    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/ws/alerts";
    let ws: WebSocket;

    const connectWebSocket = () => {
      setWsStatus("connecting");
      ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        setWsStatus("connected");
      };

      ws.onmessage = (event) => {
        try {
          const data: AnomalyAlert = JSON.parse(event.data);
          setAlerts((prev) => [data, ...prev.slice(0, 49)]); // Keep latest 50 alerts
        } catch (err) {
          console.error("Failed to parse WebSocket message:", err);
        }
      };

      ws.onerror = (error) => {
        console.error("WebSocket Error:", error);
      };

      ws.onclose = () => {
        setWsStatus("disconnected");
        // Reconnect after 3 seconds
        setTimeout(connectWebSocket, 3000);
      };
    };

    connectWebSocket();

    return () => {
      if (ws) ws.close();
    };
  }, []);

  const getSeverityBadge = (severity: AnomalyAlert["severity"]) => {
    const styles = {
      LOW: "bg-blue-900/40 text-blue-400 border-blue-700",
      MEDIUM: "bg-yellow-900/40 text-yellow-400 border-yellow-700",
      HIGH: "bg-orange-900/40 text-orange-400 border-orange-700",
      CRITICAL: "bg-red-900/40 text-red-400 border-red-700",
    };
    return (
      <span className={`px-2.5 py-1 text-xs font-semibold rounded-md border ${styles[severity] || styles.LOW}`}>
        {severity}
      </span>
    );
  };

  return (
    <div className="p-8 space-y-6 text-slate-100">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Real-Time Anomaly Detector</h1>
          <p className="text-slate-400 text-sm mt-1">Live WebSocket detection feed from AirSentinel ML engine</p>
        </div>
        <div className="flex items-center space-x-2 bg-slate-900 px-3 py-1.5 rounded-lg border border-slate-800">
          <span
            className={`h-2.5 w-2.5 rounded-full ${
              wsStatus === "connected"
                ? "bg-emerald-500 animate-pulse"
                : wsStatus === "connecting"
                ? "bg-amber-500 animate-ping"
                : "bg-rose-500"
            }`}
          />
          <span className="text-xs capitalize text-slate-300">WS: {wsStatus}</span>
        </div>
      </div>

      {/* Real-time Feed Table */}
      <div className="bg-slate-900/80 rounded-xl border border-slate-800 overflow-hidden shadow-xl backdrop-blur-sm">
        <div className="p-4 border-b border-slate-800 flex justify-between items-center">
          <h2 className="text-lg font-semibold text-slate-200">Alert Stream</h2>
          <span className="text-xs text-slate-500">{alerts.length} events received</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm text-slate-300">
            <thead className="bg-slate-950/60 text-slate-400 uppercase text-xs font-medium border-b border-slate-800">
              <tr>
                <th className="px-6 py-3.5">Timestamp</th>
                <th className="px-6 py-3.5">Sensor ID</th>
                <th className="px-6 py-3.5">Location</th>
                <th className="px-6 py-3.5">Pollutant</th>
                <th className="px-6 py-3.5">Measured</th>
                <th className="px-6 py-3.5">Threshold</th>
                <th className="px-6 py-3.5">Severity</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {alerts.length === 0 ? (
                <tr>
                  <td colSpan={7} className="px-6 py-12 text-center text-slate-500">
                    Listening for real-time anomaly telemetry...
                  </td>
                </tr>
              ) : (
                alerts.map((alert, idx) => (
                  <tr key={alert.id || idx} className="hover:bg-slate-800/40 transition-colors">
                    <td className="px-6 py-4 font-mono text-xs text-slate-400">
                      {new Date(alert.timestamp).toLocaleTimeString()}
                    </td>
                    <td className="px-6 py-4 font-medium text-slate-200">{alert.sensor_id}</td>
                    <td className="px-6 py-4 text-slate-400">{alert.location}</td>
                    <td className="px-6 py-4 font-semibold text-slate-300">{alert.pollutant}</td>
                    <td className="px-6 py-4 font-mono text-rose-400 font-bold">{alert.value} µg/m³</td>
                    <td className="px-6 py-4 font-mono text-slate-400">{alert.threshold} µg/m³</td>
                    <td className="px-6 py-4">{getSeverityBadge(alert.severity)}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}