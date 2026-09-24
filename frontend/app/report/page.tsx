"use client";
import { useState } from "react";
import { api } from "@/lib/api";

export default function ReportPage() {
  const [report, setReport] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const generate = async () => {
    setLoading(true);
    try {
      const res = await api.generateReport("london_central");
      setReport(res.report || "Report generated successfully.");
    } catch (err: any) {
      setReport("Failed to generate report: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-3xl">
      <h1 className="text-4xl font-bold text-slate-900 mb-2">Daily Intelligence Report</h1>
      <p className="text-slate-500 mb-8">Generate an automated air quality summary report</p>

      <button onClick={generate} disabled={loading} className="bg-teal-600 hover:bg-teal-500 text-white font-medium px-6 py-3 rounded-xl transition-colors mb-6">
        {loading ? "Generating..." : "Generate Daily Summary"}
      </button>

      {report && (
        <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-100 text-slate-800 whitespace-pre-wrap">
          {report}
        </div>
      )}
    </div>
  );
}