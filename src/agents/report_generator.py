"""
src/agents/report_generator.py
==============================
DAY 14 — Auto-Report Generator
------------------------------
Multi-step synthesis: Anomalies -> RAG Context -> 12h Forecast -> Report.
Includes APScheduler background script for daily report scheduling.

FIXES APPLIED:
  1. generate_report() now wraps agent.run() in try/except. A failure
     (Groq error, parsing failure, whatever) returns a structured error
     dict instead of raising — which is what was turning into the
     unhandled 500 on LONDON_KC1, since nothing upstream was catching it.
  2. AirSentinelAgent is no longer created once in __init__ and reused
     across every report. A fresh agent (fresh memory) is created per
     generate_report() call. The old shared instance's
     ConversationBufferWindowMemory was persisting between requests, so
     e.g. a LONDON_HF1 report's leftover chat_history was leaking into
     the next request for LONDON_KC1 — completely irrelevant station data
     getting injected into a different station's prompt.
"""

import asyncio
import json
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field

from src.agents.react_agent import AirSentinelAgent


class DailyReportSchema(BaseModel):
    executive_summary: str = Field(
        description="High-level overview of daily sensor trends"
    )
    regime_assessment: str = Field(
        description="Evaluation of pollution regimes and thresholds"
    )
    zone_level_analysis: str = Field(description="Detailed analysis per location_id")
    forecast_12h_analysis: str = Field(
        description="Insights from 12-hour ONNX forecasting model"
    )
    health_advisories: str = Field(
        description="Public safety recommendations based on WHO/EPA standards"
    )


class AutoReportGenerator:
    def __init__(self, output_dir: Path = Path("data/reports"), provider: str = "groq"):
        self.provider = provider
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(self, location_id: str = "LONDON_HF1") -> dict:
        logger.info(f"Synthesizing Daily Environmental Report for zone: {location_id}")
        query = (
            f"Generate a comprehensive daily environmental report for {location_id}. "
            f"First check recent anomalies with query_anomaly_db, get zone statistics with compute_zone_statistics, "
            f"fetch 12h forecasts with get_sensor_forecast, and search_env_documents for WHO guidelines. "
            f"Synthesize all findings into structured sections: Executive Summary, Regime Assessment, "
            f"Zone Analysis, 12h Forecast, and Health Advisories."
        )

        # Fresh agent per report -> fresh memory. Prevents one location_id's
        # conversation turn from leaking into the next location_id's prompt.
        agent = AirSentinelAgent(provider=self.provider)

        try:
            raw_response = agent.run(query)
            report_data = {
                "location_id": location_id,
                "status": "success",
                "report_body": raw_response,
            }
        except Exception as e:
            logger.error(f"Report generation failed for {location_id}: {e}")
            report_data = {
                "location_id": location_id,
                "status": "error",
                "report_body": None,
                "error": str(e),
            }

        out_file = self.output_dir / f"report_{location_id}.json"
        with open(out_file, "w") as f:
            json.dump(report_data, f, indent=2)

        if report_data["status"] == "success":
            logger.success(f"Report saved to {out_file}")
        else:
            logger.warning(f"Error report saved to {out_file} (generation failed)")

        return report_data


async def start_scheduled_reports(interval_hours: int = 24):
    """APScheduler Async runner for periodic daily generation."""
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    gen = AutoReportGenerator()
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        gen.generate_report, "interval", hours=interval_hours, args=["LONDON_HF1"]
    )

    logger.info(f"APScheduler active. Daily reports running every {interval_hours}h.")
    scheduler.start()

    try:
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()


if __name__ == "__main__":
    gen = AutoReportGenerator()
    gen.generate_report("LONDON_HF1")
