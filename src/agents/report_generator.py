"""
src/agents/report_generator.py
==============================
DAY 14 — Auto-Report Generator
------------------------------
Multi-step synthesis: Anomalies -> RAG Context -> 12h Forecast -> Report.
Includes APScheduler background script for daily report scheduling.
"""

import asyncio
import json
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field

from src.agents.react_agent import AirSentinelAgent
from src.agents.tools import (
    compute_zone_statistics,
    get_sensor_forecast,
    query_anomaly_db,
)


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
        
        # 1. Collect context deterministically in Python
        report_context = "\n\n".join(
            [
                f"Location: {location_id}",
                query_anomaly_db.invoke({"location_id": location_id, "hours": 3}),
                compute_zone_statistics.invoke({"location_id": location_id}),
                get_sensor_forecast.invoke({"location_id": location_id}),
            ]
        )

        raw_response = None
        last_exception = None

        # 2. Attempt synthesis with primary provider (Groq) up to 2 times
        for attempt in range(1, 3):
            try:
                agent = AirSentinelAgent(provider=self.provider)
                response = agent.synthesize_report(report_context)
                if response and response.strip():
                    raw_response = response
                    break
                else:
                    logger.warning(
                        f"Attempt {attempt} for {location_id} returned empty content. Retrying..."
                    )
            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt} for {location_id} failed: {e}")

        # 3. Fallback to Gemini if Groq fails or returns empty output
        if not raw_response:
            logger.warning(
                f"Primary provider '{self.provider}' failed for {location_id}. Triggering Google Gemini fallback..."
            )
            try:
                fallback_agent = AirSentinelAgent(provider="google")
                fallback_response = fallback_agent.synthesize_report(report_context)
                if fallback_response and fallback_response.strip():
                    raw_response = fallback_response
            except Exception as e:
                logger.error(f"Fallback synthesis for {location_id} failed: {e}")
                last_exception = e

        # 4. Construct response payload
        if raw_response:
            report_data = {
                "location_id": location_id,
                "status": "success",
                "report_body": raw_response,
            }
        else:
            report_data = {
                "location_id": location_id,
                "status": "error",
                "report_body": None,
                "error": str(last_exception) if last_exception else "No content returned from providers",
            }

        # 5. Save report output
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