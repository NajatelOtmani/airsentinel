"""
run_rag_pipeline.py
===================
Master orchestration runner for Days 11-15.
Usage:
    python run_rag_pipeline.py --step day11
    python run_rag_pipeline.py --step day12
    python run_rag_pipeline.py --step day13
    python run_rag_pipeline.py --step day14
    python run_rag_pipeline.py --step day15
    python run_rag_pipeline.py --step all
"""

import argparse

from dotenv import load_dotenv
from loguru import logger

load_dotenv()


def run_step(step: str):
    step = step.lower()

    if step in ["day11", "all"]:
        logger.info("=== Executing Day 11: FAISS Vector Ingestion ===")
        from src.rag.ingest import AirSentinelIngestor

        ingestor = AirSentinelIngestor()
        ingestor.run(force_reingest=False)
        ingestor.run_test_queries()

    if step in ["day12", "all"]:
        logger.info(
            "=== Executing Day 12: Resilient LLM Client & Adversarial Tests ==="
        )
        from src.rag.llm_client import ResilientLLMClient

        client = ResilientLLMClient(provider="google")
        client.run_adversarial_suite()

    if step in ["day13", "all"]:
        logger.info("=== Executing Day 13: ReAct Agent Query ===")
        from src.agents.react_agent import AirSentinelAgent

        agent = AirSentinelAgent(provider="groq")
        res = agent.run(
            "Show recent anomalies for LONDON_HF1 and give health recommendations based on WHO guidelines."
        )
        logger.info(f"Agent Final Output:\n{res}")

    if step in ["day14", "all"]:
        logger.info("=== Executing Day 14: Automated Environmental Report ===")
        from src.agents.report_generator import AutoReportGenerator

        gen = AutoReportGenerator(provider="groq")
        gen.generate_report(location_id="LONDON_HF1")

    if step in ["day15", "all"]:
        logger.info("=== Executing Day 15: RAGAS Evaluation ===")
        from src.rag.eval_ragas import AirSentinelRAGASEvaluator

        evaluator = AirSentinelRAGASEvaluator()
        evaluator.run_evaluation()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AirSentinel Days 11-15 Execution Pipeline"
    )
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        help="Step to run: day11, day12, day13, day14, day15, or all",
    )
    args = parser.parse_args()

    run_step(args.step)
