"""
src/agents/react_agent.py
=========================
DAY 13 — Agent Execution (Tool-Calling Architecture)
------------------------------
Uses native tool-calling instead of text-pattern ReAct, since gpt-oss-20b
(and similar function-calling-native models) don't reliably follow the
Thought/Action/Action Input text format — they want to emit real tool
calls. create_tool_calling_agent lets the model do that directly instead
of being forced into brittle text parsing.
"""

import time

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from loguru import logger

from src.agents.tools import (
    compute_zone_statistics,
    get_sensor_forecast,
    get_network_summary,
    query_anomaly_db,
    search_env_documents,
)
from src.config import settings

SYSTEM_PROMPT = """You are a Senior Environmental Data Scientist for AirSentinel.

Rules:
1. Do not call the same tool twice with the exact same arguments.
2. If a tool returns "No records found" or an empty result, do not retry it — state in your answer that no data was found for that metric and move on.
3. Use at most 3 tool calls total before giving your final answer.
4. Write your final answer in plain prose with specific metrics and concise health guidance. Do not use bracketed citation markers like [Source] or 【Observation 1】 — attribute findings inline instead, e.g. "recent anomaly data shows..." or "per WHO guidelines..."."""


class AirSentinelAgent:
    def __init__(self, provider: str = "groq"):
        self.provider = provider.lower()

        if self.provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI

            api_key = settings.GEMINI_API_KEY or settings.GOOGLE_API_KEY
            if not api_key:
                raise ValueError("Google provider requires GEMINI_API_KEY or GOOGLE_API_KEY.")

            self.llm = ChatGoogleGenerativeAI(
                model=settings.GEMINI_MODEL,
                google_api_key=api_key,
                temperature=0.1,
                max_output_tokens=2048,
            )
        elif self.provider == "groq":
            from langchain_groq import ChatGroq

            if not settings.GROQ_API_KEY:
                raise ValueError("Groq provider requires GROQ_API_KEY (set it in .env).")

            self.llm = ChatGroq(
                model=settings.GROQ_MODEL,
                api_key=settings.GROQ_API_KEY,
                temperature=0.1,
                streaming=False,
                max_tokens=1500,
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        self.tools = [
            query_anomaly_db,
            get_sensor_forecast,
            search_env_documents,
            compute_zone_statistics,
            get_network_summary,
        ]

        # Tool-calling agents need real message objects in chat_history,
        # not a flattened string — hence return_messages=True here.
        self.memory = ConversationBufferWindowMemory(
            k=1, memory_key="chat_history", return_messages=True
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(self.llm, self.tools, prompt)

        self.executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=6,
            max_execution_time=40.0,
        )

    def synthesize_report(self, report_context: str) -> str:
        """Generate a report without exposing tools to the model.
        Daily reports collect their data sources in Python first, so the
        LLM only synthesizes — it never enters a tool-calling loop here.
        """
        prompt = f"""You are a Senior Environmental Data Scientist for AirSentinel. Write a comprehensive daily environmental report from the verified data below.

{report_context}

Use these exact sections: Executive Summary, Regime Assessment, Zone Analysis, 12h Forecast, and Health Advisories. State when data is unavailable. Use plain prose with specific metrics and concise health guidance. Do not call tools, do not describe your reasoning."""

        has_google_key = bool(settings.GEMINI_API_KEY or settings.GOOGLE_API_KEY)

        for attempt in range(1, 3):
            try:
                response = self.llm.invoke(prompt)
                content = response.content if hasattr(response, "content") else str(response)
                if content and str(content).strip():
                    return str(content).strip()
            except Exception as exc:
                logger.warning(f"Report synthesis attempt {attempt}/2 failed on provider '{self.provider}': {exc}")
                if attempt < 2:
                    time.sleep(1)

        if self.provider == "groq" and has_google_key:
            logger.warning("Groq synthesis retries exhausted; switching to Google Gemini provider.")
            try:
                return AirSentinelAgent(provider="google").synthesize_report(report_context)
            except Exception as fallback_err:
                logger.error(f"Google Gemini fallback synthesis failed: {fallback_err}")

        return "Failed to synthesize environmental report due to upstream provider errors."

    def run(self, user_query: str) -> str:
        logger.info(f"Executing Agent Query: '{user_query}'")
        try:
            res = self.executor.invoke({"input": user_query})
            return res.get("output", "No response generated by agent.")
        except Exception as e:
            err_str = str(e)
            logger.error(f"Raw Agent Exception: {err_str}")

            groq_retry_errors = ("tool_use_failed", "tool choice is none", "parsing", "rate_limit_exceeded", "429")
            has_google_key = bool(settings.GEMINI_API_KEY or settings.GOOGLE_API_KEY)

            if self.provider == "groq" and has_google_key and any(m in err_str.lower() for m in groq_retry_errors):
                logger.warning("Groq execution failed; retrying query with Google Gemini provider.")
                try:
                    return AirSentinelAgent(provider="google").run(user_query)
                except Exception as fallback_error:
                    logger.error(f"Google fallback execution failed: {fallback_error}")

            if "413" in err_str or "rate_limit_exceeded" in err_str or "429" in err_str:
                return f"Rate limit exceeded on provider '{self.provider}'. Please wait a moment before retrying."

            if "Parsing failed" in err_str or "could not be parsed" in err_str:
                return "The agent encountered a parsing error while evaluating tools. Please retry your request."

            if "model_not_found" in err_str or "does not exist" in err_str:
                return f"Model on provider '{self.provider}' is unavailable/deprecated. Details: {err_str[:200]}"

            return f"Agent execution error on provider '{self.provider}': {err_str[:250]}"


if __name__ == "__main__":
    print("AirSentinel Agent ready.")