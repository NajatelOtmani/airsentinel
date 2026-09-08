"""
src/agents/react_agent.py
=========================
DAY 13 — ReAct Agent Execution
------------------------------
Combines tools, memory, and LLM into a reasoning agent.
"""

import os

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from loguru import logger

from src.agents.tools import (compute_zone_statistics, get_sensor_forecast,
                              query_anomaly_db, search_env_documents)

REACT_PROMPT_TEMPLATE = """You are a Senior Environmental Data Scientist for AirSentinel.
Answer the question using the tools available below:

{tools}

Rules:
1. Do not execute the same tool twice with the exact same arguments.
2. If a tool returns "No records found" or an empty result, do NOT stop or retry the same tool. Simply state in your report that no data was found for that metric, and proceed to synthesize your Final Answer.
3. You have a strict budget of 4 tool calls total. After your 4th Observation, you MUST immediately write "Thought: I now know the final answer" and produce the Final Answer.

Use the following format:

Question: the input question you must answer
Thought: think step-by-step about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final response with clear citations, health advisories, and metrics analysis, written in plain prose — do not use bracketed citation markers such as 【Observation 1】or [Source]; instead attribute findings inline, e.g. "recent anomaly data shows..." or "per WHO guidelines..."

Previous conversation history:
{chat_history}

Question: {input}
Thought:{agent_scratchpad}"""


class AirSentinelAgent:
    def __init__(self, provider: str = "groq", model_name: str = None):
        self.provider = provider.lower()

        if self.provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI

            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            model_name = model_name or "gemini-1.5-flash"
            self.llm = ChatGoogleGenerativeAI(
                model=model_name, google_api_key=api_key, temperature=0.2
            )

        elif self.provider == "groq":
            from langchain_groq import ChatGroq

            # Hard-code the model parameter to avoid fallback resolution to retired models
            self.llm = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0.1,
                streaming=False,
                max_tokens=512,
            )

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        self.tools = [
            query_anomaly_db,
            get_sensor_forecast,
            search_env_documents,
            compute_zone_statistics,
        ]

        self.memory = ConversationBufferWindowMemory(
            k=1, memory_key="chat_history", return_messages=False
        )

        prompt = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE)
        agent = create_react_agent(self.llm, self.tools, prompt)
        self.executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=8,
            max_execution_time=45.0,
            early_stopping_method="generate",
        )

    def run(self, user_query: str) -> str:
        logger.info(f"Executing Agent Query: '{user_query}'")
        try:
            res = self.executor.invoke({"input": user_query})
            return res["output"]
        except Exception as e:
            err_str = str(e)
            logger.error(f"Raw Agent Exception: {err_str}")
            if (
                "413" in err_str
                or "rate_limit_exceeded" in err_str
                or "too large" in err_str.lower()
            ):
                return f"Groq limit exceeded. Raw detail: {err_str[:250]}"
            if "Parsing failed" in err_str or "could not be parsed" in err_str:
                return "The AI model produced an unparseable response. Please rephrase."
            raise


if __name__ == "__main__":
    print("Day 13 ReAct Agent ready.")
