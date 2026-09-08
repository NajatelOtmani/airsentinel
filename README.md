AirSentinel — Environmental Anomaly & Reporting System
AirSentinel is an end-to-end environmental data monitoring platform. It collects sensor metrics, detects anomalies using autoencoders, predicts short-term trends with ONNX-exported Transformer forecasters, and automatically generates analytical reports using an AI agent.

🛠️ System Architecture
Data Ingestion: Kafka pipeline capturing real-time air quality metrics across sensor networks (e.g., LONDON_GR9, LONDON_HIL).

Anomaly Detection & Forecasting: PyTorch Autoencoders and Transformer models running via ONNX runtime for sub-12h predictions.

Backend API: FastAPI application executing background tasks and managing sensor data storage.

AI Report Generation: LangChain-based ReAct agent using the Groq API (ChatGroq) to query live anomaly databases, WHO guidelines, and forecasting metrics.

Dashboard: Streamlit user interface for live metric visualization and report triggers.

⚠️ Known Issue & Investigation Status
Current Status: AI Model Selection Error (404 Model Not Found)
During automated report generation via the Streamlit dashboard, certain sensor requests (LONDON_GR9, LONDON_HIL) fail during the ReAct reasoning step with the following error:

Report generation failed: Error code: 404 - 
{'error': {'message': 'The model llama-3.3-70b-versatile does not exist or you do not have access to it.', 
 'type': 'invalid_request_error', 'code': 'model_not_found'}}
Root Cause Analysis
Model Retirement: The llama-3.3-70b-versatile endpoint previously specified in the agent default configuration has been deprecated/retired on the Groq provider API.

Fallback Resolution: When the API layer passes model_name=None or empty arguments, internal library fallbacks in langchain_groq default back to the deprecated model string.

Container Context: The Dockerized FastAPI backend retains cached environment configurations that override local parameter updates.

🔧 Planned Fix & Work in Progress
Agent Model Override: Refactoring src/agents/react_agent.py to strictly enforce the active llama-3.1-8b-instant model across all agent initialization paths.

Provider Alignment: Setting provider="groq" as the global default across AutoReportGenerator (src/agents/report_generator.py) and API routers (src/api/routers/reports.py).

Container Environment Update: Rebuilding the production Docker API container (docker-compose.prod.yml) with --no-cache to propagate the updated parameters.