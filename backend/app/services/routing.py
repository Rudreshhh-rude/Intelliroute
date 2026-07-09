import time, logging
from typing import Dict, Any, List, Tuple, Optional
from pydantic import BaseModel
from backend.app.core.config import settings
from backend.app.services.retrieval import get_genai_client, _parse_json_response, _call_gemini_with_retry

logger = logging.getLogger("intelliroute.routing")

# Estimated Gemini pricing per token (USD)
MODEL_PRICING = {
    "gemini-3.5-flash": {
        "input": 0.075 / 1_000_000,
        "output": 0.30 / 1_000_000
    },
    "gemini-2.5-flash": {
        "input": 0.075 / 1_000_000,
        "output": 0.30 / 1_000_000
    },
    "gemini-2.0-flash": {
        "input": 0.075 / 1_000_000,
        "output": 0.30 / 1_000_000
    },
    "gemini-2.5-pro": {
        "input": 1.25 / 1_000_000,
        "output": 5.00 / 1_000_000
    },
    "gemini-1.5-pro": {
        "input": 1.25 / 1_000_000,
        "output": 5.00 / 1_000_000
    }
}

DEFAULT_PRICING_FLASH = {"input": 0.075 / 1_000_000, "output": 0.30 / 1_000_000}
DEFAULT_PRICING_PRO = {"input": 1.25 / 1_000_000, "output": 5.00 / 1_000_000}

class RouteMetrics(BaseModel):
    query: str
    classified_complexity: str 
    chosen_model: str
    reasoning: str
    prompt_tokens: int
    completion_tokens: int
    latency_sec: float
    cost_usd: float
    timestamp: float

from backend.app.services import database

class ObservabilityRegistry:
    def __init__(self):
        pass

    def log_transaction(self, metrics: RouteMetrics):
        database.log_route_metrics(
            query=metrics.query,
            classified_complexity=metrics.classified_complexity,
            chosen_model=metrics.chosen_model,
            reasoning=metrics.reasoning,
            prompt_tokens=metrics.prompt_tokens,
            completion_tokens=metrics.completion_tokens,
            latency_sec=metrics.latency_sec,
            cost_usd=metrics.cost_usd
        )
        logger.info(
            f"observability_log: Query='{metrics.query[:30]}...' RoutedTo={metrics.chosen_model} "
            f"Complexity={metrics.classified_complexity} Latency={metrics.latency_sec:.2f}s "
            f"Cost=${metrics.cost_usd:.6f} Tokens={metrics.prompt_tokens + metrics.completion_tokens}"
        )

    def get_summary(self) -> Dict[str, Any]:
        return database.get_metrics_summary()
observability_registry = ObservabilityRegistry()


class ModelRouter:
    def __init__(self, api_key: str):
        self.client = get_genai_client(api_key)
        self.model_flash = settings.model_name_flash
        self.model_pro = settings.model_name_pro or "gemini-2.5-pro"

    def classify_query(self, query: str) -> Tuple[str, str]:
        """Classifies incoming query as 'simple' or 'complex' using the fast Flash model."""
        prompt = f"""You are an advanced query complexity classifier for an enterprise router.
Analyze the user's query and classify it as either "simple" or "complex".

User Query: "{query}"

Classification Rules:
- "simple": Factual questions, simple definitions, short summaries of a single point, greetings, single-topic retrieval (e.g. "What is the refill rate of the rate limiter?", "Who is the admin?").
- "complex": Ambiguous questions, requests for synthesis or comparison of multiple sections, multi-step code generation, logical reasoning, troubleshooting, or open-ended analytical prompts (e.g., "Compare role-based access control with standard writer roles and write a python script to implement it").

Format output as a raw JSON object with two fields:
- "complexity": "simple" or "complex"
- "reasoning": "A brief explanation of why this classification was made."

Format output as raw JSON only.
"""
        try:
            response = _call_gemini_with_retry(
                client=self.client,
                model=self.model_flash,
                contents=prompt
            )
            data = _parse_json_response(response.text)
            complexity = data.get("complexity", "simple").lower()
            reasoning = data.get("reasoning", "Default routing classification.")
            if complexity not in ["simple", "complex"]:
                complexity = "simple"
            return complexity, reasoning
        except Exception as e:
            logger.error(f"Error in query classification: {e}. Defaulting to 'simple'")
            return "simple", f"Error during classification: {e}. Defaulted to simple routing."

    def execute_and_metrics(
        self,
        query: str,
        system_instruction: Optional[str] = None,
        contents: Any = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, RouteMetrics]:
        """Classifies query, routes to appropriate model, runs inference, and logs performance metrics."""
        # 1. Classify complexity
        complexity, reasoning = self.classify_query(query)
        
        # 2. Select model
        chosen_model = self.model_flash if complexity == "simple" else self.model_pro
        logger.info(f"Model Router chose model {chosen_model} based on '{complexity}' complexity classification.")
        
        # 3. Assemble parameters
        config = config or {}
        if system_instruction and "system_instruction" not in config:
            config["system_instruction"] = system_instruction
            
        # Standard input is just the query if contents not provided
        inputs = contents if contents is not None else query
        
        # 4. Infer and measure latency
        start_time = time.time()
        try:
            response = _call_gemini_with_retry(
                client=self.client,
                model=chosen_model,
                contents=inputs,
                config=config
            )
            latency_sec = time.time() - start_time
        except Exception as e:
            logger.error(f"Execution failed on model {chosen_model}: {e}")
            raise e

        # 5. Extract token counts and calculate cost
        prompt_tokens = 0
        completion_tokens = 0
        if response.usage_metadata:
            prompt_tokens = response.usage_metadata.prompt_token_count or 0
            completion_tokens = response.usage_metadata.candidates_token_count or 0
        if prompt_tokens == 0:
            prompt_tokens = len(str(inputs)) // 4
        if completion_tokens == 0:
            completion_tokens = len(response.text or "") // 4
            
        pricing = MODEL_PRICING.get(chosen_model)
        if not pricing:
            pricing = DEFAULT_PRICING_PRO if "pro" in chosen_model else DEFAULT_PRICING_FLASH
            
        cost_usd = (prompt_tokens * pricing["input"]) + (completion_tokens * pricing["output"])
        metrics = RouteMetrics(
            query=query,
            classified_complexity=complexity,
            chosen_model=chosen_model,
            reasoning=reasoning,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_sec=latency_sec,
            cost_usd=cost_usd,
            timestamp=time.time()
        )
        observability_registry.log_transaction(metrics)
        
        return response.text or "", metrics
