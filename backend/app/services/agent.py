import time, json, logging
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel
from google import genai
from google.genai import types
from backend.app.core.config import settings
from backend.app.services.routing import ModelRouter, RouteMetrics
from backend.app.services.retrieval import get_genai_client, _call_gemini_with_retry
from backend.app.services import database

logger = logging.getLogger("intelliroute.agent")

# Tool definitions

def create_support_ticket(title: str, description: str, priority: str = "medium") -> str:
    """Creates a customer support ticket in the helpdesk system.
    
    Args:
        title: The title of the ticket.
        description: Detailed explanation of the support request.
        priority: Priority level, e.g. low, medium, high.
    """
    ticket_id = f"TICKET-{int(time.time())}"
    ticket = {
        "ticket_id": ticket_id,
        "title": title,
        "description": description,
        "priority": priority,
        "status": "open",
        "created_at": time.time()
    }
    logger.info(f"TOOL_CALL_SUCCESS: Support ticket created: {ticket}")
    return json.dumps({
        "status": "success",
        "message": "Support ticket created successfully.",
        "ticket": ticket
    })

def export_summary_json(summary: str, filename: str) -> str:
    """Exports a text summary into a clean structured JSON file.
    
    Args:
        summary: The text summary or bullets to export.
        filename: The name of the file (should end in .json).
    """
    if not filename.endswith(".json"):
        filename += ".json"
        
    data = {
        "summary_content": summary,
        "exported_at": time.time(),
        "status": "success"
    }
    
    # Write mock summary file under a test output directory
    current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else "."
    output_dir = os.path.abspath(os.path.join(current_dir, "../../../test_outputs"))
    os.makedirs(output_dir, exist_ok=True)
    
    file_path = os.path.join(output_dir, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        
    logger.info(f"TOOL_CALL_SUCCESS: Summary exported to {file_path}")
    return json.dumps({
        "status": "success",
        "message": f"Summary exported successfully to file: {filename}",
        "file_path": file_path,
        "exported_data": data
    })

# Make import os safe
import os

# Registry mapping for dispatching tool calls dynamically
TOOLS_MAP = {
    "create_support_ticket": create_support_ticket,
    "export_summary_json": export_summary_json
}

# Adaptive prompt templates

ZERO_SHOT_SYSTEM = """You are a helpful, factual Enterprise Knowledge Assistant.
Answer the user's query directly and concisely using the retrieved context.
Context:
{context}

If the context does not contain the answer, say "I cannot find the answer in the provided documents." and offer to create a support ticket using the create_support_ticket tool.
"""

COT_SYSTEM = """You are an advanced Enterprise Knowledge Assistant.
To answer the user's query, you must show your logical reasoning step-by-step.
Context:
{context}

Follow these rules:
1. Explain your logical reasoning in a "Reasoning Process:" block.
2. Formulate your final answer based on the reasoning in the context.
3. Be precise, thorough, and point out any gaps or ambiguities in the documents.
If you cannot answer the query using the context, suggest the user create a support ticket using the create_support_ticket tool.

Example 1:
Query: "Why does the rate limiter block requests?"
Reasoning Process:
- The rate limiter is configured as a token bucket.
- The burst limit is 60 tokens, and the refill rate is 1 token/sec.
- If a client sends requests exceeding the refill rate and exhausts the 60-token burst, the bucket goes to 0 and blocks requests.
Final Answer: The rate limiter blocks requests because the token bucket has been exhausted. It allows up to 60 burst requests and refills at 1 request per second.
"""

# Memory management

class MemoryMessage(BaseModel):
    role: str  # "user" or "model"
    content: str

class ConversationMemory:
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id

    @property
    def messages(self) -> List[MemoryMessage]:
        rows = database.get_chat_messages(self.session_id)
        return [MemoryMessage(role=row["role"], content=row["content"]) for row in rows]

    @property
    def running_summary(self) -> str:
        return database.get_running_summary(self.session_id)

    @running_summary.setter
    def running_summary(self, val: str):
        database.update_running_summary(self.session_id, val)

    def add_message(self, role: str, content: str):
        database.save_chat_message(self.session_id, role, content)

    def get_context_summary(self) -> str:
        summary = self.running_summary
        if summary:
            return f"Previous conversation summary: {summary}\n"
        return ""

    def trim_and_summarize(self, api_key: str, max_messages: int = 8):
        """Trims old messages if context gets too long, and generates a running summary."""
        messages = self.messages
        if len(messages) <= max_messages:
            return

        # Take the oldest messages to trim (all except the last 4)
        trim_count = len(messages) - 4
        to_trim = messages[:trim_count]
        
        # Trim from db
        database.trim_chat_messages(self.session_id, keep_last=4)

        # Formulate a summary of the trimmed portion
        conversation_text = ""
        for msg in to_trim:
            conversation_text += f"{msg.role.upper()}: {msg.content}\n"

        prompt = f"""Summarize the following chat conversation history into a single concise paragraph.
Preserve key details discussed, queries made, and solutions provided.

Existing Summary: {self.running_summary or "None"}

New Chat Segment to Summarize:
{conversation_text}

Generate only the summary paragraph.
"""
        try:
            client = get_genai_client(api_key)
            response = _call_gemini_with_retry(
                client=client,
                model=settings.model_name_flash,
                contents=prompt
            )
            new_summary = response.text.strip()
            self.running_summary = new_summary
            logger.info(f"Memory trimmed. New running summary: '{new_summary}'")
        except Exception as e:
            logger.error(f"Failed to generate conversation summary: {e}")
            pass


class AgentEngine:
    def __init__(self, api_key: str):
        self.router = ModelRouter(api_key=api_key)
        self.client = get_genai_client(api_key)
        self.api_key = api_key

    def run_agent_loop(
        self,
        query: str,
        context: str,
        memory: ConversationMemory
    ) -> Tuple[str, RouteMetrics, Optional[Dict[str, Any]]]:
        """Runs the agent loop. 
        
        Classifies complexity, chooses zero-shot or CoT prompts,
        handles tool definitions, and returns the response and execution metrics.
        """
        # 1. Query Router determines complexity first
        complexity, reasoning = self.router.classify_query(query)
        chosen_model = self.router.model_pro if complexity == "complex" else self.router.model_flash
        
        # 2. Select dynamic prompt system instruction
        system_instruction = (
            COT_SYSTEM.format(context=context) if complexity == "complex"
            else ZERO_SHOT_SYSTEM.format(context=context)
        )
        
        # 3. Inject running memory summary if any
        if memory.get_context_summary():
            system_instruction = memory.get_context_summary() + "\n" + system_instruction

        # 4. Formulate content structure for Gemini SDK (includes memory history)
        history_contents = []
        for msg in memory.messages:
            history_contents.append(
                types.Content(
                    role="user" if msg.role == "user" else "model",
                    parts=[types.Part.from_text(text=msg.content)]
                )
            )
        # Add current query
        history_contents.append(
            types.Content(role="user", parts=[types.Part.from_text(text=query)])
        )

        # 5. Define tools list for function calling
        tools_list = [create_support_ticket, export_summary_json]
        
        # Create final configuration
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=tools_list,
            temperature=0.2
        )

        # 6. Execute inference and track latency
        start_time = time.time()
        tool_call_details = None
        
        try:
            response = _call_gemini_with_retry(
                client=self.client,
                model=chosen_model,
                contents=history_contents,
                config=config
            )
            
            # Check if Gemini requested a function call
            if response.function_calls:
                # Execute tool call
                func_call = response.function_calls[0]
                tool_name = func_call.name
                tool_args = func_call.args
                
                logger.info(f"Gemini requested tool call: {tool_name} with args {tool_args}")
                
                if tool_name in TOOLS_MAP:
                    tool_func = TOOLS_MAP[tool_name]
                    tool_result_str = tool_func(**tool_args)
                    tool_call_details = {
                        "tool_name": tool_name,
                        "arguments": tool_args,
                        "result": json.loads(tool_result_str)
                    }
                    
                    # Follow-up: Pass tool results back to Gemini to generate the final response
                    # Add model's request to content
                    history_contents.append(response.candidates[0].content)
                    
                    # Add function response part
                    function_response_part = types.Part.from_function_response(
                        name=tool_name,
                        response={"result": tool_result_str}
                    )
                    history_contents.append(
                        types.Content(role="tool", parts=[function_response_part])
                    )
                    
                    # Call Gemini again to compile the final answer
                    response = _call_gemini_with_retry(
                        client=self.client,
                        model=chosen_model,
                        contents=history_contents,
                        config=config
                    )
                else:
                    response_text = f"Error: Tool '{tool_name}' not found."
                    
            response_text = response.text or ""
            latency_sec = time.time() - start_time
            
        except Exception as e:
            logger.error(f"Inference error in agent loop: {e}")
            raise e

        # 7. Collect usage and cost metrics
        prompt_tokens = 0
        completion_tokens = 0
        if response.usage_metadata:
            prompt_tokens = response.usage_metadata.prompt_token_count or 0
            completion_tokens = response.usage_metadata.candidates_token_count or 0
            
        if prompt_tokens == 0:
            prompt_tokens = len(str(query) + str(context)) // 4
        if completion_tokens == 0:
            completion_tokens = len(response_text) // 4
            
        from backend.app.services.routing import MODEL_PRICING, DEFAULT_PRICING_FLASH, DEFAULT_PRICING_PRO
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
        
        # Log to shared observability registry
        from backend.app.services.routing import observability_registry
        observability_registry.log_transaction(metrics)
        
        return response_text, metrics, tool_call_details
