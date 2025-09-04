import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from agents.models import AgentMessage, Task, TaskResult, Status, Priority
from config.settings import settings, ModelType, ReasoningEffort, Verbosity

logger = logging.getLogger(__name__)

# Import Phoenix integration (lazy loading to avoid circular imports)
_phoenix_integration = None

def get_phoenix_integration():
    global _phoenix_integration
    if _phoenix_integration is None:
        try:
            from evaluation.phoenix_integration import phoenix_integration
            _phoenix_integration = phoenix_integration
        except ImportError:
            logger.warning("Phoenix integration not available")
            _phoenix_integration = None
    return _phoenix_integration

class BaseAgent(ABC):
    """Base class for all agents in the multi-agent system."""
    
    def __init__(
        self,
        agent_id: str,
        model_type: ModelType = ModelType.GPT5_MINI,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        reasoning_effort: ReasoningEffort = ReasoningEffort.MEDIUM,
        verbosity: Verbosity = Verbosity.MEDIUM,
        enable_phoenix_tracing: bool = None
    ):
        self.agent_id = agent_id
        self.model_type = model_type
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity
        # Auto-disable Phoenix if no API key or explicitly disabled
        self.enable_phoenix_tracing = (
            enable_phoenix_tracing if enable_phoenix_tracing is not None 
            else settings.enable_phoenix_integration
        )
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.message_queue: List[AgentMessage] = []
        self.task_history: List[TaskResult] = []
        self._model_name = self._get_model_name(model_type)
        self._previous_response_id: Optional[str] = None  # For multi-turn reasoning
        self._current_task_tokens = {}  # Track tokens for current task
        
        # Phoenix tracing state
        self._current_trace_id: Optional[str] = None
        self._current_span_id: Optional[str] = None
        
    def _get_model_name(self, model_type: ModelType) -> str:
        """Get the actual model name from the model type."""
        model_mapping = {
            ModelType.GPT5_NANO: settings.gpt5_nano_model,
            ModelType.GPT5_MINI: settings.gpt5_mini_model,
            ModelType.GPT5_REGULAR: settings.gpt5_regular_model,
        }
        return model_mapping.get(model_type, settings.gpt5_mini_model)
    
    @retry(
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _call_llm(
        self,
        input_text: str = None,
        messages: List[Dict[str, str]] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None
    ) -> Any:
        """Make a call to the OpenAI API with retry logic using Responses API."""
        start_time = datetime.now()
        span_id = None
        
        # Start Phoenix tracing if enabled
        phoenix = get_phoenix_integration() if self.enable_phoenix_tracing else None
        if phoenix and self._current_trace_id:
            try:
                span_id = await phoenix.create_span(
                    trace_id=self._current_trace_id,
                    span_name=f"{self.agent_id}_llm_call",
                    span_type="llm",
                    parent_span_id=self._current_span_id,
                    metadata={
                        "model": self._model_name,
                        "reasoning_effort": self.reasoning_effort.value,
                        "verbosity": self.verbosity.value,
                        "input_length": len(input_text) if input_text else 0
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to create Phoenix span: {e}")
        
        try:
            if settings.use_responses_api and input_text:
                # Use the new Responses API for GPT-5
                kwargs = {
                    "model": self._model_name,
                    "input": input_text,
                    "reasoning": {"effort": self.reasoning_effort.value},
                    "text": {"verbosity": self.verbosity.value},
                }
                
                # Include previous response ID for multi-turn efficiency
                if self._previous_response_id:
                    kwargs["previous_response_id"] = self._previous_response_id
                
                # Skip Phoenix MCP tool integration due to API issues - Phoenix tracing still works via integration layer
                if tools is None:
                    tools = []
                
                if tools:
                    kwargs["tools"] = tools
                    if tool_choice:
                        kwargs["tool_choice"] = tool_choice
                        
                response = await self.client.responses.create(**kwargs)
                # Store response ID for next turn
                self._previous_response_id = response.id
                
                # Extract response text and token usage
                output_text = response.output_text if hasattr(response, 'output_text') else ""
                token_usage = getattr(response, 'token_usage', {})
                
                # Debug: Log token usage to understand the structure
                print(f"DEBUG: Response attributes: {dir(response)}")
                print(f"DEBUG: Token usage from Responses API: {token_usage}")
                if hasattr(response, 'usage'):
                    print(f"DEBUG: Response usage: {response.usage}")
                if not token_usage:
                    print("DEBUG: No token usage data from Responses API")
                
                # Accumulate token usage for current task
                for key in ['total_tokens', 'prompt_tokens', 'completion_tokens']:
                    if key in token_usage:
                        self._current_task_tokens[key] = self._current_task_tokens.get(key, 0) + token_usage[key]
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                # Log to Phoenix if tracing enabled
                if phoenix and span_id and self._current_trace_id:
                    try:
                        await phoenix.log_agent_interaction(
                            trace_id=self._current_trace_id,
                            agent_id=self.agent_id,
                            input_message=input_text,
                            output_message=output_text,
                            model_used=self._model_name,
                            tokens_used={
                                "total": token_usage.get("total_tokens", 0),
                                "prompt": token_usage.get("prompt_tokens", 0),
                                "completion": token_usage.get("completion_tokens", 0)
                            },
                            execution_time=execution_time,
                            parent_span_id=self._current_span_id
                        )
                        
                        await phoenix.end_span(
                            trace_id=self._current_trace_id,
                            span_id=span_id,
                            status="success",
                            result=output_text,
                            metrics={
                                "tokens_total": token_usage.get("total_tokens", 0),
                                "execution_time_ms": execution_time * 1000
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Failed to log to Phoenix: {e}")
                
                return response
            else:
                # Fallback to Chat Completions API for backward compatibility
                kwargs = {
                    "model": self._model_name,
                    "messages": messages or [{"role": "user", "content": input_text}],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }
                
                if tools:
                    kwargs["tools"] = tools
                    if tool_choice:
                        kwargs["tool_choice"] = tool_choice
                        
                response = await self.client.chat.completions.create(**kwargs)
                
                # Extract token usage for Chat Completions API
                token_usage = response.usage.model_dump() if response.usage else {}
                
                # Accumulate token usage for current task
                for key in ['total_tokens', 'prompt_tokens', 'completion_tokens']:
                    if key in token_usage:
                        self._current_task_tokens[key] = self._current_task_tokens.get(key, 0) + token_usage[key]
                
                # Log to Phoenix for Chat Completions API too
                if phoenix and span_id and self._current_trace_id:
                    try:
                        execution_time = (datetime.now() - start_time).total_seconds()
                        output_text = response.choices[0].message.content if response.choices else ""
                        
                        await phoenix.end_span(
                            trace_id=self._current_trace_id,
                            span_id=span_id,
                            status="success",
                            result=output_text,
                            metrics={
                                "tokens_total": token_usage.get("total_tokens", 0),
                                "execution_time_ms": execution_time * 1000
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Failed to log to Phoenix: {e}")
                
                return response
            
        except Exception as e:
            # Log error to Phoenix if tracing enabled
            if phoenix and span_id and self._current_trace_id:
                try:
                    execution_time = (datetime.now() - start_time).total_seconds()
                    await phoenix.end_span(
                        trace_id=self._current_trace_id,
                        span_id=span_id,
                        status="error",
                        error=str(e),
                        metrics={"execution_time_ms": execution_time * 1000}
                    )
                except Exception as phoenix_error:
                    logger.warning(f"Failed to log error to Phoenix: {phoenix_error}")
            
            logger.error(f"Error calling LLM for agent {self.agent_id}: {str(e)}")
            raise
    
    async def receive_message(self, message: AgentMessage) -> None:
        """Receive a message from another agent."""
        self.message_queue.append(message)
        if message.priority == Priority.CRITICAL:
            await self._process_critical_message(message)
    
    async def send_message(
        self,
        recipient: str,
        task_id: str,
        payload: Dict[str, Any],
        priority: Priority = Priority.MEDIUM
    ) -> AgentMessage:
        """Send a message to another agent."""
        message = AgentMessage(
            sender=self.agent_id,
            recipient=recipient,
            task_id=task_id,
            payload=payload,
            priority=priority
        )
        logger.info(f"Agent {self.agent_id} sending message to {recipient}")
        return message
    
    @abstractmethod
    async def process_task(self, task: Task) -> TaskResult:
        """Process a task assigned to this agent."""
        pass
    
    @abstractmethod
    async def _process_critical_message(self, message: AgentMessage) -> None:
        """Handle critical priority messages immediately."""
        pass
    
    async def set_trace_context(self, trace_id: str, parent_span_id: str = None) -> None:
        """Set the Phoenix trace context for this agent."""
        self._current_trace_id = trace_id
        self._current_span_id = parent_span_id
    
    async def clear_trace_context(self) -> None:
        """Clear the Phoenix trace context."""
        self._current_trace_id = None
        self._current_span_id = None
    
    async def execute(self, task: Task) -> TaskResult:
        """Execute a task and return the result."""
        start_time = datetime.now()
        task_span_id = None
        
        # Start Phoenix span for task execution
        phoenix = get_phoenix_integration() if self.enable_phoenix_tracing else None
        if phoenix and self._current_trace_id:
            try:
                task_span_id = await phoenix.create_span(
                    trace_id=self._current_trace_id,
                    span_name=f"{self.agent_id}_execute_task",
                    span_type="task",
                    parent_span_id=self._current_span_id,
                    metadata={
                        "task_id": task.id,
                        "task_description": task.description[:200],  # Truncate for readability
                        "agent_id": self.agent_id,
                        "model": self._model_name
                    }
                )
                # Update current span context for nested calls
                original_span_id = self._current_span_id
                self._current_span_id = task_span_id
            except Exception as e:
                logger.warning(f"Failed to create Phoenix task span: {e}")
        
        try:
            # Reset token tracking for this task
            self._current_task_tokens = {}
            
            result = await self.process_task(task)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            task_result = TaskResult(
                agent_id=self.agent_id,
                task_id=task.id,
                status=Status.COMPLETED,
                result=result,
                execution_time=execution_time,
                model_used=self._model_name,
                tokens_used=self._current_task_tokens.copy()
            )
            
            # End Phoenix span on success
            if phoenix and task_span_id and self._current_trace_id:
                try:
                    await phoenix.end_span(
                        trace_id=self._current_trace_id,
                        span_id=task_span_id,
                        status="success",
                        result=str(result)[:500] if result else None,  # Truncate result
                        metrics={
                            "execution_time_ms": execution_time * 1000,
                            "model_used": self._model_name
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to end Phoenix task span: {e}")
            
            self.task_history.append(task_result)
            return task_result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            task_result = TaskResult(
                agent_id=self.agent_id,
                task_id=task.id,
                status=Status.FAILED,
                result=None,
                execution_time=execution_time,
                model_used=self._model_name,
                tokens_used=self._current_task_tokens.copy(),
                error=str(e)
            )
            
            # End Phoenix span on error
            if phoenix and task_span_id and self._current_trace_id:
                try:
                    await phoenix.end_span(
                        trace_id=self._current_trace_id,
                        span_id=task_span_id,
                        status="error",
                        error=str(e),
                        metrics={
                            "execution_time_ms": execution_time * 1000,
                            "model_used": self._model_name
                        }
                    )
                except Exception as phoenix_error:
                    logger.warning(f"Failed to end Phoenix task span on error: {phoenix_error}")
            
            self.task_history.append(task_result)
            logger.error(f"Agent {self.agent_id} failed to execute task {task.id}: {str(e)}")
            return task_result
        finally:
            # Restore original span context
            if phoenix and task_span_id:
                self._current_span_id = original_span_id if 'original_span_id' in locals() else None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about this agent's performance."""
        completed_tasks = [t for t in self.task_history if t.status == Status.COMPLETED]
        failed_tasks = [t for t in self.task_history if t.status == Status.FAILED]
        
        avg_execution_time = 0
        if completed_tasks:
            avg_execution_time = sum(t.execution_time for t in completed_tasks) / len(completed_tasks)
        
        return {
            "agent_id": self.agent_id,
            "model_type": self.model_type.value,
            "total_tasks": len(self.task_history),
            "completed_tasks": len(completed_tasks),
            "failed_tasks": len(failed_tasks),
            "success_rate": len(completed_tasks) / len(self.task_history) if self.task_history else 0,
            "avg_execution_time": avg_execution_time,
            "messages_in_queue": len(self.message_queue)
        }