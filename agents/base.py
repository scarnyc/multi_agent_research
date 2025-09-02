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

class BaseAgent(ABC):
    """Base class for all agents in the multi-agent system."""
    
    def __init__(
        self,
        agent_id: str,
        model_type: ModelType = ModelType.GPT5_MINI,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        reasoning_effort: ReasoningEffort = ReasoningEffort.MEDIUM,
        verbosity: Verbosity = Verbosity.MEDIUM
    ):
        self.agent_id = agent_id
        self.model_type = model_type
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.message_queue: List[AgentMessage] = []
        self.task_history: List[TaskResult] = []
        self._model_name = self._get_model_name(model_type)
        self._previous_response_id: Optional[str] = None  # For multi-turn reasoning
        
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
                
                if tools:
                    kwargs["tools"] = tools
                    if tool_choice:
                        kwargs["tool_choice"] = tool_choice
                        
                response = await self.client.responses.create(**kwargs)
                # Store response ID for next turn
                self._previous_response_id = response.id
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
                return response
            
        except Exception as e:
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
    
    async def execute(self, task: Task) -> TaskResult:
        """Execute a task and return the result."""
        start_time = datetime.now()
        
        try:
            result = await self.process_task(task)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            task_result = TaskResult(
                agent_id=self.agent_id,
                task_id=task.id,
                status=Status.COMPLETED,
                result=result,
                execution_time=execution_time,
                model_used=self._model_name
            )
            
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
                error=str(e)
            )
            
            self.task_history.append(task_result)
            logger.error(f"Agent {self.agent_id} failed to execute task {task.id}: {str(e)}")
            return task_result
    
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