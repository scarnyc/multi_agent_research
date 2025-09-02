import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from agents.base import BaseAgent
from agents.models import Task, TaskResult, Status, AgentMessage, Priority
from config.settings import ModelType

class ConcreteAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing."""
    
    async def process_task(self, task: Task) -> str:
        return f"Processed task: {task.description}"
    
    async def _process_critical_message(self, message: AgentMessage) -> None:
        pass

@pytest.fixture
def agent():
    return ConcreteAgent(
        agent_id="test_agent",
        model_type=ModelType.GPT5_MINI,
        temperature=0.7,
        max_tokens=1000
    )

@pytest.fixture
def sample_task():
    return Task(
        id="task_123",
        description="Test task description",
        complexity="simple"
    )

@pytest.fixture
def sample_message():
    return AgentMessage(
        sender="sender_agent",
        recipient="test_agent",
        task_id="task_123",
        payload={"test": "data"},
        priority=Priority.MEDIUM
    )

class TestBaseAgent:
    
    def test_agent_initialization(self, agent):
        assert agent.agent_id == "test_agent"
        assert agent.model_type == ModelType.GPT5_MINI
        assert agent.temperature == 0.7
        assert agent.max_tokens == 1000
        assert len(agent.message_queue) == 0
        assert len(agent.task_history) == 0
    
    def test_get_model_name(self, agent):
        assert agent._model_name == "gpt-4o-mini"
        
        agent_regular = ConcreteAgent(
            agent_id="test",
            model_type=ModelType.GPT5_REGULAR
        )
        assert agent_regular._model_name == "gpt-4o"
    
    @pytest.mark.asyncio
    async def test_receive_message(self, agent, sample_message):
        await agent.receive_message(sample_message)
        assert len(agent.message_queue) == 1
        assert agent.message_queue[0] == sample_message
    
    @pytest.mark.asyncio
    async def test_receive_critical_message(self, agent):
        critical_message = AgentMessage(
            sender="sender",
            recipient="test_agent",
            task_id="critical_task",
            payload={"urgent": True},
            priority=Priority.CRITICAL
        )
        
        with patch.object(agent, '_process_critical_message', new_callable=AsyncMock) as mock_process:
            await agent.receive_message(critical_message)
            mock_process.assert_called_once_with(critical_message)
    
    @pytest.mark.asyncio
    async def test_send_message(self, agent):
        message = await agent.send_message(
            recipient="other_agent",
            task_id="task_456",
            payload={"data": "test"},
            priority=Priority.HIGH
        )
        
        assert message.sender == "test_agent"
        assert message.recipient == "other_agent"
        assert message.task_id == "task_456"
        assert message.payload == {"data": "test"}
        assert message.priority == Priority.HIGH
    
    @pytest.mark.asyncio
    async def test_execute_success(self, agent, sample_task):
        result = await agent.execute(sample_task)
        
        assert result.agent_id == "test_agent"
        assert result.task_id == "task_123"
        assert result.status == Status.COMPLETED
        assert result.result == "Processed task: Test task description"
        assert result.execution_time > 0
        assert result.model_used == "gpt-4o-mini"
        assert result.error is None
        
        assert len(agent.task_history) == 1
        assert agent.task_history[0] == result
    
    @pytest.mark.asyncio
    async def test_execute_failure(self, agent, sample_task):
        with patch.object(agent, 'process_task', side_effect=Exception("Test error")):
            result = await agent.execute(sample_task)
            
            assert result.status == Status.FAILED
            assert result.result is None
            assert result.error == "Test error"
            assert len(agent.task_history) == 1
    
    @pytest.mark.asyncio
    async def test_call_llm_retry(self, agent):
        with patch.object(agent.client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = [
                Exception("API Error"),
                Mock(choices=[Mock(message=Mock(content="Success"))])
            ]
            
            response = await agent._call_llm([{"role": "user", "content": "test"}])
            assert response.choices[0].message.content == "Success"
            assert mock_create.call_count == 2
    
    def test_get_stats_empty(self, agent):
        stats = agent.get_stats()
        
        assert stats["agent_id"] == "test_agent"
        assert stats["model_type"] == "gpt-4o-mini"
        assert stats["total_tasks"] == 0
        assert stats["completed_tasks"] == 0
        assert stats["failed_tasks"] == 0
        assert stats["success_rate"] == 0
        assert stats["avg_execution_time"] == 0
        assert stats["messages_in_queue"] == 0
    
    @pytest.mark.asyncio
    async def test_get_stats_with_history(self, agent):
        # Add some completed tasks
        for i in range(3):
            task_result = TaskResult(
                agent_id="test_agent",
                task_id=f"task_{i}",
                status=Status.COMPLETED,
                result="success",
                execution_time=2.0 + i,
                model_used="gpt-4o-mini"
            )
            agent.task_history.append(task_result)
        
        # Add a failed task
        failed_result = TaskResult(
            agent_id="test_agent",
            task_id="failed_task",
            status=Status.FAILED,
            result=None,
            execution_time=1.0,
            model_used="gpt-4o-mini",
            error="Test failure"
        )
        agent.task_history.append(failed_result)
        
        stats = agent.get_stats()
        
        assert stats["total_tasks"] == 4
        assert stats["completed_tasks"] == 3
        assert stats["failed_tasks"] == 1
        assert stats["success_rate"] == 0.75
        assert stats["avg_execution_time"] == 3.0  # (2+3+4)/3