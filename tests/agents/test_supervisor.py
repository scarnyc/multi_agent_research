import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from uuid import uuid4

from agents.supervisor import SupervisorAgent
from agents.base import BaseAgent
from agents.models import Task, TaskResult, Status, Priority, AgentMessage
from config.settings import ModelType, ComplexityLevel

class MockSubAgent(BaseAgent):
    """Mock sub-agent for testing."""
    
    async def process_task(self, task: Task) -> str:
        return f"Mock result for {task.description}"
    
    async def _process_critical_message(self, message) -> None:
        pass

@pytest.fixture
def supervisor():
    return SupervisorAgent()

@pytest.fixture
def mock_search_agent():
    return MockSubAgent(
        agent_id="search",
        model_type=ModelType.GPT5_MINI
    )

@pytest.fixture
def mock_citation_agent():
    return MockSubAgent(
        agent_id="citation",
        model_type=ModelType.GPT5_NANO
    )

@pytest.fixture
def sample_query():
    return "What are the latest developments in quantum computing?"

class TestSupervisorAgent:
    
    def test_supervisor_initialization(self, supervisor):
        assert supervisor.agent_id == "supervisor"
        assert supervisor.model_type == ModelType.GPT5_REGULAR
        assert supervisor.temperature == 0.3
        assert len(supervisor.sub_agents) == 0
        assert len(supervisor.active_tasks) == 0
    
    def test_register_agent(self, supervisor, mock_search_agent):
        supervisor.register_agent(mock_search_agent)
        assert "search" in supervisor.sub_agents
        assert supervisor.sub_agents["search"] == mock_search_agent
    
    @pytest.mark.asyncio
    async def test_analyze_query_complexity_simple(self, supervisor):
        with patch.object(supervisor, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_response = Mock()
            mock_response.output_text = "SIMPLE"
            mock_llm.return_value = mock_response
            
            complexity = await supervisor.analyze_query_complexity("What is Python?")
            assert complexity == ComplexityLevel.SIMPLE
    
    @pytest.mark.asyncio
    async def test_analyze_query_complexity_moderate(self, supervisor):
        with patch.object(supervisor, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_response = Mock()
            mock_response.output_text = "MODERATE"
            mock_llm.return_value = mock_response
            
            complexity = await supervisor.analyze_query_complexity(
                "Compare Python and JavaScript for web development"
            )
            assert complexity == ComplexityLevel.MODERATE
    
    @pytest.mark.asyncio
    async def test_analyze_query_complexity_complex(self, supervisor):
        with patch.object(supervisor, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_response = Mock()
            mock_response.output_text = "COMPLEX"
            mock_llm.return_value = mock_response
            
            complexity = await supervisor.analyze_query_complexity(
                "Analyze the impact of AI on global economy over the next decade"
            )
            assert complexity == ComplexityLevel.COMPLEX
    
    @pytest.mark.asyncio
    async def test_analyze_query_complexity_error_handling(self, supervisor):
        with patch.object(supervisor, '_call_llm', side_effect=Exception("API Error")):
            complexity = await supervisor.analyze_query_complexity("Test query")
            assert complexity == ComplexityLevel.MODERATE  # Default fallback
    
    def test_route_to_model(self, supervisor):
        assert supervisor.route_to_model(ComplexityLevel.SIMPLE) == ModelType.GPT5_NANO
        assert supervisor.route_to_model(ComplexityLevel.MODERATE) == ModelType.GPT5_MINI
        assert supervisor.route_to_model(ComplexityLevel.COMPLEX) == ModelType.GPT5_REGULAR
    
    @pytest.mark.asyncio
    async def test_decompose_query(self, supervisor):
        with patch.object(supervisor, '_call_llm', new_callable=AsyncMock) as mock_llm:
            tasks_json = [
                {
                    "id": "task_1",
                    "description": "Search for quantum computing basics",
                    "agent_type": "search",
                    "dependencies": [],
                    "priority": "high"
                },
                {
                    "id": "task_2",
                    "description": "Find recent research papers",
                    "agent_type": "search",
                    "dependencies": ["task_1"],
                    "priority": "medium"
                }
            ]
            
            mock_response = Mock()
            mock_response.output_text = json.dumps(tasks_json)
            mock_llm.return_value = mock_response
            
            tasks = await supervisor.decompose_query(sample_query)
            
            assert len(tasks) == 2
            assert tasks[0].description == "Search for quantum computing basics"
            assert tasks[1].description == "Find recent research papers"
    
    @pytest.mark.asyncio
    async def test_decompose_query_error_handling(self, supervisor, sample_query):
        with patch.object(supervisor, '_call_llm', side_effect=Exception("API Error")):
            tasks = await supervisor.decompose_query(sample_query)
            
            assert len(tasks) == 1
            assert tasks[0].description == sample_query
    
    @pytest.mark.asyncio
    async def test_delegate_task(self, supervisor, mock_search_agent):
        task = Task(
            id="test_task",
            description="Test search",
            complexity="simple"
        )
        
        with patch.object(mock_search_agent, 'execute', new_callable=AsyncMock) as mock_execute:
            mock_result = TaskResult(
                agent_id="search",
                task_id="test_task",
                status=Status.COMPLETED,
                result="Search results",
                execution_time=1.5,
                model_used="gpt-5-mini"
            )
            mock_execute.return_value = mock_result
            
            result = await supervisor.delegate_task(task, mock_search_agent)
            
            assert result == mock_result
            mock_execute.assert_called_once_with(task)
    
    @pytest.mark.asyncio
    async def test_aggregate_responses(self, supervisor):
        responses = [
            TaskResult(
                agent_id="search",
                task_id="task_1",
                status=Status.COMPLETED,
                result="Result 1: Quantum computing basics",
                citations=[],
                execution_time=1.0,
                model_used="gpt-5-mini",
                tokens_used={"total": 100}
            ),
            TaskResult(
                agent_id="search",
                task_id="task_2",
                status=Status.COMPLETED,
                result="Result 2: Recent developments",
                citations=[],
                execution_time=2.0,
                model_used="gpt-5-mini",
                tokens_used={"total": 150}
            ),
            TaskResult(
                agent_id="citation",
                task_id="task_3",
                status=Status.FAILED,
                result=None,
                citations=[],
                execution_time=0.5,
                model_used="gpt-5-mini",
                error="Citation error"
            )
        ]
        
        with patch.object(supervisor, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_response = Mock()
            mock_response.output_text = "Synthesized response"
            mock_llm.return_value = mock_response
            
            result = await supervisor.aggregate_responses(responses)
            
            assert result["response"] == "Synthesized response"
            assert result["total_tasks"] == 3
            assert result["successful_tasks"] == 2
            assert result["failed_tasks"] == 1
            assert result["total_tokens"] == 250
            assert result["execution_time"] == 3.5
    
    @pytest.mark.asyncio
    async def test_process_task_simple(self, supervisor, mock_search_agent):
        supervisor.register_agent(mock_search_agent)
        
        task = Task(
            id="simple_task",
            description="What is Python?",
            complexity="simple"
        )
        
        with patch.object(supervisor, 'analyze_query_complexity', return_value=ComplexityLevel.SIMPLE):
            with patch.object(supervisor, 'delegate_task', new_callable=AsyncMock) as mock_delegate:
                mock_result = TaskResult(
                    agent_id="search",
                    task_id="simple_task",
                    status=Status.COMPLETED,
                    result="Python is a programming language",
                    citations=[],
                    execution_time=1.0,
                    model_used="gpt-5-mini"
                )
                mock_delegate.return_value = mock_result
                
                result = await supervisor.process_task(task)
                
                assert result["response"] == "Python is a programming language"
                assert result["execution_time"] == 1.0
    
    @pytest.mark.asyncio
    async def test_process_task_complex(self, supervisor, mock_search_agent):
        supervisor.register_agent(mock_search_agent)
        
        task = Task(
            id="complex_task",
            description="Analyze AI impact on economy",
            complexity="complex"
        )
        
        with patch.object(supervisor, 'analyze_query_complexity', return_value=ComplexityLevel.COMPLEX):
            with patch.object(supervisor, 'decompose_query', new_callable=AsyncMock) as mock_decompose:
                subtasks = [
                    Task(id="sub1", description="Search economic data", complexity="simple", assigned_agent="search"),
                    Task(id="sub2", description="Search AI trends", complexity="simple", assigned_agent="search")
                ]
                mock_decompose.return_value = subtasks
                
                with patch.object(supervisor, 'delegate_task', new_callable=AsyncMock) as mock_delegate:
                    mock_delegate.side_effect = [
                        TaskResult(
                            agent_id="search",
                            task_id="sub1",
                            status=Status.COMPLETED,
                            result="Economic data",
                            citations=[],
                            execution_time=1.0,
                            model_used="gpt-5-mini"
                        ),
                        TaskResult(
                            agent_id="search",
                            task_id="sub2",
                            status=Status.COMPLETED,
                            result="AI trends",
                            citations=[],
                            execution_time=1.5,
                            model_used="gpt-5-mini"
                        )
                    ]
                    
                    with patch.object(supervisor, 'aggregate_responses', new_callable=AsyncMock) as mock_aggregate:
                        mock_aggregate.return_value = {
                            "response": "Comprehensive analysis",
                            "citations": [],
                            "total_tasks": 2,
                            "successful_tasks": 2,
                            "failed_tasks": 0
                        }
                        
                        result = await supervisor.process_task(task)
                        
                        assert result["response"] == "Comprehensive analysis"
                        assert task.id in supervisor.active_tasks
                        assert len(supervisor.active_tasks[task.id]) == 2
    
    @pytest.mark.asyncio
    async def test_orchestrate_success(self, supervisor, mock_search_agent):
        supervisor.register_agent(mock_search_agent)
        
        with patch.object(supervisor, 'process_task', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {
                "response": "Test response",
                "citations": [],
                "execution_time": 2.0
            }
            
            result = await supervisor.orchestrate("Test query")
            
            assert result["status"] == "success"
            assert result["response"] == "Test response"
            assert "task_id" in result
    
    @pytest.mark.asyncio
    async def test_orchestrate_failure(self, supervisor):
        with patch.object(supervisor, 'process_task', side_effect=Exception("Processing error")):
            result = await supervisor.orchestrate("Test query")
            
            assert result["status"] == "error"
            assert result["error"] == "Processing error"
            assert "task_id" in result
    
    @pytest.mark.asyncio
    async def test_process_critical_message(self, supervisor):
        message = AgentMessage(
            sender="search",
            recipient="supervisor",
            task_id="failing_task",
            payload={"error": "Search failed"},
            priority=Priority.CRITICAL
        )
        
        supervisor.active_tasks["failing_task"] = [
            Task(id="task1", description="test", complexity="simple")
        ]
        
        await supervisor._process_critical_message(message)
        # Should log warning and attempt recovery (implementation details)