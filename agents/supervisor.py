import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

from agents.base import BaseAgent
from agents.models import (
    AgentMessage, Task, TaskResult, Status, Priority,
    Citation, SearchResults
)
from config.settings import settings, TaskType, ModelType, ReasoningEffort, Verbosity

logger = logging.getLogger(__name__)

class SupervisorAgent(BaseAgent):
    """Supervisor agent that orchestrates task delegation and response aggregation."""
    
    def __init__(self, reasoning_effort: ReasoningEffort = ReasoningEffort.MEDIUM, 
                 verbosity: Verbosity = Verbosity.MEDIUM):
        super().__init__(
            agent_id="supervisor",
            model_type=ModelType.GPT5_REGULAR,
            temperature=0.3,  # Lower temperature for more consistent orchestration
            max_tokens=4096,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity
        )
        self.sub_agents: Dict[str, BaseAgent] = {}
        self.active_tasks: Dict[str, List[Task]] = {}
        
    def register_agent(self, agent: BaseAgent) -> None:
        """Register a sub-agent with the supervisor."""
        self.sub_agents[agent.agent_id] = agent
        logger.info(f"Registered agent: {agent.agent_id}")
    
    async def analyze_task_type(self, query: str) -> TaskType:
        """Analyze a user query to determine what type of response is needed."""
        
        prompt = f"""Analyze the user's query and classify what type of response they need:

- DIRECT_ANSWER: Factual questions that can be answered from training data (definitions, established facts, mathematical calculations, historical dates, etc.)
- SEARCH_NEEDED: Questions requiring current/recent information, news, trends, or real-time data  
- RESEARCH_REPORT: Requests for comprehensive analysis, comparisons, or in-depth reports requiring multiple sources

Query: {query}

Examples:
- "What is photosynthesis?" → DIRECT_ANSWER
- "What's the latest news about AI regulation?" → SEARCH_NEEDED  
- "Analyze the impact of climate change on agriculture" → RESEARCH_REPORT

Respond with only: DIRECT_ANSWER, SEARCH_NEEDED, or RESEARCH_REPORT"""
        
        try:
            # Use minimal reasoning for this classification task
            original_effort = self.reasoning_effort
            self.reasoning_effort = ReasoningEffort.MINIMAL
            
            response = await self._call_llm(input_text=prompt)
            
            # Restore original reasoning effort
            self.reasoning_effort = original_effort
            
            # Extract task type from response
            task_type_str = response.output_text.strip().upper() if hasattr(response, 'output_text') else response.choices[0].message.content.strip().upper()
            
            # Map to TaskType enum
            if task_type_str == "DIRECT_ANSWER":
                return TaskType.DIRECT_ANSWER
            elif task_type_str == "SEARCH_NEEDED":
                return TaskType.SEARCH_NEEDED
            elif task_type_str == "RESEARCH_REPORT":
                return TaskType.RESEARCH_REPORT
            else:
                # Default to search if unclear
                return TaskType.SEARCH_NEEDED
                
        except Exception as e:
            logger.error(f"Error analyzing task type: {str(e)}")
            # Default to search on error
            return TaskType.SEARCH_NEEDED
    
    
    def get_reasoning_effort(self, task_type: TaskType) -> ReasoningEffort:
        """Determine reasoning effort based on task type."""
        if task_type == TaskType.DIRECT_ANSWER:
            return ReasoningEffort.LOW  # Simple factual answers
        elif task_type == TaskType.SEARCH_NEEDED:
            return ReasoningEffort.MEDIUM  # Need good search and synthesis
        else:  # RESEARCH_REPORT
            return ReasoningEffort.HIGH  # Complex analysis requires thorough reasoning
    
    async def _handle_direct_answer(self, task: Task) -> Dict[str, Any]:
        """Handle direct answer tasks where supervisor can respond from training data."""
        
        prompt = f"""You are a knowledgeable assistant. The user is asking a factual question that can be answered from your training data. Provide a comprehensive, accurate answer.

Question: {task.description}

Provide a clear, factual response. If you're uncertain about any details, say so. Do not search for additional information - use only your training knowledge."""
        
        try:
            # Set appropriate reasoning effort for direct answers
            original_effort = self.reasoning_effort
            self.reasoning_effort = self.get_reasoning_effort(TaskType.DIRECT_ANSWER)
            
            response = await self._call_llm(input_text=prompt)
            
            # Restore original reasoning effort
            self.reasoning_effort = original_effort
            
            answer = response.output_text if hasattr(response, 'output_text') else response.choices[0].message.content
            
            return {
                "response": answer,
                "citations": [],  # No external sources for direct answers
                "execution_time": 0.0,  # Would need to track this properly
                "model_used": self._model_name,
                "total_tokens": response.usage.total_tokens if hasattr(response, 'usage') and response.usage else 0,
                "tokens_used": {
                    "input_tokens": response.usage.input_tokens if hasattr(response, 'usage') and response.usage else 0,
                    "output_tokens": response.usage.output_tokens if hasattr(response, 'usage') and response.usage else 0,
                    "total_tokens": response.usage.total_tokens if hasattr(response, 'usage') and response.usage else 0
                },
                "task_type": TaskType.DIRECT_ANSWER.value
            }
            
        except Exception as e:
            logger.error(f"Direct answer handling failed: {str(e)}")
            return {
                "response": f"I apologize, but I encountered an error while answering your question: {str(e)}",
                "citations": [],
                "execution_time": 0.0,
                "model_used": "error",
                "task_type": TaskType.DIRECT_ANSWER.value
            }
    
    async def decompose_query(self, query: str) -> List[Task]:
        """Decompose a complex query into subtasks."""
        
        prompt = f"""You are a task decomposition specialist. Break down the user's query into specific subtasks.
        Each subtask should be:
        1. Atomic and focused on a single objective
        2. Assigned to the appropriate agent type (search, citation, analysis)
        3. Include any dependencies on other tasks
        
        Return a JSON array of tasks with the following structure:
        [
            {{
                "id": "unique_task_id",
                "description": "task description",
                "agent_type": "search|citation|analysis",
                "dependencies": ["task_id1", "task_id2"],
                "priority": "low|medium|high|critical"
            }}
        ]
        
        Query: {query}"""
        
        try:
            # Use higher reasoning for task decomposition
            original_effort = self.reasoning_effort
            self.reasoning_effort = ReasoningEffort.MEDIUM
            
            response = await self._call_llm(input_text=prompt)
            
            # Restore original reasoning effort
            self.reasoning_effort = original_effort
            
            # Extract JSON from response
            response_text = response.output_text if hasattr(response, 'output_text') else response.choices[0].message.content
            tasks_json = json.loads(response_text)
            
            tasks = []
            for task_data in tasks_json:
                task = Task(
                    id=task_data.get("id", str(uuid4())),
                    description=task_data["description"],
                    complexity="simple",  # Subtasks should be simple
                    assigned_agent=task_data.get("agent_type", "search")
                )
                tasks.append(task)
            
            return tasks
            
        except Exception as e:
            logger.error(f"Error decomposing query: {str(e)}")
            # Return a single task as fallback
            return [Task(
                id=str(uuid4()),
                description=query,
                complexity="moderate",
                assigned_agent="search"
            )]
    
    async def delegate_task(self, task: Task, agent: BaseAgent) -> TaskResult:
        """Delegate a task to a specific agent."""
        logger.info(f"Delegating task {task.id} to agent {agent.agent_id}")
        
        # Send task to agent
        message = await self.send_message(
            recipient=agent.agent_id,
            task_id=task.id,
            payload={"task": task.model_dump()},
            priority=Priority.HIGH
        )
        
        # Execute task
        result = await agent.execute(task)
        return result
    
    async def aggregate_responses(self, responses: List[TaskResult]) -> Dict[str, Any]:
        """Aggregate and synthesize responses from multiple agents."""
        
        # Collect all successful results
        successful_results = [r for r in responses if r.status == Status.COMPLETED]
        failed_results = [r for r in responses if r.status == Status.FAILED]
        
        # Collect all citations
        all_citations = []
        for result in successful_results:
            all_citations.extend(result.citations)
        
        # Remove duplicate citations based on URL
        unique_citations = {c.url: c for c in all_citations}.values()
        
        # Prepare synthesis prompt
        results_text = "\n\n".join([
            f"Task: {r.task_id}\nResult: {r.result}"
            for r in successful_results
        ])
        
        prompt = f"""You are a research synthesis specialist. 
        Combine the following task results into a coherent, comprehensive response.
        Maintain factual accuracy and properly attribute information to sources.
        
        Task Results:
        {results_text}
        
        Synthesize these results into a unified response."""
        
        try:
            response = await self._call_llm(input_text=prompt)
            synthesized_response = response.output_text if hasattr(response, 'output_text') else response.choices[0].message.content
            
            return {
                "response": synthesized_response,
                "citations": list(unique_citations),
                "total_tasks": len(responses),
                "successful_tasks": len(successful_results),
                "failed_tasks": len(failed_results),
                "total_tokens": sum(r.tokens_used.get("total", 0) for r in responses),
                "execution_time": sum(r.execution_time for r in responses)
            }
            
        except Exception as e:
            logger.error(f"Error aggregating responses: {str(e)}")
            return {
                "response": "Error synthesizing responses",
                "citations": list(unique_citations),
                "total_tasks": len(responses),
                "successful_tasks": len(successful_results),
                "failed_tasks": len(failed_results),
                "error": str(e)
            }
    
    async def process_task(self, task: Task) -> Any:
        """Process a task as the supervisor."""
        
        try:
            # Analyze task type with fallback
            try:
                task_type = await self.analyze_task_type(task.description)
                logger.info(f"Task type: {task_type.value}")
            except Exception as e:
                logger.warning(f"Task type analysis failed, defaulting to SEARCH_NEEDED: {str(e)}")
                task_type = TaskType.SEARCH_NEEDED
            
            # Handle different task types
            if task_type == TaskType.DIRECT_ANSWER:
                # Supervisor handles directly
                return await self._handle_direct_answer(task)
            
            elif task_type == TaskType.RESEARCH_REPORT:
                # Complex research workflow
                try:
                    subtasks = await self.decompose_query(task.description)
                    self.active_tasks[task.id] = subtasks
                    
                    # Execute subtasks with error handling
                    results = []
                    for subtask in subtasks:
                        try:
                            agent_type = subtask.assigned_agent
                            
                            if agent_type in self.sub_agents:
                                agent = self.sub_agents[agent_type]
                                result = await self.delegate_task(subtask, agent)
                                if result.status == Status.COMPLETED:
                                    results.append(result)
                                else:
                                    logger.warning(f"Subtask failed: {result.error}")
                            else:
                                logger.warning(f"No agent found for type: {agent_type}")
                        except Exception as e:
                            logger.error(f"Subtask execution failed: {str(e)}")
                            continue
                    
                    # Aggregate results with fallback
                    if results:
                        final_response = await self.aggregate_responses(results)
                        return final_response
                    else:
                        # Fallback: try search instead
                        logger.warning("Research report processing failed, falling back to search")
                        task_type = TaskType.SEARCH_NEEDED
                        
                except Exception as e:
                    logger.error(f"Research report processing failed: {str(e)}")
                    task_type = TaskType.SEARCH_NEEDED  # Fallback
            
            # SEARCH_NEEDED processing (fallback for failed research reports too)
            if task_type == TaskType.SEARCH_NEEDED and "search" in self.sub_agents:
                try:
                    agent = self.sub_agents["search"]
                    # Let the search agent decide its own model - no routing from supervisor
                    result = await self.delegate_task(task, agent)
                    
                    if result.status == Status.COMPLETED:
                        return {
                            "response": result.result,
                            "citations": result.citations,
                            "execution_time": result.execution_time,
                            "model_used": result.model_used,
                            "total_tokens": result.tokens_used.get('total_tokens', 0),
                            "tokens_used": result.tokens_used
                        }
                    else:
                        return {
                            "response": f"Search agent failed: {result.error}",
                            "citations": [],
                            "execution_time": result.execution_time,
                            "model_used": result.model_used,
                            "total_tokens": result.tokens_used.get('total_tokens', 0),
                            "tokens_used": result.tokens_used
                        }
                except Exception as e:
                    logger.error(f"Search agent delegation failed: {str(e)}")
                    return {
                        "response": f"I apologize, but I encountered an error processing your request: {str(e)}",
                        "citations": [],
                        "execution_time": 0.0,
                        "model_used": "error"
                    }
            else:
                return {
                    "response": "I apologize, but the search functionality is currently unavailable.",
                    "citations": [],
                    "execution_time": 0.0,
                    "model_used": "fallback"
                }
                
        except Exception as e:
            logger.error(f"Supervisor task processing failed: {str(e)}")
            return {
                "response": f"I encountered an unexpected error: {str(e)}",
                "citations": [],
                "execution_time": 0.0,
                "model_used": "error"
            }
    
    async def _process_critical_message(self, message: AgentMessage) -> None:
        """Handle critical priority messages immediately."""
        logger.warning(f"Critical message received from {message.sender}: {message.payload}")
        
        # Handle error recovery or urgent coordination
        if "error" in message.payload:
            # Attempt to reassign task to another agent
            task_id = message.task_id
            if task_id in self.active_tasks:
                # Find alternative agent or retry
                logger.info(f"Attempting to recover task {task_id}")
                # Implementation for task recovery
                pass
    
    async def orchestrate(self, query: str, trace_id: str = None) -> Dict[str, Any]:
        """Main orchestration entry point for processing user queries."""
        
        # Create main task
        main_task = Task(
            id=str(uuid4()),
            description=query,
            complexity="unknown"
        )
        
        logger.info(f"Starting orchestration for task {main_task.id}")
        
        # Set up Phoenix tracing if trace_id provided
        if trace_id:
            await self.set_trace_context(trace_id)
            # Also set trace context for all registered sub-agents
            for agent in self.sub_agents.values():
                await agent.set_trace_context(trace_id)
        
        try:
            result = await self.process_task(main_task)
            return {
                "status": "success",
                "task_id": main_task.id,
                **result
            }
        except Exception as e:
            logger.error(f"Orchestration failed: {str(e)}")
            return {
                "status": "error",
                "task_id": main_task.id,
                "error": str(e)
            }
        finally:
            # Clear trace context after orchestration
            if trace_id:
                await self.clear_trace_context()
                for agent in self.sub_agents.values():
                    await agent.clear_trace_context()