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
from config.settings import settings, ComplexityLevel, ModelType, ReasoningEffort, Verbosity

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
    
    async def analyze_query_complexity(self, query: str) -> ComplexityLevel:
        """Analyze the complexity of a user query to determine routing."""
        
        prompt = f"""Analyze the complexity of the given query and classify it as:
        - SIMPLE: Factual queries, definitions, simple lookups (< 100 tokens, single concept)
        - MODERATE: Multi-step reasoning, synthesis of 2-3 sources (< 500 tokens, 2-3 concepts)
        - COMPLEX: Deep analysis, multiple domains, creative tasks (> 500 tokens or multiple domains)
        
        Query: {query}
        
        Respond with only: SIMPLE, MODERATE, or COMPLEX"""
        
        try:
            # Use minimal reasoning for this classification task
            original_effort = self.reasoning_effort
            self.reasoning_effort = ReasoningEffort.MINIMAL
            
            response = await self._call_llm(input_text=prompt)
            
            # Restore original reasoning effort
            self.reasoning_effort = original_effort
            
            # Extract complexity from response
            complexity_str = response.output_text.strip().upper() if hasattr(response, 'output_text') else response.choices[0].message.content.strip().upper()
            
            # Map to ComplexityLevel enum
            if complexity_str == "SIMPLE":
                return ComplexityLevel.SIMPLE
            elif complexity_str == "MODERATE":
                return ComplexityLevel.MODERATE
            else:
                return ComplexityLevel.COMPLEX
                
        except Exception as e:
            logger.error(f"Error analyzing query complexity: {str(e)}")
            # Default to MODERATE on error
            return ComplexityLevel.MODERATE
    
    def route_to_model(self, complexity: ComplexityLevel) -> ModelType:
        """Route to appropriate model based on complexity."""
        if complexity == ComplexityLevel.SIMPLE:
            return ModelType.GPT5_NANO
        elif complexity == ComplexityLevel.MODERATE:
            return ModelType.GPT5_MINI
        else:
            return ModelType.GPT5_REGULAR
    
    def get_reasoning_effort(self, complexity: ComplexityLevel) -> ReasoningEffort:
        """Determine reasoning effort based on complexity."""
        if complexity == ComplexityLevel.SIMPLE:
            return ReasoningEffort.MINIMAL
        elif complexity == ComplexityLevel.MODERATE:
            return ReasoningEffort.LOW
        else:
            return ReasoningEffort.MEDIUM  # or HIGH for very complex tasks
    
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
        
        # Analyze complexity
        complexity = await self.analyze_query_complexity(task.description)
        logger.info(f"Query complexity: {complexity.value}")
        
        # Decompose into subtasks if complex
        if complexity == ComplexityLevel.COMPLEX:
            subtasks = await self.decompose_query(task.description)
            self.active_tasks[task.id] = subtasks
            
            # Execute subtasks in parallel where possible
            results = []
            for subtask in subtasks:
                # Determine which agent should handle this
                agent_type = subtask.assigned_agent
                
                if agent_type in self.sub_agents:
                    agent = self.sub_agents[agent_type]
                    result = await self.delegate_task(subtask, agent)
                    results.append(result)
                else:
                    logger.warning(f"No agent found for type: {agent_type}")
            
            # Aggregate results
            final_response = await self.aggregate_responses(results)
            return final_response
            
        else:
            # For simple/moderate queries, route to appropriate agent directly
            if "search" in self.sub_agents:
                agent = self.sub_agents["search"]
                # Update agent's model based on complexity
                agent.model_type = self.route_to_model(complexity)
                result = await self.delegate_task(task, agent)
                
                return {
                    "response": result.result,
                    "citations": result.citations,
                    "execution_time": result.execution_time,
                    "model_used": result.model_used
                }
            else:
                return {"error": "No search agent available"}
    
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
    
    async def orchestrate(self, query: str) -> Dict[str, Any]:
        """Main orchestration entry point for processing user queries."""
        
        # Create main task
        main_task = Task(
            id=str(uuid4()),
            description=query,
            complexity="unknown"
        )
        
        logger.info(f"Starting orchestration for task {main_task.id}")
        
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