#!/usr/bin/env python3
"""
Evaluation framework with Phoenix tracing integration.
Provides comprehensive evaluation capabilities for the multi-agent research system.
"""
import asyncio
import json
import logging
import statistics
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from uuid import uuid4

from agents.supervisor import SupervisorAgent
from agents.models import Citation
from evaluation_dataset import EVALUATION_QUERIES, EvalQuery, get_queries_by_task_type
from evaluation.phoenix_integration import phoenix_integration
from config.settings import settings, TaskType

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Result of evaluating a single query."""
    query_id: int
    query_text: str
    expected_complexity: TaskType
    actual_complexity: TaskType
    response: str
    citations: List[Citation]
    execution_time: float
    total_tokens: int
    model_used: str
    
    # Quality metrics
    accuracy_score: Optional[float] = None
    citation_completeness: Optional[float] = None
    response_coherence: Optional[float] = None
    source_relevance: Optional[float] = None
    
    # Status
    success: bool = True
    error: Optional[str] = None

@dataclass
class EvaluationSession:
    """Configuration and results for an evaluation session."""
    session_id: str
    session_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Configuration
    agent_config: Dict[str, Any] = None
    phoenix_enabled: bool = True
    
    # Results
    results: List[EvaluationResult] = None
    summary_metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.results is None:
            self.results = []

class EvaluationFramework:
    """Main evaluation framework with Phoenix integration."""
    
    def __init__(self, 
                 supervisor_agent: SupervisorAgent,
                 enable_phoenix: bool = True,
                 parallel_execution: bool = True,
                 max_concurrent: int = 3):
        self.supervisor_agent = supervisor_agent
        self.enable_phoenix = enable_phoenix
        self.parallel_execution = parallel_execution
        self.max_concurrent = max_concurrent
        self._current_session: Optional[EvaluationSession] = None
        
    async def create_session(self, session_name: str = None) -> EvaluationSession:
        """Create a new evaluation session."""
        session_id = str(uuid4())
        session_name = session_name or f"eval_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        session = EvaluationSession(
            session_id=session_id,
            session_name=session_name,
            start_time=datetime.now(),
            phoenix_enabled=self.enable_phoenix
        )
        
        self._current_session = session
        
        # Start Phoenix session
        if self.enable_phoenix:
            try:
                await phoenix_integration.start_evaluation_session(session_name)
                logger.info(f"Started Phoenix evaluation session: {session_name}")
            except Exception as e:
                logger.warning(f"Failed to start Phoenix session: {e}")
        
        return session
    
    async def evaluate_single_query(self, 
                                   eval_query: EvalQuery, 
                                   trace_id: str = None) -> EvaluationResult:
        """Evaluate a single query with Phoenix tracing."""
        start_time = datetime.now()
        
        # Start trace if not provided
        if trace_id is None and self.enable_phoenix:
            try:
                trace_id = await phoenix_integration.start_trace(
                    trace_name=f"eval_query_{eval_query.id}",
                    metadata={
                        "query_id": eval_query.id,
                        "expected_complexity": eval_query.complexity.value,
                        "domain": eval_query.domain,
                        "requires_current_info": eval_query.requires_current_info
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to start Phoenix trace: {e}")
                trace_id = None
        
        try:
            # Execute query using supervisor agent
            result = await self.supervisor_agent.orchestrate(
                query=eval_query.query,
                trace_id=trace_id
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            if result.get("status") == "success":
                # Extract response details
                response_text = result.get("response", "")
                citations = result.get("citations", [])
                total_tokens = result.get("total_tokens", 0)
                model_used = result.get("model_used", "unknown")
                
                # Determine actual complexity (this would be set by the supervisor)
                # For now, we'll use a simple heuristic
                actual_complexity = self._determine_actual_complexity(result)
                
                # Create evaluation result
                eval_result = EvaluationResult(
                    query_id=eval_query.id,
                    query_text=eval_query.query,
                    expected_complexity=eval_query.complexity,
                    actual_complexity=actual_complexity,
                    response=response_text,
                    citations=citations,
                    execution_time=execution_time,
                    total_tokens=total_tokens,
                    model_used=model_used,
                    success=True
                )
                
                # Analyze quality using Phoenix MCP if available
                if self.enable_phoenix:
                    try:
                        quality_scores = await phoenix_integration.analyze_response_quality(
                            query=eval_query.query,
                            response=response_text,
                            citations=citations,
                            expected_sources=eval_query.expected_sources
                        )
                        
                        eval_result.accuracy_score = quality_scores.get("factual_accuracy", 0.0)
                        eval_result.citation_completeness = quality_scores.get("citation_completeness", 0.0)
                        eval_result.response_coherence = quality_scores.get("response_coherence", 0.0)
                        eval_result.source_relevance = quality_scores.get("source_relevance", 0.0)
                        
                    except Exception as e:
                        logger.warning(f"Failed to analyze response quality: {e}")
                
                # Log evaluation result to Phoenix
                if self.enable_phoenix and trace_id:
                    try:
                        await phoenix_integration.log_evaluation_result(
                            trace_id=trace_id,
                            query_id=eval_query.id,
                            query_text=eval_query.query,
                            expected_complexity=eval_query.complexity.value,
                            actual_complexity=actual_complexity.value,
                            response=response_text,
                            citations=citations,
                            execution_time=execution_time,
                            total_tokens=total_tokens,
                            accuracy_score=eval_result.accuracy_score,
                            quality_metrics={
                                "citation_completeness": eval_result.citation_completeness,
                                "response_coherence": eval_result.response_coherence,
                                "source_relevance": eval_result.source_relevance
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Failed to log evaluation result: {e}")
                
                return eval_result
                
            else:
                # Handle orchestration failure
                execution_time = (datetime.now() - start_time).total_seconds()
                error_msg = result.get("error", "Unknown orchestration error")
                
                return EvaluationResult(
                    query_id=eval_query.id,
                    query_text=eval_query.query,
                    expected_complexity=eval_query.complexity,
                    actual_complexity=eval_query.complexity,  # Default to expected
                    response="",
                    citations=[],
                    execution_time=execution_time,
                    total_tokens=0,
                    model_used="unknown",
                    success=False,
                    error=error_msg
                )
                
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Evaluation failed for query {eval_query.id}: {str(e)}")
            
            return EvaluationResult(
                query_id=eval_query.id,
                query_text=eval_query.query,
                expected_complexity=eval_query.complexity,
                actual_complexity=eval_query.complexity,
                response="",
                citations=[],
                execution_time=execution_time,
                total_tokens=0,
                model_used="unknown",
                success=False,
                error=str(e)
            )
    
    async def evaluate_complexity_level(self, 
                                      complexity: TaskType,
                                      max_queries: int = None) -> List[EvaluationResult]:
        """Evaluate all queries of a specific complexity level."""
        queries = get_queries_by_task_type(complexity)
        
        if max_queries:
            queries = queries[:max_queries]
        
        logger.info(f"Evaluating {len(queries)} queries of complexity: {complexity.value}")
        
        if self.parallel_execution:
            # Use semaphore to limit concurrent execution
            semaphore = asyncio.Semaphore(self.max_concurrent)
            
            async def evaluate_with_semaphore(query):
                async with semaphore:
                    return await self.evaluate_single_query(query)
            
            # Execute queries in parallel
            tasks = [evaluate_with_semaphore(query) for query in queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            evaluation_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Query {queries[i].id} failed: {result}")
                    # Create failed result
                    evaluation_results.append(EvaluationResult(
                        query_id=queries[i].id,
                        query_text=queries[i].query,
                        expected_complexity=queries[i].complexity,
                        actual_complexity=queries[i].complexity,
                        response="",
                        citations=[],
                        execution_time=0.0,
                        total_tokens=0,
                        model_used="unknown",
                        success=False,
                        error=str(result)
                    ))
                else:
                    evaluation_results.append(result)
            
            return evaluation_results
        else:
            # Sequential execution
            results = []
            for query in queries:
                try:
                    result = await self.evaluate_single_query(query)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Query {query.id} failed: {e}")
                    results.append(EvaluationResult(
                        query_id=query.id,
                        query_text=query.query,
                        expected_complexity=query.complexity,
                        actual_complexity=query.complexity,
                        response="",
                        citations=[],
                        execution_time=0.0,
                        total_tokens=0,
                        model_used="unknown",
                        success=False,
                        error=str(e)
                    ))
            
            return results
    
    async def evaluate_full_dataset(self, 
                                   max_queries_per_complexity: int = None) -> EvaluationSession:
        """Evaluate the complete dataset across all complexity levels."""
        if not self._current_session:
            await self.create_session("full_dataset_evaluation")
        
        session = self._current_session
        all_results = []
        
        # Evaluate each complexity level
        for complexity in TaskType:
            logger.info(f"Starting evaluation for complexity: {complexity.value}")
            
            complexity_results = await self.evaluate_complexity_level(
                complexity, 
                max_queries_per_complexity
            )
            all_results.extend(complexity_results)
            
            logger.info(f"Completed {len(complexity_results)} queries for {complexity.value}")
        
        session.results = all_results
        session.end_time = datetime.now()
        session.summary_metrics = self._calculate_summary_metrics(all_results)
        
        # Close Phoenix session
        if self.enable_phoenix:
            try:
                final_metrics = await phoenix_integration.close_session()
                session.summary_metrics["phoenix_metrics"] = final_metrics
            except Exception as e:
                logger.warning(f"Failed to close Phoenix session: {e}")
        
        return session
    
    def _determine_actual_complexity(self, orchestration_result: Dict[str, Any]) -> TaskType:
        """Determine the actual complexity based on orchestration result."""
        # This is a simple heuristic - in practice, the supervisor would track this
        model_used = orchestration_result.get("model_used", "")
        
        if "nano" in model_used.lower():
            return TaskType.DIRECT_ANSWER
        elif "mini" in model_used.lower():
            return TaskType.SEARCH_NEEDED
        else:
            return TaskType.RESEARCH_REPORT
    
    def _calculate_summary_metrics(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Calculate summary metrics for evaluation results."""
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        if not results:
            return {}
        
        metrics = {
            "total_queries": len(results),
            "successful_queries": len(successful_results),
            "failed_queries": len(failed_results),
            "success_rate": len(successful_results) / len(results),
            
            "avg_execution_time": statistics.mean([r.execution_time for r in successful_results]) if successful_results else 0,
            "median_execution_time": statistics.median([r.execution_time for r in successful_results]) if successful_results else 0,
            "total_tokens": sum([r.total_tokens for r in successful_results]),
            "avg_tokens_per_query": statistics.mean([r.total_tokens for r in successful_results]) if successful_results else 0,
        }
        
        # Complexity accuracy
        complexity_correct = sum(1 for r in successful_results 
                               if r.expected_complexity == r.actual_complexity)
        metrics["complexity_accuracy"] = complexity_correct / len(successful_results) if successful_results else 0
        
        # Quality metrics (if available)
        quality_results = [r for r in successful_results if r.accuracy_score is not None]
        if quality_results:
            metrics["avg_accuracy_score"] = statistics.mean([r.accuracy_score for r in quality_results])
            metrics["avg_citation_completeness"] = statistics.mean([r.citation_completeness for r in quality_results])
            metrics["avg_response_coherence"] = statistics.mean([r.response_coherence for r in quality_results])
            metrics["avg_source_relevance"] = statistics.mean([r.source_relevance for r in quality_results])
        
        # Breakdown by complexity
        for complexity in TaskType:
            complexity_results = [r for r in results if r.expected_complexity == complexity]
            complexity_successful = [r for r in complexity_results if r.success]
            
            key_prefix = f"{complexity.value.lower()}_"
            metrics[f"{key_prefix}total"] = len(complexity_results)
            metrics[f"{key_prefix}successful"] = len(complexity_successful)
            metrics[f"{key_prefix}success_rate"] = len(complexity_successful) / len(complexity_results) if complexity_results else 0
            
            if complexity_successful:
                metrics[f"{key_prefix}avg_execution_time"] = statistics.mean([r.execution_time for r in complexity_successful])
                metrics[f"{key_prefix}avg_tokens"] = statistics.mean([r.total_tokens for r in complexity_successful])
        
        return metrics
    
    def export_results(self, session: EvaluationSession, format: str = "json") -> str:
        """Export evaluation results in the specified format."""
        if format.lower() == "json":
            return self._export_json(session)
        elif format.lower() == "csv":
            return self._export_csv(session)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_json(self, session: EvaluationSession) -> str:
        """Export results as JSON."""
        export_data = {
            "session_info": {
                "session_id": session.session_id,
                "session_name": session.session_name,
                "start_time": session.start_time.isoformat(),
                "end_time": session.end_time.isoformat() if session.end_time else None,
                "phoenix_enabled": session.phoenix_enabled
            },
            "summary_metrics": session.summary_metrics,
            "results": [
                {
                    "query_id": r.query_id,
                    "query_text": r.query_text,
                    "expected_complexity": r.expected_complexity.value,
                    "actual_complexity": r.actual_complexity.value,
                    "response": r.response,
                    "citations_count": len(r.citations),
                    "execution_time": r.execution_time,
                    "total_tokens": r.total_tokens,
                    "model_used": r.model_used,
                    "success": r.success,
                    "error": r.error,
                    "quality_scores": {
                        "accuracy_score": r.accuracy_score,
                        "citation_completeness": r.citation_completeness,
                        "response_coherence": r.response_coherence,
                        "source_relevance": r.source_relevance
                    }
                }
                for r in session.results
            ]
        }
        
        return json.dumps(export_data, indent=2, default=str)
    
    def _export_csv(self, session: EvaluationSession) -> str:
        """Export results as CSV."""
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Headers
        headers = [
            "query_id", "query_text", "expected_complexity", "actual_complexity",
            "execution_time", "total_tokens", "model_used", "success", "error",
            "accuracy_score", "citation_completeness", "response_coherence", "source_relevance",
            "citations_count", "response_length"
        ]
        writer.writerow(headers)
        
        # Data rows
        for r in session.results:
            writer.writerow([
                r.query_id, r.query_text, r.expected_complexity.value, r.actual_complexity.value,
                r.execution_time, r.total_tokens, r.model_used, r.success, r.error,
                r.accuracy_score, r.citation_completeness, r.response_coherence, r.source_relevance,
                len(r.citations), len(r.response)
            ])
        
        return output.getvalue()

# Global framework instance (to be initialized with actual agents)
evaluation_framework: Optional[EvaluationFramework] = None

def initialize_framework(supervisor_agent: SupervisorAgent, **kwargs) -> EvaluationFramework:
    """Initialize the global evaluation framework."""
    global evaluation_framework
    evaluation_framework = EvaluationFramework(supervisor_agent, **kwargs)
    return evaluation_framework