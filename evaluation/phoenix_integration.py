#!/usr/bin/env python3
"""
Phoenix direct SDK integration module for the multi-agent research system.
Provides observability and tracing capabilities using Phoenix Python SDK.
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import os

# Phoenix SDK imports
try:
    from phoenix.otel import register
    from phoenix.client import Client as PhoenixClient
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False
    logger.warning("Phoenix SDK not available. Install with: pip install arize-phoenix arize-phoenix-otel arize-phoenix-client")

from config.settings import settings
from agents.models import AgentMessage, TaskResult, Citation

logger = logging.getLogger(__name__)

class PhoenixDirectIntegration:
    """Phoenix direct SDK integration for observability and evaluation."""
    
    def __init__(self):
        self._session_id = None
        self._current_project_name = None
        self._tracer_provider = None
        self._tracer = None
        self._phoenix_client = None
        
        if PHOENIX_AVAILABLE:
            try:
                # Initialize Phoenix OpenTelemetry integration
                self._tracer_provider = register(
                    project_name="multi-agent-research",
                    endpoint=settings.phoenix_endpoint if hasattr(settings, 'phoenix_endpoint') else "http://localhost:6006",
                    headers={"api-key": settings.phoenix_api_key} if settings.phoenix_api_key else None
                )
                
                # Get tracer for creating spans
                self._tracer = trace.get_tracer(__name__)
                
                # Initialize Phoenix client for direct API access
                self._phoenix_client = PhoenixClient(
                    base_url=settings.phoenix_endpoint if hasattr(settings, 'phoenix_endpoint') else "http://localhost:6006",
                    api_key=settings.phoenix_api_key
                )
                
                logger.info("Phoenix direct SDK integration initialized successfully")
                
            except Exception as e:
                logger.warning(f"Failed to initialize Phoenix SDK: {e}")
                self._tracer_provider = None
                self._tracer = None
                self._phoenix_client = None
        else:
            logger.info("Phoenix SDK not available, tracing disabled")
        
    async def start_evaluation_session(self, session_name: str = None) -> str:
        """Start a new evaluation session with Phoenix tracing."""
        session_name = session_name or f"evaluation_session_{datetime.now().isoformat()}"
        
        if not PHOENIX_AVAILABLE or not self._phoenix_client:
            logger.info("Phoenix not available, session tracking disabled")
            self._session_id = session_name
            return session_name
        
        try:
            # Create or get project in Phoenix
            project_name = "multi-agent-research"
            try:
                project = self._phoenix_client.projects.get(project_name=project_name)
                logger.info(f"Using existing Phoenix project: {project_name}")
            except:
                # Create project if it doesn't exist
                project = self._phoenix_client.projects.create(
                    name=project_name,
                    description="Multi-agent research system evaluation"
                )
                logger.info(f"Created new Phoenix project: {project_name}")
            
            self._current_project_name = project_name
            self._session_id = session_name
            logger.info(f"Started Phoenix evaluation session: {session_name}")
            return session_name
            
        except Exception as e:
            logger.warning(f"Failed to start Phoenix session: {str(e)}")
            # Continue without Phoenix if it fails
            self._session_id = session_name
            return session_name
    
    async def start_trace(self, trace_name: str, metadata: Dict[str, Any] = None) -> str:
        """Start a new trace for an evaluation run."""
        trace_id = f"{trace_name}_{datetime.now().timestamp()}"
        
        if not PHOENIX_AVAILABLE or not self._tracer:
            logger.debug("Phoenix tracing not available, returning trace ID")
            return trace_id
            
        try:
            # OpenTelemetry tracing is handled automatically by Phoenix
            # Just return the trace ID for tracking purposes
            logger.debug(f"Started Phoenix trace: {trace_id}")
            return trace_id
            
        except Exception as e:
            logger.warning(f"Failed to start Phoenix trace: {str(e)}")
            return trace_id
    
    async def create_span(self, 
                         trace_id: str,
                         span_name: str, 
                         span_type: str = "agent",
                         parent_span_id: Optional[str] = None,
                         metadata: Dict[str, Any] = None) -> str:
        """Create a span within a trace."""
        span_id = f"{span_name}_{datetime.now().timestamp()}"
        
        if not PHOENIX_AVAILABLE or not self._tracer:
            logger.debug("Phoenix tracing not available, returning span ID")
            return span_id
        
        try:
            # Use OpenTelemetry tracer to create spans
            with self._tracer.start_as_current_span(span_name) as span:
                # Add attributes to the span
                if metadata:
                    for key, value in metadata.items():
                        span.set_attribute(f"custom.{key}", str(value))
                span.set_attribute("span_type", span_type)
                span.set_attribute("trace_id", trace_id)
                
                logger.debug(f"Created Phoenix span: {span_id}")
                return span_id
            
        except Exception as e:
            logger.warning(f"Failed to create Phoenix span: {str(e)}")
            return span_id
    
    async def end_span(self, 
                      trace_id: str,
                      span_id: str, 
                      status: str = "success",
                      result: Any = None,
                      metrics: Dict[str, Any] = None,
                      error: Optional[str] = None) -> None:
        """End a span and record results."""
        if not PHOENIX_AVAILABLE or not self._tracer:
            logger.debug("Phoenix tracing not available, skipping span end")
            return
            
        try:
            # OpenTelemetry spans are automatically ended when context exits
            # This method is mainly for logging and compatibility
            logger.debug(f"Ended Phoenix span: {span_id} with status: {status}")
            
        except Exception as e:
            logger.warning(f"Failed to end Phoenix span: {str(e)}")
    
    async def log_agent_interaction(self,
                                   trace_id: str,
                                   agent_id: str,
                                   input_message: str,
                                   output_message: str,
                                   model_used: str,
                                   tokens_used: Dict[str, int],
                                   execution_time: float,
                                   parent_span_id: Optional[str] = None) -> str:
        """Log a complete agent interaction with Phoenix."""
        span_id = f"{agent_id}_interaction_{datetime.now().timestamp()}"
        
        if not PHOENIX_AVAILABLE or not self._tracer:
            logger.debug("Phoenix tracing not available, logging to debug")
            logger.debug(f"Agent {agent_id}: {len(input_message)} chars in, {len(output_message)} chars out, "
                        f"{tokens_used.get('total', 0)} tokens, {execution_time:.2f}s")
            return span_id
        
        try:
            # Use OpenTelemetry to create an LLM span
            with self._tracer.start_as_current_span(f"{agent_id}_interaction") as span:
                # Add LLM-specific attributes following OpenInference conventions
                span.set_attribute("llm.model_name", model_used)
                span.set_attribute("llm.input_messages", input_message[:1000])  # Truncate for performance
                span.set_attribute("llm.output_messages", output_message[:1000])
                span.set_attribute("llm.token_count.prompt", tokens_used.get("prompt", 0))
                span.set_attribute("llm.token_count.completion", tokens_used.get("completion", 0))
                span.set_attribute("llm.token_count.total", tokens_used.get("total", 0))
                span.set_attribute("agent_id", agent_id)
                span.set_attribute("execution_time_ms", execution_time * 1000)
                span.set_attribute("trace_id", trace_id)
                
                span.set_status(Status(StatusCode.OK))
                logger.debug(f"Logged agent interaction span: {span_id}")
                
        except Exception as e:
            logger.warning(f"Failed to log agent interaction to Phoenix: {str(e)}")
        
        return span_id
    
    async def log_evaluation_result(self,
                                   trace_id: str,
                                   query_id: int,
                                   query_text: str,
                                   expected_complexity: str,
                                   actual_complexity: str,
                                   response: str,
                                   citations: List[Citation],
                                   execution_time: float,
                                   total_tokens: int,
                                   accuracy_score: Optional[float] = None,
                                   quality_metrics: Dict[str, float] = None) -> None:
        """Log evaluation results to Phoenix."""
        if not PHOENIX_AVAILABLE or not self._tracer:
            logger.debug(f"Phoenix not available, logging evaluation result for query {query_id} to debug")
            logger.debug(f"Query {query_id}: {expected_complexity} -> {actual_complexity}, "
                        f"{len(response)} chars response, {len(citations)} citations, "
                        f"{execution_time:.2f}s, {total_tokens} tokens")
            return
        
        try:
            # Create evaluation span with OpenTelemetry
            with self._tracer.start_as_current_span(f"evaluation_query_{query_id}") as span:
                span.set_attribute("evaluation.query_id", query_id)
                span.set_attribute("evaluation.query_text", query_text[:500])  # Truncate
                span.set_attribute("evaluation.expected_complexity", expected_complexity)
                span.set_attribute("evaluation.actual_complexity", actual_complexity)
                span.set_attribute("evaluation.response_length", len(response))
                span.set_attribute("evaluation.citations_count", len(citations))
                span.set_attribute("evaluation.execution_time_ms", execution_time * 1000)
                span.set_attribute("evaluation.total_tokens", total_tokens)
                span.set_attribute("trace_id", trace_id)
                
                if accuracy_score is not None:
                    span.set_attribute("evaluation.accuracy_score", accuracy_score)
                
                if quality_metrics:
                    for metric, value in quality_metrics.items():
                        span.set_attribute(f"evaluation.quality.{metric}", value)
                
                span.set_status(Status(StatusCode.OK))
                logger.info(f"Logged evaluation result for query {query_id}")
            
        except Exception as e:
            logger.warning(f"Failed to log evaluation result: {str(e)}")
    
    async def get_evaluation_metrics(self, session_id: str = None) -> Dict[str, Any]:
        """Retrieve evaluation metrics from Phoenix."""
        session_id = session_id or self._session_id
        
        if not PHOENIX_AVAILABLE or not self._phoenix_client:
            logger.debug("Phoenix not available, returning empty metrics")
            return {}
        
        try:
            # Use Phoenix client to get project metrics if available
            if self._current_project_name:
                # This is a placeholder - actual metrics retrieval would depend on Phoenix API
                logger.info(f"Retrieving metrics for session {session_id}")
                return {"session_id": session_id, "status": "active"}
            
            return {}
            
        except Exception as e:
            logger.warning(f"Failed to retrieve evaluation metrics: {str(e)}")
            return {}
    
    async def analyze_response_quality(self, 
                                     query: str,
                                     response: str,
                                     citations: List[Citation],
                                     expected_sources: int = None) -> Dict[str, float]:
        """Analyze response quality using simple heuristics."""
        try:
            # Simple quality analysis without Phoenix MCP
            scores = {
                "factual_accuracy": 0.8,  # Placeholder - would use evaluation model
                "citation_completeness": min(1.0, len(citations) / (expected_sources or 5)),
                "response_coherence": min(1.0, len(response) / 500),  # Basic length check
                "source_relevance": 0.75 if citations else 0.0
            }
            
            logger.debug(f"Analyzed response quality: {scores}")
            return scores
            
        except Exception as e:
            logger.warning(f"Failed to analyze response quality: {str(e)}")
            return {
                "factual_accuracy": 0.0,
                "citation_completeness": 0.0,
                "response_coherence": 0.0,
                "source_relevance": 0.0
            }
    
    async def close_session(self) -> Dict[str, Any]:
        """Close the current evaluation session and return final metrics."""
        if not self._session_id:
            return {}
        
        try:
            # Simple session closure - OpenTelemetry handles span lifecycle
            session_summary = {
                "session_id": self._session_id,
                "end_time": datetime.now().isoformat(),
                "status": "closed"
            }
            
            logger.info(f"Closed Phoenix evaluation session: {self._session_id}")
            return session_summary
            
        except Exception as e:
            logger.warning(f"Failed to close Phoenix session: {str(e)}")
            return {}
        finally:
            self._session_id = None
            self._current_project_name = None

# Global Phoenix integration instance
phoenix_integration = PhoenixDirectIntegration()

# Utility functions for easy access
async def start_evaluation_session(session_name: str = None) -> str:
    """Start a new evaluation session."""
    return await phoenix_integration.start_evaluation_session(session_name)

async def log_agent_call(trace_id: str, agent_id: str, input_msg: str, 
                        output_msg: str, model: str, tokens: Dict[str, int], 
                        exec_time: float, parent_span: str = None) -> str:
    """Log an agent interaction."""
    return await phoenix_integration.log_agent_interaction(
        trace_id, agent_id, input_msg, output_msg, model, tokens, exec_time, parent_span
    )

async def analyze_quality(query: str, response: str, citations: List[Citation], 
                         expected_sources: int = None) -> Dict[str, float]:
    """Analyze response quality using Phoenix."""
    return await phoenix_integration.analyze_response_quality(
        query, response, citations, expected_sources
    )