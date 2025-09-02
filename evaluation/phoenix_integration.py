#!/usr/bin/env python3
"""
Phoenix MCP integration module for the multi-agent research system.
Provides observability and tracing capabilities using Phoenix MCP server.
"""
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from openai import OpenAI
from openai.types.responses import Response as OpenAIResponse

from config.settings import settings
from agents.models import AgentMessage, TaskResult, Citation

logger = logging.getLogger(__name__)

class PhoenixMCPIntegration:
    """Phoenix MCP server integration for observability and evaluation."""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.phoenix_mcp_config = {
            "type": "mcp",
            "server_label": settings.phoenix_mcp_server_label,
            "server_url": settings.phoenix_mcp_server_url,
            "authorization": settings.phoenix_api_key,
            "require_approval": settings.phoenix_mcp_require_approval
        }
        self._session_id = None
        self._current_conversation_id = None
        
    async def start_evaluation_session(self, session_name: str = None) -> str:
        """Start a new evaluation session with Phoenix tracing."""
        session_name = session_name or f"evaluation_session_{datetime.now().isoformat()}"
        
        try:
            # Initialize Phoenix session using MCP
            response = await self._call_phoenix_mcp("create_session", {
                "session_name": session_name,
                "metadata": {
                    "type": "evaluation",
                    "timestamp": datetime.now().isoformat(),
                    "system": "multi-agent-research"
                }
            })
            
            self._session_id = session_name
            logger.info(f"Started Phoenix evaluation session: {session_name}")
            return session_name
            
        except Exception as e:
            logger.error(f"Failed to start Phoenix session: {str(e)}")
            # Continue without Phoenix if it fails
            self._session_id = session_name
            return session_name
    
    async def start_trace(self, trace_name: str, metadata: Dict[str, Any] = None) -> str:
        """Start a new trace for an evaluation run."""
        trace_id = f"{trace_name}_{datetime.now().timestamp()}"
        
        try:
            await self._call_phoenix_mcp("start_trace", {
                "trace_id": trace_id,
                "trace_name": trace_name,
                "session_id": self._session_id,
                "metadata": metadata or {}
            })
            
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
        
        try:
            await self._call_phoenix_mcp("create_span", {
                "trace_id": trace_id,
                "span_id": span_id,
                "span_name": span_name,
                "span_type": span_type,
                "parent_span_id": parent_span_id,
                "start_time": datetime.now().isoformat(),
                "metadata": metadata or {}
            })
            
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
        try:
            await self._call_phoenix_mcp("end_span", {
                "trace_id": trace_id,
                "span_id": span_id,
                "end_time": datetime.now().isoformat(),
                "status": status,
                "result": result,
                "metrics": metrics or {},
                "error": error
            })
            
            logger.debug(f"Ended Phoenix span: {span_id}")
            
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
        span_id = await self.create_span(
            trace_id=trace_id,
            span_name=f"{agent_id}_interaction",
            span_type="llm",
            parent_span_id=parent_span_id,
            metadata={
                "agent_id": agent_id,
                "model": model_used,
                "input_length": len(input_message),
                "output_length": len(output_message)
            }
        )
        
        # Log the LLM call details
        try:
            await self._call_phoenix_mcp("log_llm_call", {
                "trace_id": trace_id,
                "span_id": span_id,
                "model": model_used,
                "messages": [
                    {"role": "user", "content": input_message},
                    {"role": "assistant", "content": output_message}
                ],
                "token_usage": tokens_used,
                "execution_time_ms": execution_time * 1000,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.warning(f"Failed to log LLM call to Phoenix: {str(e)}")
        
        await self.end_span(
            trace_id=trace_id,
            span_id=span_id,
            status="success",
            result=output_message,
            metrics={
                "tokens_total": tokens_used.get("total", 0),
                "tokens_prompt": tokens_used.get("prompt", 0),
                "tokens_completion": tokens_used.get("completion", 0),
                "execution_time_ms": execution_time * 1000
            }
        )
        
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
        try:
            await self._call_phoenix_mcp("log_evaluation", {
                "trace_id": trace_id,
                "evaluation_data": {
                    "query_id": query_id,
                    "query_text": query_text,
                    "expected_complexity": expected_complexity,
                    "actual_complexity": actual_complexity,
                    "response_length": len(response),
                    "citations_count": len(citations),
                    "execution_time_ms": execution_time * 1000,
                    "total_tokens": total_tokens,
                    "accuracy_score": accuracy_score,
                    "quality_metrics": quality_metrics or {},
                    "timestamp": datetime.now().isoformat()
                }
            })
            
            logger.info(f"Logged evaluation result for query {query_id}")
            
        except Exception as e:
            logger.warning(f"Failed to log evaluation result: {str(e)}")
    
    async def get_evaluation_metrics(self, session_id: str = None) -> Dict[str, Any]:
        """Retrieve evaluation metrics from Phoenix."""
        session_id = session_id or self._session_id
        
        try:
            response = await self._call_phoenix_mcp("get_session_metrics", {
                "session_id": session_id
            })
            
            return response if response else {}
            
        except Exception as e:
            logger.error(f"Failed to retrieve evaluation metrics: {str(e)}")
            return {}
    
    async def analyze_response_quality(self, 
                                     query: str,
                                     response: str,
                                     citations: List[Citation],
                                     expected_sources: int = None) -> Dict[str, float]:
        """Use Phoenix MCP to analyze response quality."""
        try:
            analysis_result = await self._call_phoenix_mcp("analyze_response_quality", {
                "query": query,
                "response": response,
                "citations": [{"url": c.url, "title": c.title, "snippet": c.content} for c in citations],
                "expected_sources": expected_sources,
                "criteria": [
                    "factual_accuracy",
                    "citation_completeness", 
                    "response_coherence",
                    "source_relevance"
                ]
            })
            
            return analysis_result.get("scores", {})
            
        except Exception as e:
            logger.error(f"Failed to analyze response quality: {str(e)}")
            return {
                "factual_accuracy": 0.0,
                "citation_completeness": 0.0,
                "response_coherence": 0.0,
                "source_relevance": 0.0
            }
    
    async def _call_phoenix_mcp(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Make a call to Phoenix MCP server using GPT-5 Responses API."""
        try:
            # Use GPT-5 Responses API with Phoenix MCP server
            response = self.client.responses.create(
                model="gpt-5",
                tools=[self.phoenix_mcp_config],
                input=f"Use the {tool_name} tool with these arguments: {json.dumps(arguments)}"
            )
            
            # Extract MCP call result from response
            for output_item in response.output:
                if output_item.type == "mcp_call" and output_item.name == tool_name:
                    if output_item.error:
                        raise Exception(f"Phoenix MCP error: {output_item.error}")
                    
                    # Parse JSON output if it's a string
                    output = output_item.output
                    if isinstance(output, str):
                        try:
                            output = json.loads(output)
                        except json.JSONDecodeError:
                            pass
                    
                    return output
            
            # If no MCP call was made, the tool might not exist
            logger.warning(f"Phoenix MCP tool '{tool_name}' was not called")
            return None
            
        except Exception as e:
            logger.error(f"Phoenix MCP call failed for {tool_name}: {str(e)}")
            raise
    
    async def close_session(self) -> Dict[str, Any]:
        """Close the current evaluation session and return final metrics."""
        if not self._session_id:
            return {}
        
        try:
            final_metrics = await self._call_phoenix_mcp("close_session", {
                "session_id": self._session_id,
                "end_time": datetime.now().isoformat()
            })
            
            logger.info(f"Closed Phoenix evaluation session: {self._session_id}")
            return final_metrics or {}
            
        except Exception as e:
            logger.error(f"Failed to close Phoenix session: {str(e)}")
            return {}
        finally:
            self._session_id = None
            self._current_conversation_id = None

# Global Phoenix integration instance
phoenix_integration = PhoenixMCPIntegration()

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