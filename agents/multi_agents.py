#!/usr/bin/env python3
"""
Multi-Agent Research System Integration
Provides easy setup and orchestration of the complete agent system.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

from agents.supervisor import SupervisorAgent
from agents.search import SearchAgent
from agents.citation import CitationAgent
from config.settings import settings, ReasoningEffort, Verbosity

logger = logging.getLogger(__name__)

class MultiAgentResearchSystem:
    """Complete multi-agent research system orchestrator."""
    
    def __init__(self, 
                 supervisor_reasoning: ReasoningEffort = ReasoningEffort.MEDIUM,
                 supervisor_verbosity: Verbosity = Verbosity.MEDIUM,
                 search_reasoning: ReasoningEffort = ReasoningEffort.LOW,
                 citation_reasoning: ReasoningEffort = ReasoningEffort.LOW,
                 enable_phoenix_tracing: bool = True):
        """Initialize the complete multi-agent system."""
        
        self.enable_phoenix_tracing = enable_phoenix_tracing
        
        # Initialize agents
        self.supervisor = SupervisorAgent(
            reasoning_effort=supervisor_reasoning,
            verbosity=supervisor_verbosity
        )
        
        self.search_agent = SearchAgent(
            reasoning_effort=search_reasoning,
            verbosity=Verbosity.MEDIUM  # Search results need good detail
        )
        
        self.citation_agent = CitationAgent(
            reasoning_effort=citation_reasoning,
            verbosity=Verbosity.LOW  # Citations are more mechanical
        )
        
        # Register agents with supervisor
        self.supervisor.register_agent(self.search_agent)
        self.supervisor.register_agent(self.citation_agent)
        
        # System state
        self.is_initialized = True
        self.active_sessions = {}
        
        logger.info("Multi-Agent Research System initialized successfully")
        logger.info(f"Agents registered: {list(self.supervisor.sub_agents.keys())}")
    
    async def process_query(self, query: str, trace_id: str = None, 
                          session_id: str = None) -> Dict[str, Any]:
        """Process a research query through the complete agent system."""
        
        if not self.is_initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        # Generate IDs if not provided
        if trace_id is None:
            trace_id = str(uuid4())
        
        if session_id is None:
            session_id = str(uuid4())
        
        logger.info(f"Processing query in session {session_id}: {query[:100]}...")
        
        try:
            # Start session tracking
            self.active_sessions[session_id] = {
                'query': query,
                'trace_id': trace_id,
                'start_time': asyncio.get_event_loop().time(),
                'status': 'processing'
            }
            
            # Process through supervisor
            result = await self.supervisor.orchestrate(query, trace_id)
            
            # Update session
            self.active_sessions[session_id]['status'] = 'completed'
            self.active_sessions[session_id]['result'] = result
            
            # Enhance result with system metadata
            enhanced_result = {
                **result,
                'session_id': session_id,
                'trace_id': trace_id,
                'system_version': '1.0.0',
                'agents_used': list(self.supervisor.sub_agents.keys())
            }
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Query processing failed for session {session_id}: {str(e)}")
            
            # Update session with error
            self.active_sessions[session_id]['status'] = 'failed'
            self.active_sessions[session_id]['error'] = str(e)
            
            return {
                'status': 'error',
                'session_id': session_id,
                'trace_id': trace_id,
                'error': str(e),
                'query': query
            }
    
    async def batch_process_queries(self, queries: List[str], 
                                  max_concurrent: int = 3) -> List[Dict[str, Any]]:
        """Process multiple queries concurrently."""
        
        logger.info(f"Starting batch processing of {len(queries)} queries")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def bounded_process(query):
            async with semaphore:
                return await self.process_query(query)
        
        # Process all queries
        tasks = [bounded_process(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'status': 'error',
                    'query': queries[i],
                    'error': str(result)
                })
            else:
                processed_results.append(result)
        
        logger.info(f"Batch processing completed. {len(processed_results)} results returned")
        return processed_results
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        
        # Get stats from each agent
        supervisor_stats = self.supervisor.get_stats()
        search_stats = self.search_agent.get_search_stats()
        citation_stats = self.citation_agent.get_citation_stats()
        
        # Session statistics
        active_count = len([s for s in self.active_sessions.values() if s['status'] == 'processing'])
        completed_count = len([s for s in self.active_sessions.values() if s['status'] == 'completed'])
        failed_count = len([s for s in self.active_sessions.values() if s['status'] == 'failed'])
        
        return {
            'system_info': {
                'version': '1.0.0',
                'is_initialized': self.is_initialized,
                'agents_count': len(self.supervisor.sub_agents),
                'phoenix_tracing': self.enable_phoenix_tracing
            },
            'session_stats': {
                'total_sessions': len(self.active_sessions),
                'active_sessions': active_count,
                'completed_sessions': completed_count,
                'failed_sessions': failed_count,
                'success_rate': completed_count / len(self.active_sessions) if self.active_sessions else 0
            },
            'agent_performance': {
                'supervisor': supervisor_stats,
                'search': search_stats,
                'citation': citation_stats
            }
        }
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific session."""
        return self.active_sessions.get(session_id)
    
    async def cleanup_completed_sessions(self, max_age_hours: int = 24):
        """Clean up old completed sessions to prevent memory bloat."""
        import time
        
        current_time = time.time()
        cutoff_time = current_time - (max_age_hours * 3600)
        
        sessions_to_remove = []
        for session_id, session_data in self.active_sessions.items():
            if session_data.get('status') in ['completed', 'failed']:
                if session_data.get('start_time', current_time) < cutoff_time:
                    sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.active_sessions[session_id]
        
        logger.info(f"Cleaned up {len(sessions_to_remove)} old sessions")
    
    async def shutdown(self):
        """Gracefully shutdown the system."""
        logger.info("Shutting down Multi-Agent Research System...")
        
        # Wait for active sessions to complete (with timeout)
        active_sessions = [s for s in self.active_sessions.values() if s['status'] == 'processing']
        if active_sessions:
            logger.info(f"Waiting for {len(active_sessions)} active sessions to complete...")
            
            # Wait up to 30 seconds for completion
            wait_time = 0
            while active_sessions and wait_time < 30:
                await asyncio.sleep(1)
                wait_time += 1
                active_sessions = [s for s in self.active_sessions.values() if s['status'] == 'processing']
        
        # Clear resources
        self.active_sessions.clear()
        self.is_initialized = False
        
        logger.info("Multi-Agent Research System shutdown complete")


# Convenience functions for easy system usage
_global_system: Optional[MultiAgentResearchSystem] = None

def initialize_system(**kwargs) -> MultiAgentResearchSystem:
    """Initialize the global research system instance."""
    global _global_system
    
    if _global_system is not None:
        logger.warning("System already initialized. Returning existing instance.")
        return _global_system
    
    _global_system = MultiAgentResearchSystem(**kwargs)
    return _global_system

def get_system() -> Optional[MultiAgentResearchSystem]:
    """Get the global system instance."""
    return _global_system

async def research_query(query: str, **kwargs) -> Dict[str, Any]:
    """Convenience function to process a single query."""
    system = get_system()
    if system is None:
        system = initialize_system()
    
    return await system.process_query(query, **kwargs)

async def research_batch(queries: List[str], **kwargs) -> List[Dict[str, Any]]:
    """Convenience function to process multiple queries."""
    system = get_system()
    if system is None:
        system = initialize_system()
    
    return await system.batch_process_queries(queries, **kwargs)

def get_stats() -> Dict[str, Any]:
    """Get system statistics."""
    system = get_system()
    if system is None:
        return {"error": "System not initialized"}
    
    return system.get_system_stats()

# Demo/Testing functions
async def run_system_demo():
    """Run a simple demo of the system."""
    print("🚀 Multi-Agent Research System Demo")
    print("=" * 50)
    
    # Initialize system
    system = initialize_system()
    
    # Test queries of different complexities
    demo_queries = [
        "What is the speed of light?",  # Simple
        "How does photosynthesis convert light energy into chemical energy?",  # Moderate
        "What are the latest breakthroughs in quantum error correction and their impact on quantum computing?",  # Complex
    ]
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n📝 Demo Query {i}: {query}")
        print("-" * 40)
        
        try:
            result = await system.process_query(query)
            
            print(f"✅ Status: {result.get('status', 'unknown')}")
            print(f"🔍 Response: {result.get('response', 'No response')[:200]}...")
            print(f"📚 Citations: {len(result.get('citations', []))}")
            print(f"⏱️ Time: {result.get('execution_time', 0):.2f}s")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    # Show system stats
    print(f"\n📊 System Statistics:")
    stats = system.get_system_stats()
    print(f"Sessions: {stats['session_stats']['total_sessions']}")
    print(f"Success Rate: {stats['session_stats']['success_rate']:.2%}")
    
    print("\n🎉 Demo completed!")

if __name__ == "__main__":
    # Run demo if script is executed directly
    asyncio.run(run_system_demo())