#!/usr/bin/env python3
"""
Evaluation runner with Phoenix observability.
Main entry point for running evaluations with comprehensive logging and monitoring.
"""
import asyncio
import logging
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.supervisor import SupervisorAgent
from config.settings import settings, TaskType, ReasoningEffort, Verbosity, ModelType
from evaluation.framework import EvaluationFramework, initialize_framework
from evaluation.test_suites import quality_test_suite
from evaluation.phoenix_integration import phoenix_integration
from evaluation_dataset import EVALUATION_QUERIES, get_queries_by_task_type

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EvaluationRunner:
    """Main evaluation runner with Phoenix integration."""
    
    def __init__(self,
                 supervisor_agent: Optional[SupervisorAgent] = None,
                 phoenix_enabled: bool = True,
                 output_dir: str = "evaluation_results"):
        
        self.supervisor_agent = supervisor_agent or self._create_default_supervisor()
        self.phoenix_enabled = phoenix_enabled
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize evaluation framework
        self.framework = initialize_framework(
            supervisor_agent=self.supervisor_agent,
            enable_phoenix=phoenix_enabled,
            parallel_execution=True,
            max_concurrent=3
        )
        
    def _create_default_supervisor(self) -> SupervisorAgent:
        """Create a default supervisor agent for evaluation."""
        supervisor = SupervisorAgent(
            reasoning_effort=ReasoningEffort.MEDIUM,
            verbosity=Verbosity.MEDIUM
        )
        
        # Note: In a full implementation, you would register sub-agents here
        # For now, we'll use the supervisor alone
        logger.info("Created default supervisor agent for evaluation")
        return supervisor
    
    async def run_single_query_evaluation(self, 
                                        query_id: int,
                                        save_results: bool = True,
                                        run_quality_tests: bool = True) -> Dict[str, Any]:
        """Run evaluation for a single query."""
        # Find the query
        query = next((q for q in EVALUATION_QUERIES if q.id == query_id), None)
        if not query:
            raise ValueError(f"Query with ID {query_id} not found")
        
        logger.info(f"Running evaluation for query {query_id}: {query.query[:100]}...")
        
        # Create evaluation session
        session = await self.framework.create_session(f"single_query_{query_id}")
        
        # Evaluate the query
        result = await self.framework.evaluate_single_query(query)
        
        # Run quality tests if requested
        quality_results = None
        if run_quality_tests and result.success:
            try:
                quality_results = await quality_test_suite.run_all_tests(
                    query=query.query,
                    response=result.response,
                    citations=result.citations,
                    expected_sources=query.expected_sources,
                    metadata={
                        "execution_time": result.execution_time,
                        "complexity": result.expected_complexity,
                        "total_tokens": result.total_tokens
                    },
                    trace_id=session.session_id
                )
                
                overall_quality_score = quality_test_suite.get_overall_score(quality_results)
                logger.info(f"Overall quality score: {overall_quality_score:.3f}")
                
            except Exception as e:
                logger.error(f"Quality tests failed: {e}")
                quality_results = {}
        
        # Prepare results
        evaluation_result = {
            "session_info": {
                "session_id": session.session_id,
                "timestamp": datetime.now().isoformat(),
                "phoenix_enabled": self.phoenix_enabled
            },
            "query_info": {
                "id": query.id,
                "text": query.query,
                "expected_complexity": query.complexity.value,
                "domain": query.domain,
                "expected_sources": query.expected_sources
            },
            "evaluation_result": {
                "success": result.success,
                "actual_complexity": result.actual_complexity.value,
                "response": result.response,
                "citations_count": len(result.citations),
                "execution_time": result.execution_time,
                "total_tokens": result.total_tokens,
                "model_used": result.model_used,
                "error": result.error
            },
            "quality_metrics": {
                "accuracy_score": result.accuracy_score,
                "citation_completeness": result.citation_completeness,
                "response_coherence": result.response_coherence,
                "source_relevance": result.source_relevance
            }
        }
        
        if quality_results:
            evaluation_result["quality_tests"] = {
                "overall_score": quality_test_suite.get_overall_score(quality_results),
                "summary": quality_test_suite.get_test_summary(quality_results),
                "individual_results": {
                    name: {
                        "score": test_result.score,
                        "passed": test_result.passed,
                        "details": test_result.details,
                        "error": test_result.error
                    }
                    for name, test_result in quality_results.items()
                }
            }
        
        # Save results if requested
        if save_results:
            filename = f"single_query_eval_{query_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(evaluation_result, f, indent=2, default=str)
            
            logger.info(f"Results saved to: {filepath}")
            evaluation_result["output_file"] = str(filepath)
        
        return evaluation_result
    
    async def run_complexity_evaluation(self,
                                      complexity: TaskType,
                                      max_queries: int = None,
                                      save_results: bool = True,
                                      run_quality_tests: bool = True) -> Dict[str, Any]:
        """Run evaluation for all queries of a specific complexity level."""
        logger.info(f"Running evaluation for complexity: {complexity.value}")
        
        # Create evaluation session  
        session_name = f"complexity_{complexity.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session = await self.framework.create_session(session_name)
        
        # Evaluate queries
        results = await self.framework.evaluate_complexity_level(complexity, max_queries)
        
        # Run quality tests on successful results
        quality_test_results = []
        if run_quality_tests:
            for result in results:
                if result.success:
                    try:
                        quality_results = await quality_test_suite.run_all_tests(
                            query=result.query_text,
                            response=result.response,
                            citations=result.citations,
                            expected_sources=None,  # Would need to look up from dataset
                            metadata={
                                "execution_time": result.execution_time,
                                "complexity": result.expected_complexity,
                                "total_tokens": result.total_tokens
                            },
                            trace_id=session.session_id
                        )
                        
                        quality_test_results.append({
                            "query_id": result.query_id,
                            "overall_score": quality_test_suite.get_overall_score(quality_results),
                            "results": quality_results
                        })
                        
                    except Exception as e:
                        logger.warning(f"Quality tests failed for query {result.query_id}: {e}")
        
        # Calculate summary metrics
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        summary_metrics = {
            "total_queries": len(results),
            "successful_queries": len(successful_results),
            "failed_queries": len(failed_results),
            "success_rate": len(successful_results) / len(results) if results else 0,
        }
        
        if successful_results:
            summary_metrics.update({
                "avg_execution_time": sum(r.execution_time for r in successful_results) / len(successful_results),
                "avg_tokens": sum(r.total_tokens for r in successful_results) / len(successful_results),
                "total_tokens": sum(r.total_tokens for r in successful_results)
            })
        
        if quality_test_results:
            overall_quality_scores = [q["overall_score"] for q in quality_test_results]
            summary_metrics["avg_quality_score"] = sum(overall_quality_scores) / len(overall_quality_scores)
        
        # Prepare final results
        evaluation_results = {
            "session_info": {
                "session_id": session.session_id,
                "session_name": session_name,
                "complexity": complexity.value,
                "timestamp": datetime.now().isoformat(),
                "phoenix_enabled": self.phoenix_enabled
            },
            "summary_metrics": summary_metrics,
            "individual_results": [
                {
                    "query_id": r.query_id,
                    "query_text": r.query_text[:200] + "..." if len(r.query_text) > 200 else r.query_text,
                    "success": r.success,
                    "execution_time": r.execution_time,
                    "total_tokens": r.total_tokens,
                    "model_used": r.model_used,
                    "error": r.error,
                    "quality_scores": {
                        "accuracy": r.accuracy_score,
                        "citation_completeness": r.citation_completeness,
                        "coherence": r.response_coherence,
                        "relevance": r.source_relevance
                    }
                }
                for r in results
            ]
        }
        
        if quality_test_results:
            evaluation_results["quality_test_summary"] = {
                "queries_tested": len(quality_test_results),
                "avg_overall_score": summary_metrics.get("avg_quality_score", 0),
                "detailed_results": quality_test_results
            }
        
        # Save results if requested
        if save_results:
            filename = f"complexity_eval_{complexity.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(evaluation_results, f, indent=2, default=str)
            
            logger.info(f"Results saved to: {filepath}")
            evaluation_results["output_file"] = str(filepath)
        
        return evaluation_results
    
    async def run_full_evaluation(self,
                                max_queries_per_complexity: int = None,
                                save_results: bool = True,
                                run_quality_tests: bool = True,
                                export_formats: List[str] = None) -> Dict[str, Any]:
        """Run complete evaluation across all complexity levels."""
        logger.info("Starting full dataset evaluation")
        
        if export_formats is None:
            export_formats = ["json"]
        
        # Create evaluation session
        session_name = f"full_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session = await self.framework.create_session(session_name)
        
        # Run full evaluation
        session = await self.framework.evaluate_full_dataset(max_queries_per_complexity)
        
        # Run quality tests on successful results
        if run_quality_tests:
            logger.info("Running quality tests on successful results...")
            
            for result in session.results:
                if result.success:
                    try:
                        quality_results = await quality_test_suite.run_all_tests(
                            query=result.query_text,
                            response=result.response,
                            citations=result.citations,
                            metadata={
                                "execution_time": result.execution_time,
                                "complexity": result.expected_complexity,
                                "total_tokens": result.total_tokens
                            },
                            trace_id=session.session_id
                        )
                        
                        # Update result with quality scores if not already set
                        if result.accuracy_score is None:
                            overall_score = quality_test_suite.get_overall_score(quality_results)
                            result.accuracy_score = overall_score
                        
                    except Exception as e:
                        logger.warning(f"Quality tests failed for query {result.query_id}: {e}")
        
        # Prepare comprehensive results
        evaluation_summary = {
            "session_info": {
                "session_id": session.session_id,
                "session_name": session.session_name,
                "start_time": session.start_time.isoformat(),
                "end_time": session.end_time.isoformat() if session.end_time else None,
                "phoenix_enabled": session.phoenix_enabled
            },
            "summary_metrics": session.summary_metrics,
            "detailed_results": {
                "total_queries": len(session.results),
                "successful_queries": len([r for r in session.results if r.success]),
                "failed_queries": len([r for r in session.results if not r.success]),
                "by_complexity": {}
            }
        }
        
        # Add breakdown by complexity
        for complexity in TaskType:
            complexity_results = [r for r in session.results if r.expected_complexity == complexity]
            complexity_successful = [r for r in complexity_results if r.success]
            
            evaluation_summary["detailed_results"]["by_complexity"][complexity.value] = {
                "total": len(complexity_results),
                "successful": len(complexity_successful),
                "success_rate": len(complexity_successful) / len(complexity_results) if complexity_results else 0,
                "avg_execution_time": sum(r.execution_time for r in complexity_successful) / len(complexity_successful) if complexity_successful else 0,
                "avg_tokens": sum(r.total_tokens for r in complexity_successful) / len(complexity_successful) if complexity_successful else 0
            }
        
        # Save results in requested formats
        if save_results:
            output_files = []
            
            for export_format in export_formats:
                try:
                    if export_format.lower() == "json":
                        filename = f"full_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        filepath = self.output_dir / filename
                        
                        # Export full session data
                        export_data = self.framework.export_results(session, "json")
                        with open(filepath, 'w') as f:
                            f.write(export_data)
                        
                        output_files.append(str(filepath))
                        
                    elif export_format.lower() == "csv":
                        filename = f"full_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        filepath = self.output_dir / filename
                        
                        export_data = self.framework.export_results(session, "csv")
                        with open(filepath, 'w') as f:
                            f.write(export_data)
                        
                        output_files.append(str(filepath))
                        
                    # Also save the summary
                    summary_filename = f"evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    summary_filepath = self.output_dir / summary_filename
                    
                    with open(summary_filepath, 'w') as f:
                        json.dump(evaluation_summary, f, indent=2, default=str)
                    
                    output_files.append(str(summary_filepath))
                    
                except Exception as e:
                    logger.error(f"Failed to export in format {export_format}: {e}")
            
            evaluation_summary["output_files"] = output_files
            logger.info(f"Results saved to: {', '.join(output_files)}")
        
        return evaluation_summary

async def main():
    """Main entry point for evaluation runner."""
    parser = argparse.ArgumentParser(description="Run multi-agent research system evaluation")
    parser.add_argument("--query-id", type=int, help="Evaluate a single query by ID")
    parser.add_argument("--complexity", choices=["simple", "moderate", "complex"], 
                       help="Evaluate queries of specific complexity")
    parser.add_argument("--full", action="store_true", help="Run full dataset evaluation")
    parser.add_argument("--max-queries", type=int, help="Maximum queries per complexity level")
    parser.add_argument("--output-dir", default="evaluation_results", help="Output directory")
    parser.add_argument("--no-phoenix", action="store_true", help="Disable Phoenix integration")
    parser.add_argument("--no-quality-tests", action="store_true", help="Skip quality tests")
    parser.add_argument("--export-formats", nargs="+", default=["json"], 
                       choices=["json", "csv"], help="Export formats")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = EvaluationRunner(
        phoenix_enabled=not args.no_phoenix,
        output_dir=args.output_dir
    )
    
    try:
        if args.query_id:
            # Single query evaluation
            result = await runner.run_single_query_evaluation(
                query_id=args.query_id,
                save_results=True,
                run_quality_tests=not args.no_quality_tests
            )
            print(f"Single query evaluation completed. Success: {result['evaluation_result']['success']}")
            
        elif args.complexity:
            # Complexity-specific evaluation
            complexity_map = {
                "simple": TaskType.DIRECT_ANSWER,
                "moderate": TaskType.SEARCH_NEEDED,
                "complex": TaskType.RESEARCH_REPORT
            }
            
            result = await runner.run_complexity_evaluation(
                complexity=complexity_map[args.complexity],
                max_queries=args.max_queries,
                save_results=True,
                run_quality_tests=not args.no_quality_tests
            )
            print(f"Complexity evaluation completed. Success rate: {result['summary_metrics']['success_rate']:.2%}")
            
        elif args.full:
            # Full dataset evaluation
            result = await runner.run_full_evaluation(
                max_queries_per_complexity=args.max_queries,
                save_results=True,
                run_quality_tests=not args.no_quality_tests,
                export_formats=args.export_formats
            )
            print(f"Full evaluation completed. Overall success rate: {result['summary_metrics']['success_rate']:.2%}")
            
        else:
            print("Please specify --query-id, --complexity, or --full")
            return
            
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())