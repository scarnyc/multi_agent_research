#!/usr/bin/env python3
"""
Automated quality test suites with Phoenix metrics integration.
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

from agents.models import Citation
from evaluation.phoenix_integration import phoenix_integration
from config.settings import ComplexityLevel

logger = logging.getLogger(__name__)

@dataclass 
class TestResult:
    """Result of a quality test."""
    test_name: str
    score: float  # 0.0 to 1.0
    passed: bool
    details: Dict[str, Any] = None
    error: Optional[str] = None

class QualityTest(ABC):
    """Base class for quality tests."""
    
    @abstractmethod
    async def run(self, query: str, response: str, citations: List[Citation], 
                  expected_sources: int = None, metadata: Dict[str, Any] = None) -> TestResult:
        """Run the quality test and return a result."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the test."""
        pass
    
    @property
    def threshold(self) -> float:
        """Passing threshold for this test (0.0 to 1.0)."""
        return 0.7  # Default threshold

class FactualAccuracyTest(QualityTest):
    """Test for factual accuracy using Phoenix MCP analysis."""
    
    @property
    def name(self) -> str:
        return "factual_accuracy"
    
    @property
    def threshold(self) -> float:
        return 0.8  # Higher threshold for accuracy
    
    async def run(self, query: str, response: str, citations: List[Citation], 
                  expected_sources: int = None, metadata: Dict[str, Any] = None) -> TestResult:
        """Run factual accuracy test."""
        try:
            # Use Phoenix MCP to analyze factual accuracy
            quality_scores = await phoenix_integration.analyze_response_quality(
                query=query,
                response=response,
                citations=citations,
                expected_sources=expected_sources
            )
            
            accuracy_score = quality_scores.get("factual_accuracy", 0.0)
            passed = accuracy_score >= self.threshold
            
            return TestResult(
                test_name=self.name,
                score=accuracy_score,
                passed=passed,
                details={
                    "threshold": self.threshold,
                    "raw_score": accuracy_score,
                    "query_length": len(query),
                    "response_length": len(response),
                    "citations_count": len(citations)
                }
            )
            
        except Exception as e:
            logger.error(f"Factual accuracy test failed: {str(e)}")
            return TestResult(
                test_name=self.name,
                score=0.0,
                passed=False,
                error=str(e)
            )

class CitationCompletenessTest(QualityTest):
    """Test for citation completeness."""
    
    @property
    def name(self) -> str:
        return "citation_completeness"
    
    @property
    def threshold(self) -> float:
        return 0.9  # High threshold for citations
    
    async def run(self, query: str, response: str, citations: List[Citation], 
                  expected_sources: int = None, metadata: Dict[str, Any] = None) -> TestResult:
        """Run citation completeness test."""
        try:
            # Use Phoenix MCP for detailed analysis
            quality_scores = await phoenix_integration.analyze_response_quality(
                query=query,
                response=response,
                citations=citations,
                expected_sources=expected_sources
            )
            
            citation_score = quality_scores.get("citation_completeness", 0.0)
            passed = citation_score >= self.threshold
            
            # Additional local checks
            has_citations = len(citations) > 0
            meets_expected_sources = expected_sources is None or len(citations) >= expected_sources
            
            # Combine scores
            local_score = 1.0 if (has_citations and meets_expected_sources) else 0.0
            final_score = (citation_score + local_score) / 2.0
            
            return TestResult(
                test_name=self.name,
                score=final_score,
                passed=final_score >= self.threshold,
                details={
                    "threshold": self.threshold,
                    "phoenix_score": citation_score,
                    "local_score": local_score,
                    "final_score": final_score,
                    "has_citations": has_citations,
                    "citations_count": len(citations),
                    "expected_sources": expected_sources,
                    "meets_expected": meets_expected_sources
                }
            )
            
        except Exception as e:
            logger.error(f"Citation completeness test failed: {str(e)}")
            return TestResult(
                test_name=self.name,
                score=0.0,
                passed=False,
                error=str(e)
            )

class ResponseCoherenceTest(QualityTest):
    """Test for response coherence and structure."""
    
    @property
    def name(self) -> str:
        return "response_coherence"
    
    async def run(self, query: str, response: str, citations: List[Citation], 
                  expected_sources: int = None, metadata: Dict[str, Any] = None) -> TestResult:
        """Run response coherence test."""
        try:
            # Use Phoenix MCP for AI-based coherence analysis
            quality_scores = await phoenix_integration.analyze_response_quality(
                query=query,
                response=response,
                citations=citations,
                expected_sources=expected_sources
            )
            
            coherence_score = quality_scores.get("response_coherence", 0.0)
            
            # Additional local coherence checks
            local_checks = self._local_coherence_checks(response)
            local_score = sum(local_checks.values()) / len(local_checks)
            
            # Combine scores
            final_score = (coherence_score + local_score) / 2.0
            passed = final_score >= self.threshold
            
            return TestResult(
                test_name=self.name,
                score=final_score,
                passed=passed,
                details={
                    "threshold": self.threshold,
                    "phoenix_score": coherence_score,
                    "local_score": local_score,
                    "final_score": final_score,
                    "local_checks": local_checks,
                    "response_length": len(response),
                    "word_count": len(response.split())
                }
            )
            
        except Exception as e:
            logger.error(f"Response coherence test failed: {str(e)}")
            return TestResult(
                test_name=self.name,
                score=0.0,
                passed=False,
                error=str(e)
            )
    
    def _local_coherence_checks(self, response: str) -> Dict[str, float]:
        """Perform local coherence checks."""
        checks = {}
        
        # Check for minimum length
        checks["min_length"] = 1.0 if len(response.strip()) >= 50 else 0.0
        
        # Check for sentence structure
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        checks["has_sentences"] = 1.0 if len(sentences) >= 2 else 0.5 if len(sentences) == 1 else 0.0
        
        # Check for paragraph structure
        paragraphs = [p.strip() for p in response.split('\n\n') if p.strip()]
        checks["has_paragraphs"] = 1.0 if len(paragraphs) >= 2 else 0.5 if len(paragraphs) == 1 else 0.0
        
        # Check for proper capitalization
        checks["proper_capitalization"] = 1.0 if response and response[0].isupper() else 0.0
        
        # Check for balanced punctuation
        open_parens = response.count('(')
        close_parens = response.count(')')
        checks["balanced_parentheses"] = 1.0 if open_parens == close_parens else 0.0
        
        return checks

class SourceRelevanceTest(QualityTest):
    """Test for source relevance to the query."""
    
    @property
    def name(self) -> str:
        return "source_relevance"
    
    async def run(self, query: str, response: str, citations: List[Citation], 
                  expected_sources: int = None, metadata: Dict[str, Any] = None) -> TestResult:
        """Run source relevance test."""
        try:
            if not citations:
                return TestResult(
                    test_name=self.name,
                    score=0.0,
                    passed=False,
                    details={"no_citations": True}
                )
            
            # Use Phoenix MCP for relevance analysis
            quality_scores = await phoenix_integration.analyze_response_quality(
                query=query,
                response=response,
                citations=citations,
                expected_sources=expected_sources
            )
            
            relevance_score = quality_scores.get("source_relevance", 0.0)
            passed = relevance_score >= self.threshold
            
            # Additional local checks
            unique_domains = set()
            for citation in citations:
                try:
                    from urllib.parse import urlparse
                    domain = urlparse(citation.url).netloc
                    unique_domains.add(domain)
                except:
                    pass
            
            diversity_score = min(1.0, len(unique_domains) / max(1, len(citations) // 2))
            
            return TestResult(
                test_name=self.name,
                score=relevance_score,
                passed=passed,
                details={
                    "threshold": self.threshold,
                    "relevance_score": relevance_score,
                    "diversity_score": diversity_score,
                    "unique_domains": len(unique_domains),
                    "total_citations": len(citations),
                    "citation_titles": [c.title for c in citations[:3]]  # Sample titles
                }
            )
            
        except Exception as e:
            logger.error(f"Source relevance test failed: {str(e)}")
            return TestResult(
                test_name=self.name,
                score=0.0,
                passed=False,
                error=str(e)
            )

class LatencyTest(QualityTest):
    """Test for response latency based on complexity."""
    
    @property
    def name(self) -> str:
        return "latency"
    
    def get_complexity_threshold(self, complexity: ComplexityLevel) -> float:
        """Get latency threshold based on complexity."""
        thresholds = {
            ComplexityLevel.SIMPLE: 3.0,    # 3 seconds
            ComplexityLevel.MODERATE: 10.0,  # 10 seconds  
            ComplexityLevel.COMPLEX: 20.0    # 20 seconds
        }
        return thresholds.get(complexity, 15.0)
    
    async def run(self, query: str, response: str, citations: List[Citation], 
                  expected_sources: int = None, metadata: Dict[str, Any] = None) -> TestResult:
        """Run latency test."""
        try:
            execution_time = metadata.get("execution_time", 0.0) if metadata else 0.0
            complexity = metadata.get("complexity", ComplexityLevel.MODERATE) if metadata else ComplexityLevel.MODERATE
            
            threshold = self.get_complexity_threshold(complexity)
            
            # Score based on how much under the threshold we are
            if execution_time <= threshold:
                score = 1.0 - (execution_time / threshold) * 0.5  # Scale from 0.5 to 1.0
            else:
                score = max(0.0, 1.0 - ((execution_time - threshold) / threshold))
            
            passed = execution_time <= threshold
            
            return TestResult(
                test_name=self.name,
                score=score,
                passed=passed,
                details={
                    "execution_time": execution_time,
                    "threshold": threshold,
                    "complexity": complexity.value,
                    "performance_ratio": execution_time / threshold if threshold > 0 else 0
                }
            )
            
        except Exception as e:
            logger.error(f"Latency test failed: {str(e)}")
            return TestResult(
                test_name=self.name,
                score=0.0,
                passed=False,
                error=str(e)
            )

class TokenEfficiencyTest(QualityTest):
    """Test for token efficiency (response quality vs tokens used)."""
    
    @property
    def name(self) -> str:
        return "token_efficiency" 
    
    @property
    def threshold(self) -> float:
        return 0.6  # Moderate threshold for efficiency
    
    async def run(self, query: str, response: str, citations: List[Citation], 
                  expected_sources: int = None, metadata: Dict[str, Any] = None) -> TestResult:
        """Run token efficiency test."""
        try:
            total_tokens = metadata.get("total_tokens", 0) if metadata else 0
            complexity = metadata.get("complexity", ComplexityLevel.MODERATE) if metadata else ComplexityLevel.MODERATE
            
            if total_tokens == 0:
                return TestResult(
                    test_name=self.name,
                    score=0.0,
                    passed=False,
                    details={"no_token_data": True}
                )
            
            # Define token budgets by complexity
            token_budgets = {
                ComplexityLevel.SIMPLE: 500,
                ComplexityLevel.MODERATE: 2000,
                ComplexityLevel.COMPLEX: 5000
            }
            
            budget = token_budgets.get(complexity, 2000)
            
            # Calculate efficiency score
            if total_tokens <= budget:
                # Bonus for being under budget
                efficiency_score = 1.0 - (total_tokens / budget) * 0.3  # Scale from 0.7 to 1.0
            else:
                # Penalty for exceeding budget
                efficiency_score = max(0.0, 1.0 - ((total_tokens - budget) / budget))
            
            # Factor in response quality (length as a proxy)
            response_quality = min(1.0, len(response) / 200)  # Normalized response length
            
            final_score = (efficiency_score + response_quality) / 2.0
            passed = final_score >= self.threshold
            
            return TestResult(
                test_name=self.name,
                score=final_score,
                passed=passed,
                details={
                    "total_tokens": total_tokens,
                    "token_budget": budget,
                    "efficiency_score": efficiency_score,
                    "response_quality": response_quality,
                    "final_score": final_score,
                    "threshold": self.threshold,
                    "complexity": complexity.value,
                    "tokens_per_char": total_tokens / max(1, len(response))
                }
            )
            
        except Exception as e:
            logger.error(f"Token efficiency test failed: {str(e)}")
            return TestResult(
                test_name=self.name,
                score=0.0,
                passed=False,
                error=str(e)
            )

class QualityTestSuite:
    """Suite of quality tests with Phoenix integration."""
    
    def __init__(self, enable_phoenix_logging: bool = True):
        self.enable_phoenix_logging = enable_phoenix_logging
        self.tests = [
            FactualAccuracyTest(),
            CitationCompletenessTest(),
            ResponseCoherenceTest(),
            SourceRelevanceTest(),
            LatencyTest(),
            TokenEfficiencyTest()
        ]
    
    async def run_all_tests(self, 
                           query: str,
                           response: str, 
                           citations: List[Citation],
                           expected_sources: int = None,
                           metadata: Dict[str, Any] = None,
                           trace_id: str = None) -> Dict[str, TestResult]:
        """Run all quality tests and return results."""
        results = {}
        
        for test in self.tests:
            try:
                logger.debug(f"Running test: {test.name}")
                result = await test.run(query, response, citations, expected_sources, metadata)
                results[test.name] = result
                
                # Log to Phoenix if enabled
                if self.enable_phoenix_logging and trace_id:
                    try:
                        await self._log_test_result_to_phoenix(trace_id, test.name, result)
                    except Exception as e:
                        logger.warning(f"Failed to log test result to Phoenix: {e}")
                        
            except Exception as e:
                logger.error(f"Test {test.name} failed: {str(e)}")
                results[test.name] = TestResult(
                    test_name=test.name,
                    score=0.0,
                    passed=False,
                    error=str(e)
                )
        
        return results
    
    async def _log_test_result_to_phoenix(self, trace_id: str, test_name: str, result: TestResult) -> None:
        """Log test result to Phoenix."""
        try:
            # Use Phoenix MCP to log test metrics
            await phoenix_integration._call_phoenix_mcp("log_test_result", {
                "trace_id": trace_id,
                "test_name": test_name,
                "score": result.score,
                "passed": result.passed,
                "details": result.details,
                "error": result.error,
                "timestamp": str(asyncio.get_event_loop().time())
            })
        except Exception as e:
            logger.warning(f"Failed to log test result to Phoenix: {e}")
    
    def get_overall_score(self, test_results: Dict[str, TestResult]) -> float:
        """Calculate overall quality score from test results."""
        if not test_results:
            return 0.0
        
        # Weight different tests
        weights = {
            "factual_accuracy": 0.25,
            "citation_completeness": 0.20,
            "response_coherence": 0.20,
            "source_relevance": 0.15,
            "latency": 0.10,
            "token_efficiency": 0.10
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for test_name, result in test_results.items():
            weight = weights.get(test_name, 0.1)  # Default weight
            weighted_score += result.score * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def get_test_summary(self, test_results: Dict[str, TestResult]) -> Dict[str, Any]:
        """Get summary statistics for test results."""
        if not test_results:
            return {}
        
        passed_tests = [r for r in test_results.values() if r.passed]
        failed_tests = [r for r in test_results.values() if not r.passed]
        
        return {
            "total_tests": len(test_results),
            "passed_tests": len(passed_tests),
            "failed_tests": len(failed_tests),
            "pass_rate": len(passed_tests) / len(test_results),
            "overall_score": self.get_overall_score(test_results),
            "individual_scores": {name: result.score for name, result in test_results.items()},
            "failed_test_names": [r.test_name for r in failed_tests]
        }

# Global test suite instance
quality_test_suite = QualityTestSuite()

async def run_quality_tests(query: str, response: str, citations: List[Citation],
                           expected_sources: int = None, metadata: Dict[str, Any] = None,
                           trace_id: str = None) -> Dict[str, TestResult]:
    """Convenience function to run all quality tests."""
    return await quality_test_suite.run_all_tests(
        query, response, citations, expected_sources, metadata, trace_id
    )