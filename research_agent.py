#!/usr/bin/env python3
"""
Multi-agent research system with intelligent model routing
"""
import time
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from openai import OpenAI
from config import config, ComplexityLevel
from pydantic import BaseModel

@dataclass
class ResearchResult:
    """Result from a research query"""
    query: str
    response: str
    sources: List[str]
    model_used: str
    complexity_detected: ComplexityLevel
    execution_time: float
    token_usage: Dict[str, int]
    citations: List[str]

class ComplexityAnalyzer:
    """Analyzes query complexity for model routing"""
    
    def __init__(self):
        # Keywords that indicate complexity levels
        self.complex_keywords = [
            'analyze', 'compare', 'evaluate', 'examine', 'implications', 
            'trends', 'impact', 'relationship', 'effectiveness', 'strategies'
        ]
        
        self.moderate_keywords = [
            'differences', 'how', 'why', 'pros and cons', 'advantages',
            'disadvantages', 'benefits', 'explain', 'factors'
        ]
    
    def analyze_complexity(self, query: str) -> ComplexityLevel:
        """Determine complexity level of a research query"""
        query_lower = query.lower()
        word_count = len(query.split())
        
        # Check for complex analysis keywords
        complex_matches = sum(1 for keyword in self.complex_keywords if keyword in query_lower)
        moderate_matches = sum(1 for keyword in self.moderate_keywords if keyword in query_lower)
        
        # Rules for complexity classification
        if complex_matches >= 2 or word_count > 15:
            return ComplexityLevel.COMPLEX
        elif complex_matches >= 1 or moderate_matches >= 2 or word_count > 8:
            return ComplexityLevel.MODERATE
        else:
            return ComplexityLevel.SIMPLE

class WebSearchAgent:
    """Agent that performs web searches using GPT-5"""
    
    def __init__(self, client: OpenAI):
        self.client = client
    
    def search_and_synthesize(self, query: str, model: str) -> Dict[str, Any]:
        """Perform web search and synthesize results"""
        
        # Create a research-focused prompt that encourages web search
        system_prompt = """You are a research assistant with access to current web information. 
        For the user's query:
        1. Search for current, relevant information
        2. Provide a comprehensive answer based on your findings
        3. Include specific URLs and sources in your response
        4. Format sources as [Source: URL] after relevant information
        5. Ensure all claims are properly attributed to sources"""
        
        try:
            start_time = time.time()
            
            # GPT-5 models have different parameter requirements
            chat_params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                "max_completion_tokens": 1000
            }
            
            # GPT-5 models don't support temperature parameter - use default
            
            response = self.client.chat.completions.create(**chat_params)
            
            execution_time = time.time() - start_time
            content = response.choices[0].message.content
            
            # Extract sources from the response
            sources = self._extract_sources(content)
            citations = self._extract_citations(content)
            
            return {
                'content': content,
                'sources': sources,
                'citations': citations,
                'execution_time': execution_time,
                'token_usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            }
            
        except Exception as e:
            return {
                'content': f"Error performing research: {str(e)}",
                'sources': [],
                'citations': [],
                'execution_time': 0,
                'token_usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
            }
    
    def _extract_sources(self, content: str) -> List[str]:
        """Extract URLs from response content"""
        # Look for URLs in various formats
        url_patterns = [
            r'https?://[^\s\)]+',
            r'\[Source: (https?://[^\]]+)\]',
            r'\(https?://[^\)]+\)'
        ]
        
        sources = []
        for pattern in url_patterns:
            matches = re.findall(pattern, content)
            sources.extend(matches)
        
        return list(set(sources))  # Remove duplicates
    
    def _extract_citations(self, content: str) -> List[str]:
        """Extract citation text from response"""
        # Look for citation patterns like [Source: ...] or (Source: ...)
        citation_patterns = [
            r'\[Source: ([^\]]+)\]',
            r'\(Source: ([^\)]+)\)',
            r'Source: ([^\n]+)'
        ]
        
        citations = []
        for pattern in citation_patterns:
            matches = re.findall(pattern, content)
            citations.extend(matches)
        
        return citations

class ResearchAgent:
    """Main research agent with model routing and web search"""
    
    def __init__(self):
        self.client = OpenAI(api_key=config.openai_api_key)
        self.complexity_analyzer = ComplexityAnalyzer()
        self.search_agent = WebSearchAgent(self.client)
    
    def research(self, query: str, complexity_override: Optional[ComplexityLevel] = None) -> ResearchResult:
        """Perform research with intelligent model routing"""
        
        # Determine complexity and select model
        complexity = complexity_override or self.complexity_analyzer.analyze_complexity(query)
        model = config.model_for_complexity[complexity]
        
        print(f"Query: {query}")
        print(f"Detected complexity: {complexity.value} → Using model: {model}")
        
        # Perform research using the selected model
        search_results = self.search_agent.search_and_synthesize(query, model)
        
        return ResearchResult(
            query=query,
            response=search_results['content'],
            sources=search_results['sources'],
            model_used=model,
            complexity_detected=complexity,
            execution_time=search_results['execution_time'],
            token_usage=search_results['token_usage'],
            citations=search_results['citations']
        )
    
    def batch_research(self, queries: List[str]) -> List[ResearchResult]:
        """Perform research on multiple queries"""
        results = []
        total_start = time.time()
        
        for i, query in enumerate(queries, 1):
            print(f"\n[{i}/{len(queries)}] Processing query...")
            result = self.research(query)
            results.append(result)
            print(f"Completed in {result.execution_time:.2f}s using {result.token_usage['total_tokens']} tokens")
        
        total_time = time.time() - total_start
        print(f"\nBatch completed in {total_time:.2f}s")
        return results

def test_research_agent():
    """Test the research agent with sample queries"""
    agent = ResearchAgent()
    
    # Test queries of different complexity
    test_queries = [
        "What is Python?",  # Simple
        "Compare React vs Vue.js performance",  # Moderate  
        "Analyze the economic impact of AI on healthcare"  # Complex
    ]
    
    print("Testing Research Agent")
    print("=" * 50)
    
    for query in test_queries:
        result = agent.research(query)
        print(f"\nQuery: {query}")
        print(f"Model: {result.model_used}")
        print(f"Time: {result.execution_time:.2f}s")
        print(f"Tokens: {result.token_usage['total_tokens']}")
        print(f"Sources found: {len(result.sources)}")
        print(f"Response preview: {result.response[:200]}...")
        print("-" * 30)

if __name__ == "__main__":
    test_research_agent()