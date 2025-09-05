#!/usr/bin/env python3
"""
Multi-agent research system with intelligent model routing
"""
import time
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from openai import OpenAI
from config.settings import settings, TaskType
from pydantic import BaseModel

@dataclass
class ResearchResult:
    """Result from a research query"""
    query: str
    response: str
    sources: List[str]
    model_used: str
    task_type_detected: TaskType
    execution_time: float
    token_usage: Dict[str, int]
    citations: List[str]

class TaskTypeAnalyzer:
    """Analyzes query to determine what type of response is needed"""
    
    def __init__(self):
        # Keywords that indicate different task types
        self.research_report_keywords = [
            'analyze', 'compare', 'evaluate', 'examine', 'implications', 
            'trends', 'impact', 'relationship', 'effectiveness', 'strategies',
            'comprehensive', 'in-depth', 'detailed analysis'
        ]
        
        self.search_needed_keywords = [
            'latest', 'recent', 'current', 'news', 'updates', 'new',
            '2024', '2025', 'now', 'today', 'this year', 'breaking'
        ]
        
        self.direct_answer_keywords = [
            'what is', 'define', 'definition', 'meaning', 'explain simply',
            'calculate', 'formula', 'basic', 'simple explanation'
        ]
    
    def analyze_task_type(self, query: str) -> TaskType:
        """Determine what type of response the user needs"""
        query_lower = query.lower()
        word_count = len(query.split())
        
        # Check for different patterns
        research_matches = sum(1 for keyword in self.research_report_keywords if keyword in query_lower)
        search_matches = sum(1 for keyword in self.search_needed_keywords if keyword in query_lower)
        direct_matches = sum(1 for keyword in self.direct_answer_keywords if keyword in query_lower)
        
        # Rules for task type classification
        if research_matches >= 1 or word_count > 20:
            return TaskType.RESEARCH_REPORT
        elif search_matches >= 1:
            return TaskType.SEARCH_NEEDED
        elif direct_matches >= 1 or (word_count <= 8 and '?' in query):
            return TaskType.DIRECT_ANSWER
        else:
            # Default to search for unclear cases
            return TaskType.SEARCH_NEEDED

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
            
            # Use the BaseAgent's _call_llm method for proper API handling
            # This will automatically use Responses API if enabled
            full_prompt = f"{system_prompt}\n\nUser: {query}\nAssistant:"
            
            # Simple research agent doesn't inherit from BaseAgent, so use direct API call
            # But follow the Responses API pattern from settings
            if settings.use_responses_api:
                response = self.client.responses.create(
                    model=model,
                    input=full_prompt,
                    reasoning={"effort": "medium"},
                    text={"verbosity": "medium"}
                )
                execution_time = time.time() - start_time
                content = response.output_text if hasattr(response, 'output_text') else ""
                
                # Extract token usage from Responses API
                if hasattr(response, 'usage') and response.usage:
                    token_usage = {
                        'prompt_tokens': response.usage.input_tokens,
                        'completion_tokens': response.usage.output_tokens,
                        'total_tokens': response.usage.total_tokens
                    }
                else:
                    token_usage = {}
            else:
                # Fallback to Chat Completions
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query}
                    ],
                    max_completion_tokens=1000
                )
                execution_time = time.time() - start_time
                content = response.choices[0].message.content
                token_usage = response.usage.model_dump() if response.usage else {}
            
            # Extract sources from the response
            sources = self._extract_sources(content)
            citations = self._extract_citations(content)
            
            return {
                'content': content,
                'sources': sources,
                'citations': citations,
                'execution_time': execution_time,
                'token_usage': {
                    'prompt_tokens': token_usage.get('prompt_tokens', 0),
                    'completion_tokens': token_usage.get('completion_tokens', 0),
                    'total_tokens': token_usage.get('total_tokens', 0)
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
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.task_type_analyzer = TaskTypeAnalyzer()
        self.search_agent = WebSearchAgent(self.client)
    
    def research(self, query: str, task_type_override: Optional[TaskType] = None) -> ResearchResult:
        """Perform research with intelligent model routing"""
        
        # Determine task type and select model
        task_type = task_type_override or self.task_type_analyzer.analyze_task_type(query)
        
        # Simple model selection based on task type
        if task_type == TaskType.DIRECT_ANSWER:
            model = settings.gpt5_nano_model  # Fast for simple facts
        elif task_type == TaskType.SEARCH_NEEDED:
            model = settings.gpt5_mini_model  # Good for search and synthesis
        else:  # RESEARCH_REPORT
            model = settings.gpt5_regular_model  # Full power for analysis
        
        print(f"Query: {query}")
        print(f"Detected task type: {task_type.value} → Using model: {model}")
        
        # Perform research using the selected model
        search_results = self.search_agent.search_and_synthesize(query, model)
        
        return ResearchResult(
            query=query,
            response=search_results['content'],
            sources=search_results['sources'],
            model_used=model,
            task_type_detected=task_type,
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