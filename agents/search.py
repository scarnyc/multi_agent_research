import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from agents.base import BaseAgent
from agents.models import Task, TaskResult, AgentMessage, Priority, Citation, SearchResults, SearchResult
from config.settings import settings, ModelType, ReasoningEffort, Verbosity

logger = logging.getLogger(__name__)

class SearchAgent(BaseAgent):
    """Agent specialized in web search and information retrieval."""
    
    def __init__(self, reasoning_effort: ReasoningEffort = ReasoningEffort.LOW,
                 verbosity: Verbosity = Verbosity.MEDIUM):
        super().__init__(
            agent_id="search",
            model_type=ModelType.GPT5_MINI,  # Default to mini, will be adjusted by supervisor
            temperature=0.2,  # Lower temperature for more focused search
            max_tokens=2048,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity
        )
        self.max_search_results = 10
        self.search_timeout = 30  # seconds
    
    async def search(self, query: str, max_results: int = 5, current_info_required: bool = False) -> SearchResults:
        """Execute web search using OpenAI's websearch tool."""
        logger.info(f"Executing search for query: {query[:100]}...")
        
        search_prompt = self._build_search_prompt(query, current_info_required)
        
        try:
            # Call LLM with WebSearchTool - only for search agent
            response = await self._call_llm(
                input_text=search_prompt,
                tools=[{"type": "web_search_preview"}]  # Use OpenAI's web search tool
            )
            
            # Extract search results from response
            search_results = self._parse_search_response(response)
            
            # Convert dict results to SearchResult objects first
            search_result_objects = []
            for r in search_results[:max_results]:
                search_result_objects.append(SearchResult(
                    title=r.get('title', 'No Title'),
                    url=r.get('url', ''),
                    snippet=r.get('snippet', ''),
                    score=r.get('relevance_score', 0.5)
                ))
            
            # Simple relevance sorting (by score)
            search_result_objects.sort(key=lambda x: x.score, reverse=True)
            
            return SearchResults(
                query=query,
                results=search_result_objects,
                total_results=len(search_results),
                search_time=0.0  # TODO: Track actual search time
            )
            
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {str(e)}")
            # Return empty results on failure
            return SearchResults(
                query=query,
                results=[],
                total_results=0,
                search_time=0.0
            )
    
    def _build_search_prompt(self, query: str, current_info_required: bool) -> str:
        """Build an optimized search prompt."""
        time_constraint = "Use recent sources (2023-2024)" if current_info_required else "Any reliable sources"
        
        return f"""You are a research search specialist. Use the web_search tool to find comprehensive, reliable information about the following query.

Query: {query}

Search Requirements:
- {time_constraint}
- Focus on authoritative sources (academic, government, established news)
- Prioritize factual accuracy over opinion
- Look for diverse perspectives if the topic is complex

Execute the search and provide the most relevant results."""
    
    def _parse_search_response(self, response) -> List[Dict[str, Any]]:
        """Parse search results from LLM response."""
        results = []
        
        try:
            # Handle both Responses API and Chat Completions API formats
            if hasattr(response, 'output_text'):
                # Responses API format
                response_text = response.output_text
            else:
                # Chat Completions API format
                response_text = response.choices[0].message.content if response.choices else ""
            
            # The websearch tool provides results directly in the response text
            # No tool calls needed - results are integrated into the response content
            
            # Fallback: try to extract URLs and snippets from response text
            if not results:
                results = self._extract_results_from_text(response_text)
            
        except Exception as e:
            logger.error(f"Error parsing search response: {str(e)}")
        
        return results
    
    def _extract_search_results(self, search_data: Dict) -> List[Dict[str, Any]]:
        """Extract structured search results from tool response."""
        results = []
        
        # This format will depend on OpenAI's websearch tool structure
        if 'results' in search_data:
            for item in search_data['results']:
                result = {
                    'title': item.get('title', ''),
                    'url': item.get('url', ''),
                    'snippet': item.get('snippet', item.get('content', '')),
                    'source_type': self._determine_source_type(item.get('url', '')),
                    'relevance_score': 0.5  # Will be updated by ranking
                }
                results.append(result)
        
        return results
    
    def _extract_results_from_text(self, response_text: str) -> List[Dict[str, Any]]:
        """Fallback method to extract URLs and content from response text."""
        results = []
        
        # Look for URL patterns and associated content
        import re
        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, response_text)
        
        for url in urls[:5]:  # Limit to 5 URLs
            # Try to extract title and snippet near the URL
            url_index = response_text.find(url)
            
            # Extract surrounding text as snippet
            start = max(0, url_index - 200)
            end = min(len(response_text), url_index + 300)
            context = response_text[start:end]
            
            result = {
                'title': self._extract_title_from_context(context, url),
                'url': url,
                'snippet': context.replace(url, '').strip(),
                'source_type': self._determine_source_type(url),
                'relevance_score': 0.5
            }
            results.append(result)
        
        return results
    
    def _extract_title_from_context(self, context: str, url: str) -> str:
        """Extract a reasonable title from context."""
        # Look for title-like text before the URL
        lines = context.split('\n')
        for line in lines:
            if line.strip() and url not in line and len(line.strip()) < 200:
                return line.strip()
        
        # Fallback: use domain name
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            return f"Content from {domain}"
        except:
            return "Search Result"
    
    def _determine_source_type(self, url: str) -> str:
        """Determine the type of source based on URL."""
        url_lower = url.lower()
        
        if any(domain in url_lower for domain in ['.edu', 'scholar.google', 'pubmed', 'arxiv']):
            return 'academic'
        elif any(domain in url_lower for domain in ['.gov', '.mil']):
            return 'government'
        elif any(domain in url_lower for domain in ['reuters', 'ap.org', 'bbc', 'npr', 'cnn', 'nytimes']):
            return 'news'
        elif any(domain in url_lower for domain in ['wikipedia', 'britannica']):
            return 'reference'
        else:
            return 'general'
    
    async def _rank_by_relevance(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Rank search results by relevance to the query."""
        if not results:
            return results
        
        ranking_prompt = f"""You are a search relevance expert. Analyze these search results and rank them by relevance to the query.
        
Query: {query}

Search Results:
{self._format_results_for_ranking(results)}

For each result, provide a relevance score from 0.0 to 1.0 where:
- 1.0 = Directly answers the query with authoritative information
- 0.8 = Highly relevant with good supporting information  
- 0.6 = Moderately relevant, covers some aspects
- 0.4 = Somewhat relevant, tangentially related
- 0.2 = Low relevance, minimal connection
- 0.0 = Not relevant to the query

Respond with a JSON array of scores in the same order: [0.9, 0.7, 0.5, ...]"""
        
        try:
            response = await self._call_llm(input_text=ranking_prompt)
            response_text = response.output_text if hasattr(response, 'output_text') else response.choices[0].message.content
            
            # Extract scores from response
            scores = json.loads(response_text.strip())
            
            # Apply scores to results
            for i, result in enumerate(results):
                if i < len(scores):
                    result['relevance_score'] = scores[i]
            
            # Sort by relevance score
            results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            
        except Exception as e:
            logger.warning(f"Failed to rank results by relevance: {str(e)}")
            # Fallback: prefer academic and government sources
            results.sort(key=lambda x: (
                x.get('source_type') in ['academic', 'government'],
                len(x.get('snippet', ''))
            ), reverse=True)
        
        return results
    
    def _format_results_for_ranking(self, results: List[Dict[str, Any]]) -> str:
        """Format results for the ranking prompt."""
        formatted = []
        for i, result in enumerate(results):
            formatted.append(f"{i+1}. {result.get('title', 'No title')}")
            formatted.append(f"   URL: {result.get('url', 'No URL')}")
            formatted.append(f"   Type: {result.get('source_type', 'unknown')}")
            formatted.append(f"   Snippet: {result.get('snippet', 'No snippet')[:200]}...")
            formatted.append("")
        
        return '\n'.join(formatted)
    
    async def extract_relevant_content(self, search_results: SearchResults, context: str) -> Dict[str, Any]:
        """Extract the most relevant content from search results for a specific context."""
        if not search_results.results:
            return {"content": "", "sources": []}
        
        extraction_prompt = f"""You are a content extraction specialist. Given these search results and the research context, extract and synthesize the most relevant information.

Research Context: {context}

Search Results:
{self._format_search_results(search_results)}

Instructions:
1. Extract key facts, figures, and insights relevant to the research context
2. Maintain factual accuracy - only use information directly stated in the sources
3. Organize the information logically
4. Note which sources support each piece of information
5. Identify any conflicting information between sources

Provide a structured response with:
- Main findings
- Supporting details
- Source attribution
- Any conflicting information noted"""
        
        try:
            response = await self._call_llm(input_text=extraction_prompt)
            extracted_content = response.output_text if hasattr(response, 'output_text') else response.choices[0].message.content
            
            return {
                "content": extracted_content,
                "sources": [r.url for r in search_results.results],
                "source_count": len(search_results.results),
                "extraction_time": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Content extraction failed: {str(e)}")
            return {
                "content": f"Error extracting content: {str(e)}",
                "sources": [],
                "source_count": 0,
                "error": str(e)
            }
    
    def _format_search_results(self, search_results: SearchResults) -> str:
        """Format search results for content extraction."""
        formatted = []
        for i, result in enumerate(search_results.results):
            formatted.append(f"Source {i+1}:")
            formatted.append(f"Title: {result.title}")
            formatted.append(f"URL: {result.url}")
            formatted.append(f"Type: {getattr(result, 'metadata', {}).get('source_type', 'web')}")
            formatted.append(f"Relevance: {result.score:.2f}")
            formatted.append(f"Content: {result.snippet}")
            formatted.append("-" * 50)
        
        return '\n'.join(formatted)
    
    async def process_task(self, task: Task) -> Any:
        """Process a search task."""
        logger.info(f"SearchAgent processing task: {task.id}")
        
        # Select appropriate model for this query
        selected_model = self._select_model_for_query(task.description)
        if selected_model != self.model_type:
            logger.info(f"Switching from {self.model_type.value} to {selected_model.value} for this query")
            self.model_type = selected_model
        
        # Determine if current information is needed
        current_info_needed = any(keyword in task.description.lower() 
                                for keyword in ['current', 'recent', 'latest', '2024', '2025', 'now'])
        
        # Determine search parameters based on model selection
        if selected_model == ModelType.GPT5_NANO:
            max_results = 3  # Simple searches need fewer results
        elif selected_model == ModelType.GPT5_MINI:
            max_results = 5  # Moderate searches need more sources
        else:
            max_results = 8  # Complex analysis needs comprehensive sources
        
        # Execute search
        search_results = await self.search(
            query=task.description,
            max_results=max_results,
            current_info_required=current_info_needed
        )
        
        # Extract relevant content
        extracted_content = await self.extract_relevant_content(search_results, task.description)
        
        # Create citations from search results
        citations = []
        for result in search_results.results:
            citation = Citation(
                content=result.snippet,
                url=result.url,
                title=result.title,
                credibility_score=result.score
            )
            citations.append(citation)
        
        return {
            "search_results": search_results.results,
            "extracted_content": extracted_content["content"],
            "sources_consulted": len(search_results.results),
            "citations": citations,
            "search_quality": {
                "total_found": search_results.total_results,
                "avg_relevance": sum(r.score for r in search_results.results) / len(search_results.results) if search_results.results else 0,
                "source_diversity": len(set('web' for r in search_results.results))  # Simplified for now
            }
        }
    
    def _select_model_for_query(self, query: str) -> ModelType:
        """Select the most appropriate model based on query characteristics."""
        word_count = len(query.split())
        has_multiple_concepts = any(word in query.lower() for word in ['and', 'vs', 'versus', 'compare', 'different', 'analyze'])
        has_complex_reasoning = any(word in query.lower() for word in ['implications', 'impact', 'trends', 'effectiveness', 'relationship'])
        has_recent_requirement = any(word in query.lower() for word in ['latest', 'recent', 'current', '2024', '2025'])
        
        # More sophisticated model selection based on query characteristics
        if has_complex_reasoning or (has_multiple_concepts and word_count > 15):
            return ModelType.GPT5_REGULAR  # Complex analysis requires full model
        elif has_multiple_concepts or word_count > 8 or has_recent_requirement:
            return ModelType.GPT5_MINI  # Multi-step reasoning or recent info needs mini
        else:
            return ModelType.GPT5_NANO  # Simple factual searches can use nano
    
    async def _process_critical_message(self, message: AgentMessage) -> None:
        """Handle critical priority messages."""
        logger.warning(f"SearchAgent received critical message from {message.sender}: {message.payload}")
        
        # Handle search timeout or failure recovery
        if "timeout" in message.payload or "retry" in message.payload:
            logger.info("Implementing search retry strategy")
            # Could implement backup search strategies here
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get statistics specific to search operations."""
        stats = self.get_stats()
        
        # Add search-specific metrics
        completed_searches = [t for t in self.task_history if t.status.name == 'COMPLETED']
        
        if completed_searches:
            # Calculate average sources found (would need to track this in task results)
            stats.update({
                "avg_sources_per_search": 0,  # Would be calculated from task results
                "search_success_rate": len(completed_searches) / len(self.task_history) if self.task_history else 0,
                "preferred_source_types": {},  # Would track most common source types
            })
        
        return stats