import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Set
from datetime import datetime
from urllib.parse import urlparse
import re

from agents.base import BaseAgent
from agents.models import Task, TaskResult, AgentMessage, Priority, Citation
from config.settings import settings, ModelType, ReasoningEffort, Verbosity

logger = logging.getLogger(__name__)

class CitationAgent(BaseAgent):
    """Agent specialized in source tracking, citation formatting, and credibility verification."""
    
    def __init__(self, reasoning_effort: ReasoningEffort = ReasoningEffort.LOW,
                 verbosity: Verbosity = Verbosity.LOW):
        super().__init__(
            agent_id="citation",
            model_type=ModelType.GPT5_NANO,  # Lightweight processing for citations
            temperature=0.1,  # Very low temperature for consistent formatting
            max_tokens=1024,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity
        )
        self.tracked_sources: Dict[str, Citation] = {}
        self.citation_styles = {
            'APA': self._format_apa_citation,
            'MLA': self._format_mla_citation,
            'Chicago': self._format_chicago_citation,
            'IEEE': self._format_ieee_citation
        }
        # Domain-based credibility scoring
        self.credibility_domains = {
            'high': ['.edu', '.gov', '.mil', 'scholar.google', 'pubmed.ncbi', 'arxiv.org', 'jstor.org'],
            'medium_high': ['reuters.com', 'ap.org', 'bbc.co.uk', 'npr.org', 'nature.com', 'science.org'],
            'medium': ['wikipedia.org', 'britannica.com', 'pew.org', 'brookings.edu'],
            'low': ['blog', '.com', '.net', '.org'],  # Generic domains get lower scores
        }
    
    async def track_source(self, content: str, url: str, metadata: Dict[str, Any] = None) -> Citation:
        """Track a source and create a citation."""
        logger.info(f"Tracking source: {url}")
        
        if metadata is None:
            metadata = {}
        
        # Extract title and other metadata if not provided
        if 'title' not in metadata:
            metadata['title'] = await self._extract_title_from_content(content, url)
        
        if 'author' not in metadata:
            metadata['author'] = await self._extract_author_from_content(content, url)
        
        if 'publish_date' not in metadata:
            metadata['publish_date'] = await self._extract_publish_date(content, url)
        
        # Create citation object
        citation = Citation(
            url=url,
            title=metadata.get('title', 'Unknown Title'),
            author=metadata.get('author'),
            publish_date=metadata.get('publish_date'),
            accessed_date=datetime.now(),
            source_type=self._determine_source_type(url),
            relevance_score=metadata.get('relevance_score', 0.5),
            credibility_score=await self.verify_credibility(url),
            content_snippet=content[:500] if content else "",
            metadata=metadata
        )
        
        # Store in tracked sources
        self.tracked_sources[url] = citation
        
        return citation
    
    async def verify_credibility(self, url: str) -> float:
        """Verify the credibility of a source based on domain and other factors."""
        try:
            parsed_url = urlparse(url.lower())
            domain = parsed_url.netloc.lower()
            
            # Domain-based scoring
            base_score = 0.3  # Default score
            
            for credibility_level, domains in self.credibility_domains.items():
                if any(trusted_domain in domain for trusted_domain in domains):
                    if credibility_level == 'high':
                        base_score = 0.9
                    elif credibility_level == 'medium_high':
                        base_score = 0.8
                    elif credibility_level == 'medium':
                        base_score = 0.6
                    else:  # low
                        base_score = 0.4
                    break
            
            # Additional credibility factors
            credibility_boost = 0.0
            
            # HTTPS adds credibility
            if parsed_url.scheme == 'https':
                credibility_boost += 0.05
            
            # Recent domains might be less credible
            if any(new_tld in domain for new_tld in ['.tk', '.ml', '.ga', '.cf']):
                credibility_boost -= 0.2
            
            # Academic and research indicators
            if any(indicator in domain for indicator in ['research', 'institute', 'university', 'academic']):
                credibility_boost += 0.1
            
            # News outlets with good reputation
            reputable_news = ['reuters', 'ap.org', 'bbc', 'npr', 'pbs', 'economist']
            if any(outlet in domain for outlet in reputable_news):
                credibility_boost += 0.1
            
            final_score = min(1.0, max(0.0, base_score + credibility_boost))
            
            logger.debug(f"Credibility score for {domain}: {final_score}")
            return final_score
            
        except Exception as e:
            logger.warning(f"Error verifying credibility for {url}: {str(e)}")
            return 0.5  # Default middle score on error
    
    def _determine_source_type(self, url: str) -> str:
        """Determine the type of source based on URL patterns."""
        url_lower = url.lower()
        
        if any(domain in url_lower for domain in ['.edu', 'scholar.google', 'pubmed', 'arxiv']):
            return 'academic'
        elif any(domain in url_lower for domain in ['.gov', '.mil']):
            return 'government'
        elif any(domain in url_lower for domain in ['reuters', 'ap.org', 'bbc', 'npr', 'cnn', 'nytimes', 'wsj']):
            return 'news'
        elif any(domain in url_lower for domain in ['wikipedia', 'britannica']):
            return 'reference'
        elif any(domain in url_lower for domain in ['blog', 'medium.com', 'substack']):
            return 'blog'
        elif any(domain in url_lower for domain in ['youtube', 'vimeo', 'podcast']):
            return 'media'
        else:
            return 'general'
    
    async def _extract_title_from_content(self, content: str, url: str) -> str:
        """Extract title from content using LLM."""
        if not content:
            return self._generate_title_from_url(url)
        
        title_prompt = f"""Extract the main title from this content. Return only the title, no additional text.

Content: {content[:1000]}

Title:"""
        
        try:
            response = await self._call_llm(input_text=title_prompt)
            title = response.output_text if hasattr(response, 'output_text') else response.choices[0].message.content
            title = title.strip().strip('"').strip("'")
            
            return title if title and len(title) < 200 else self._generate_title_from_url(url)
            
        except Exception as e:
            logger.warning(f"Failed to extract title from content: {str(e)}")
            return self._generate_title_from_url(url)
    
    def _generate_title_from_url(self, url: str) -> str:
        """Generate a title from URL when content extraction fails."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.replace('www.', '')
            path = parsed.path.strip('/')
            
            if path:
                # Convert path to readable title
                title_parts = path.split('/')[-1].split('-')
                title_parts = [part.replace('_', ' ').title() for part in title_parts]
                return f"{' '.join(title_parts)} - {domain}"
            else:
                return f"Content from {domain}"
                
        except Exception:
            return "Unknown Source"
    
    async def _extract_author_from_content(self, content: str, url: str) -> Optional[str]:
        """Extract author information from content."""
        if not content:
            return None
        
        author_prompt = f"""Extract the author's name from this content. Look for bylines, author credits, or author information.
        Return only the author's name, or "None" if no clear author is found.

Content: {content[:800]}

Author:"""
        
        try:
            response = await self._call_llm(input_text=author_prompt)
            author = response.output_text if hasattr(response, 'output_text') else response.choices[0].message.content
            author = author.strip().strip('"').strip("'")
            
            if author and author.lower() not in ['none', 'unknown', 'n/a', ''] and len(author) < 100:
                return author
            else:
                return None
                
        except Exception as e:
            logger.warning(f"Failed to extract author from content: {str(e)}")
            return None
    
    async def _extract_publish_date(self, content: str, url: str) -> Optional[datetime]:
        """Extract publication date from content."""
        if not content:
            return None
        
        date_prompt = f"""Extract the publication date from this content. Look for dates in various formats.
        Return the date in YYYY-MM-DD format, or "None" if no clear publication date is found.

Content: {content[:800]}

Publication Date:"""
        
        try:
            response = await self._call_llm(input_text=date_prompt)
            date_str = response.output_text if hasattr(response, 'output_text') else response.choices[0].message.content
            date_str = date_str.strip().strip('"').strip("'")
            
            if date_str and date_str.lower() != 'none':
                # Try to parse the date
                return self._parse_date_string(date_str)
            else:
                return None
                
        except Exception as e:
            logger.warning(f"Failed to extract publish date from content: {str(e)}")
            return None
    
    def _parse_date_string(self, date_str: str) -> Optional[datetime]:
        """Parse various date string formats."""
        try:
            # Try common formats
            formats = [
                '%Y-%m-%d',
                '%Y/%m/%d', 
                '%m/%d/%Y',
                '%m-%d-%Y',
                '%B %d, %Y',
                '%d %B %Y',
                '%Y-%m-%d %H:%M:%S'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_str.strip(), fmt)
                except ValueError:
                    continue
            
            # If all formats fail, try regex for YYYY
            year_match = re.search(r'20[0-9]{2}', date_str)
            if year_match:
                year = int(year_match.group())
                return datetime(year, 1, 1)  # Default to January 1st
            
        except Exception as e:
            logger.debug(f"Date parsing failed for '{date_str}': {str(e)}")
        
        return None
    
    def format_citation(self, citation: Citation, style: str = 'APA') -> str:
        """Format a citation in the specified style."""
        formatter = self.citation_styles.get(style.upper(), self._format_apa_citation)
        return formatter(citation)
    
    def _format_apa_citation(self, citation: Citation) -> str:
        """Format citation in APA style."""
        parts = []
        
        # Author (Year). Title. Source. URL
        if citation.author:
            if citation.publish_date:
                year = citation.publish_date.year
                parts.append(f"{citation.author} ({year}).")
            else:
                parts.append(f"{citation.author} (n.d.).")
        else:
            if citation.publish_date:
                year = citation.publish_date.year
                parts.append(f"({year}).")
            else:
                parts.append("(n.d.).")
        
        # Title
        if citation.source_type == 'news':
            parts.append(f"{citation.title}.")
        else:
            parts.append(f"*{citation.title}*.")
        
        # URL and access date
        accessed = citation.accessed_date.strftime('%B %d, %Y')
        parts.append(f"Retrieved {accessed}, from {citation.url}")
        
        return ' '.join(parts)
    
    def _format_mla_citation(self, citation: Citation) -> str:
        """Format citation in MLA style."""
        parts = []
        
        # Author. "Title." Website Name, Date, URL.
        if citation.author:
            parts.append(f"{citation.author}.")
        
        parts.append(f'"{citation.title}."')
        
        # Website name from URL
        try:
            domain = urlparse(citation.url).netloc.replace('www.', '')
            parts.append(f"{domain.title()},")
        except:
            pass
        
        if citation.publish_date:
            date_str = citation.publish_date.strftime('%d %b %Y')
            parts.append(f"{date_str},")
        
        parts.append(f"{citation.url}.")
        
        return ' '.join(parts)
    
    def _format_chicago_citation(self, citation: Citation) -> str:
        """Format citation in Chicago style."""
        parts = []
        
        # Author. "Title." Website Name. Accessed Date. URL.
        if citation.author:
            parts.append(f"{citation.author}.")
        
        parts.append(f'"{citation.title}."')
        
        try:
            domain = urlparse(citation.url).netloc.replace('www.', '')
            parts.append(f"{domain.title()}.")
        except:
            pass
        
        accessed = citation.accessed_date.strftime('%B %d, %Y')
        parts.append(f"Accessed {accessed}.")
        
        parts.append(f"{citation.url}.")
        
        return ' '.join(parts)
    
    def _format_ieee_citation(self, citation: Citation) -> str:
        """Format citation in IEEE style."""
        parts = []
        
        # [1] Author, "Title," Website, Date. [Online]. Available: URL
        if citation.author:
            parts.append(f'{citation.author}, ')
        
        parts.append(f'"{citation.title},"')
        
        try:
            domain = urlparse(citation.url).netloc.replace('www.', '')
            parts.append(f"{domain},")
        except:
            pass
        
        if citation.publish_date:
            date_str = citation.publish_date.strftime('%b. %Y')
            parts.append(f"{date_str}.")
        
        parts.append("[Online].")
        parts.append(f"Available: {citation.url}")
        
        return ''.join(parts)
    
    def generate_bibliography(self, citations: List[Citation], style: str = 'APA') -> str:
        """Generate a formatted bibliography from citations."""
        if not citations:
            return "No sources cited."
        
        # Remove duplicates by URL
        unique_citations = {c.url: c for c in citations}.values()
        
        # Sort citations alphabetically by title or author
        sorted_citations = sorted(
            unique_citations,
            key=lambda c: (c.author or c.title or 'zzz').lower()
        )
        
        bibliography_lines = []
        bibliography_lines.append(f"\n## Bibliography ({style} Style)\n")
        
        for i, citation in enumerate(sorted_citations, 1):
            if style.upper() == 'IEEE':
                formatted = f"[{i}] {self.format_citation(citation, style)}"
            else:
                formatted = self.format_citation(citation, style)
            
            bibliography_lines.append(formatted)
            bibliography_lines.append("")  # Empty line between citations
        
        return '\n'.join(bibliography_lines)
    
    async def detect_potential_misinformation(self, content: str, citations: List[Citation]) -> Dict[str, Any]:
        """Detect potential misinformation indicators."""
        
        misinformation_prompt = f"""You are a fact-checking specialist. Analyze this content and citations for potential misinformation indicators.

Content: {content[:2000]}

Citations: {[c.url for c in citations[:5]]}

Look for these red flags:
1. Extraordinary claims without extraordinary evidence
2. Sources with low credibility scores
3. Lack of citations for factual claims
4. Contradictions within the content
5. Emotional language designed to manipulate
6. Claims that go against scientific consensus without proper evidence

Respond with a JSON object containing:
{{
    "risk_level": "low|medium|high",
    "risk_score": 0.0-1.0,
    "concerns": ["list of specific concerns"],
    "recommendations": ["list of recommendations"]
}}"""
        
        try:
            response = await self._call_llm(input_text=misinformation_prompt)
            response_text = response.output_text if hasattr(response, 'output_text') else response.choices[0].message.content
            
            analysis = json.loads(response_text)
            
            # Add credibility analysis of citations
            if citations:
                avg_credibility = sum(c.credibility_score for c in citations) / len(citations)
                analysis['citation_credibility'] = avg_credibility
                
                if avg_credibility < 0.5:
                    analysis['concerns'].append("Low credibility sources used")
                    analysis['risk_score'] = min(1.0, analysis.get('risk_score', 0) + 0.2)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Misinformation detection failed: {str(e)}")
            return {
                "risk_level": "unknown",
                "risk_score": 0.5,
                "concerns": ["Unable to analyze content"],
                "recommendations": ["Manual fact-checking recommended"],
                "error": str(e)
            }
    
    async def process_task(self, task: Task) -> Any:
        """Process a citation-related task."""
        logger.info(f"CitationAgent processing task: {task.id}")
        
        task_data = task.description if isinstance(task.description, str) else task.description
        
        if isinstance(task_data, dict):
            # Handle structured citation tasks
            if 'action' in task_data:
                action = task_data['action']
                
                if action == 'track_sources':
                    sources = task_data.get('sources', [])
                    citations = []
                    
                    for source in sources:
                        citation = await self.track_source(
                            content=source.get('content', ''),
                            url=source.get('url', ''),
                            metadata=source.get('metadata', {})
                        )
                        citations.append(citation)
                    
                    return {
                        "citations": citations,
                        "total_sources": len(citations),
                        "avg_credibility": sum(c.credibility_score for c in citations) / len(citations) if citations else 0
                    }
                
                elif action == 'generate_bibliography':
                    citations = [Citation(**c) for c in task_data.get('citations', [])]
                    style = task_data.get('style', 'APA')
                    
                    bibliography = self.generate_bibliography(citations, style)
                    
                    return {
                        "bibliography": bibliography,
                        "citation_count": len(citations),
                        "style": style
                    }
                
                elif action == 'verify_credibility':
                    urls = task_data.get('urls', [])
                    credibility_scores = {}
                    
                    for url in urls:
                        score = await self.verify_credibility(url)
                        credibility_scores[url] = score
                    
                    return {
                        "credibility_scores": credibility_scores,
                        "avg_credibility": sum(credibility_scores.values()) / len(credibility_scores) if credibility_scores else 0
                    }
        
        # Default: treat as content that needs citation analysis
        return {
            "message": "Citation task processed",
            "task_type": "analysis",
            "tracked_sources": len(self.tracked_sources)
        }
    
    async def _process_critical_message(self, message: AgentMessage) -> None:
        """Handle critical priority messages."""
        logger.warning(f"CitationAgent received critical message from {message.sender}: {message.payload}")
        
        # Handle citation verification failures or urgent fact-checking requests
        if "verify" in message.payload or "fact_check" in message.payload:
            logger.info("Implementing urgent fact-checking protocol")
    
    def get_citation_stats(self) -> Dict[str, Any]:
        """Get statistics specific to citation operations."""
        stats = self.get_stats()
        
        if self.tracked_sources:
            # Add citation-specific metrics
            credibility_scores = [c.credibility_score for c in self.tracked_sources.values()]
            source_types = [c.source_type for c in self.tracked_sources.values()]
            
            stats.update({
                "total_sources_tracked": len(self.tracked_sources),
                "avg_credibility_score": sum(credibility_scores) / len(credibility_scores),
                "source_type_distribution": {
                    source_type: source_types.count(source_type) 
                    for source_type in set(source_types)
                },
                "high_credibility_sources": len([s for s in credibility_scores if s >= 0.8]),
                "low_credibility_sources": len([s for s in credibility_scores if s < 0.5])
            })
        
        return stats