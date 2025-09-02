#!/usr/bin/env python3
"""
Evaluation dataset for testing research agents
"""
from pydantic import BaseModel
from typing import List, Optional
from enum import Enum
from config import ComplexityLevel

class EvalQuery(BaseModel):
    id: int
    query: str
    complexity: ComplexityLevel
    expected_sources: int  # Minimum number of sources expected
    requires_current_info: bool  # Whether query needs recent data
    domain: str  # Subject domain for categorization
    max_time_seconds: int  # Maximum acceptable response time

# Create evaluation dataset with 30 diverse queries
EVALUATION_QUERIES = [
    # SIMPLE queries (10) - Basic facts, definitions
    EvalQuery(
        id=1, 
        query="What is Python programming language?",
        complexity=ComplexityLevel.SIMPLE,
        expected_sources=2,
        requires_current_info=False,
        domain="technology",
        max_time_seconds=15
    ),
    EvalQuery(
        id=2,
        query="Who is the current President of the United States?", 
        complexity=ComplexityLevel.SIMPLE,
        expected_sources=2,
        requires_current_info=True,
        domain="politics",
        max_time_seconds=15
    ),
    EvalQuery(
        id=3,
        query="What is photosynthesis?",
        complexity=ComplexityLevel.SIMPLE,
        expected_sources=2,
        requires_current_info=False,
        domain="science",
        max_time_seconds=15
    ),
    EvalQuery(
        id=4,
        query="What is the capital of France?",
        complexity=ComplexityLevel.SIMPLE,
        expected_sources=1,
        requires_current_info=False,
        domain="geography",
        max_time_seconds=10
    ),
    EvalQuery(
        id=5,
        query="Define machine learning",
        complexity=ComplexityLevel.SIMPLE,
        expected_sources=2,
        requires_current_info=False,
        domain="technology",
        max_time_seconds=15
    ),
    EvalQuery(
        id=6,
        query="What is GDP?",
        complexity=ComplexityLevel.SIMPLE,
        expected_sources=2,
        requires_current_info=False,
        domain="economics",
        max_time_seconds=15
    ),
    EvalQuery(
        id=7,
        query="How many planets are in our solar system?",
        complexity=ComplexityLevel.SIMPLE,
        expected_sources=1,
        requires_current_info=False,
        domain="science",
        max_time_seconds=10
    ),
    EvalQuery(
        id=8,
        query="What is React.js?",
        complexity=ComplexityLevel.SIMPLE,
        expected_sources=2,
        requires_current_info=False,
        domain="technology",
        max_time_seconds=15
    ),
    EvalQuery(
        id=9,
        query="What is climate change?",
        complexity=ComplexityLevel.SIMPLE,
        expected_sources=2,
        requires_current_info=False,
        domain="environment",
        max_time_seconds=15
    ),
    EvalQuery(
        id=10,
        query="What is blockchain?",
        complexity=ComplexityLevel.SIMPLE,
        expected_sources=2,
        requires_current_info=False,
        domain="technology",
        max_time_seconds=15
    ),

    # MODERATE queries (10) - Multi-step reasoning, comparisons
    EvalQuery(
        id=11,
        query="Compare React vs Vue.js performance and ecosystem",
        complexity=ComplexityLevel.MODERATE,
        expected_sources=4,
        requires_current_info=True,
        domain="technology",
        max_time_seconds=30
    ),
    EvalQuery(
        id=12,
        query="What are the main differences between renewable and non-renewable energy sources?",
        complexity=ComplexityLevel.MODERATE,
        expected_sources=3,
        requires_current_info=False,
        domain="environment",
        max_time_seconds=25
    ),
    EvalQuery(
        id=13,
        query="How has remote work affected productivity in 2024?",
        complexity=ComplexityLevel.MODERATE,
        expected_sources=4,
        requires_current_info=True,
        domain="business",
        max_time_seconds=30
    ),
    EvalQuery(
        id=14,
        query="Explain quantum computing and its practical applications",
        complexity=ComplexityLevel.MODERATE,
        expected_sources=3,
        requires_current_info=True,
        domain="technology",
        max_time_seconds=30
    ),
    EvalQuery(
        id=15,
        query="What are the pros and cons of electric vehicles?",
        complexity=ComplexityLevel.MODERATE,
        expected_sources=4,
        requires_current_info=True,
        domain="automotive",
        max_time_seconds=25
    ),
    EvalQuery(
        id=16,
        query="How do different investment strategies perform during inflation?",
        complexity=ComplexityLevel.MODERATE,
        expected_sources=4,
        requires_current_info=True,
        domain="finance",
        max_time_seconds=30
    ),
    EvalQuery(
        id=17,
        query="Compare the effectiveness of different programming paradigms",
        complexity=ComplexityLevel.MODERATE,
        expected_sources=3,
        requires_current_info=False,
        domain="technology",
        max_time_seconds=25
    ),
    EvalQuery(
        id=18,
        query="What are the latest trends in artificial intelligence?",
        complexity=ComplexityLevel.MODERATE,
        expected_sources=5,
        requires_current_info=True,
        domain="technology",
        max_time_seconds=30
    ),
    EvalQuery(
        id=19,
        query="How does social media impact mental health?",
        complexity=ComplexityLevel.MODERATE,
        expected_sources=4,
        requires_current_info=True,
        domain="psychology",
        max_time_seconds=30
    ),
    EvalQuery(
        id=20,
        query="What factors influence cryptocurrency prices?",
        complexity=ComplexityLevel.MODERATE,
        expected_sources=4,
        requires_current_info=True,
        domain="finance",
        max_time_seconds=30
    ),

    # COMPLEX queries (10) - Deep analysis, multiple domains
    EvalQuery(
        id=21,
        query="Analyze the economic impact of AI automation on different industries and job markets",
        complexity=ComplexityLevel.COMPLEX,
        expected_sources=6,
        requires_current_info=True,
        domain="economics_technology",
        max_time_seconds=45
    ),
    EvalQuery(
        id=22,
        query="Compare venture capital investment trends 2023 vs 2024 across different sectors",
        complexity=ComplexityLevel.COMPLEX,
        expected_sources=6,
        requires_current_info=True,
        domain="finance_business",
        max_time_seconds=45
    ),
    EvalQuery(
        id=23,
        query="Evaluate the effectiveness of different climate change mitigation strategies globally",
        complexity=ComplexityLevel.COMPLEX,
        expected_sources=7,
        requires_current_info=True,
        domain="environment_policy",
        max_time_seconds=50
    ),
    EvalQuery(
        id=24,
        query="How do cultural differences affect international business negotiations and success rates?",
        complexity=ComplexityLevel.COMPLEX,
        expected_sources=5,
        requires_current_info=True,
        domain="business_culture",
        max_time_seconds=40
    ),
    EvalQuery(
        id=25,
        query="Analyze the relationship between education funding and student outcomes across different countries",
        complexity=ComplexityLevel.COMPLEX,
        expected_sources=6,
        requires_current_info=True,
        domain="education_policy",
        max_time_seconds=45
    ),
    EvalQuery(
        id=26,
        query="What are the long-term implications of quantum computing on cybersecurity and data privacy?",
        complexity=ComplexityLevel.COMPLEX,
        expected_sources=5,
        requires_current_info=True,
        domain="technology_security",
        max_time_seconds=40
    ),
    EvalQuery(
        id=27,
        query="Examine how demographic changes are reshaping healthcare delivery and costs",
        complexity=ComplexityLevel.COMPLEX,
        expected_sources=6,
        requires_current_info=True,
        domain="healthcare_demographics",
        max_time_seconds=45
    ),
    EvalQuery(
        id=28,
        query="Analyze the geopolitical implications of renewable energy transitions on global power dynamics",
        complexity=ComplexityLevel.COMPLEX,
        expected_sources=7,
        requires_current_info=True,
        domain="politics_environment",
        max_time_seconds=50
    ),
    EvalQuery(
        id=29,
        query="How do different urban planning approaches affect community wellbeing and economic development?",
        complexity=ComplexityLevel.COMPLEX,
        expected_sources=5,
        requires_current_info=True,
        domain="urban_planning_economics",
        max_time_seconds=40
    ),
    EvalQuery(
        id=30,
        query="Evaluate the effectiveness of different approaches to combating misinformation in the digital age",
        complexity=ComplexityLevel.COMPLEX,
        expected_sources=6,
        requires_current_info=True,
        domain="media_technology_policy",
        max_time_seconds=45
    ),
]

def get_queries_by_complexity(complexity: ComplexityLevel) -> List[EvalQuery]:
    """Get all queries of a specific complexity level"""
    return [q for q in EVALUATION_QUERIES if q.complexity == complexity]

def get_query_by_id(query_id: int) -> Optional[EvalQuery]:
    """Get a specific query by ID"""
    return next((q for q in EVALUATION_QUERIES if q.id == query_id), None)

def print_dataset_summary():
    """Print summary statistics of the evaluation dataset"""
    print("Evaluation Dataset Summary")
    print("=" * 40)
    
    by_complexity = {}
    by_domain = {}
    
    for query in EVALUATION_QUERIES:
        # Count by complexity
        complexity_key = query.complexity.value
        by_complexity[complexity_key] = by_complexity.get(complexity_key, 0) + 1
        
        # Count by domain
        by_domain[query.domain] = by_domain.get(query.domain, 0) + 1
    
    print("By Complexity:")
    for complexity, count in by_complexity.items():
        print(f"  {complexity}: {count} queries")
    
    print("\nBy Domain:")
    for domain, count in sorted(by_domain.items()):
        print(f"  {domain}: {count} queries")
    
    print(f"\nTotal queries: {len(EVALUATION_QUERIES)}")
    print(f"Requiring current info: {sum(1 for q in EVALUATION_QUERIES if q.requires_current_info)}")
    print(f"Average expected sources: {sum(q.expected_sources for q in EVALUATION_QUERIES) / len(EVALUATION_QUERIES):.1f}")

if __name__ == "__main__":
    print_dataset_summary()