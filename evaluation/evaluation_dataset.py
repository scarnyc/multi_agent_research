#!/usr/bin/env python3
"""
Evaluation dataset for testing research agents
40 diverse queries across 4 complexity levels
"""
from pydantic import BaseModel
from typing import List, Optional
from enum import Enum
import pandas as pd
from config.settings import ComplexityLevel

class EvalQuery(BaseModel):
    id: int
    query: str
    complexity: ComplexityLevel
    expected_sources: int  # Minimum number of sources expected
    requires_current_info: bool  # Whether query needs recent data
    domain: str  # Subject domain for categorization
    max_time_seconds: int  # Maximum acceptable response time
    query_type: str  # "qa" or "research"

# Create evaluation dataset with 40 diverse queries
EVALUATION_QUERIES = [
    # ============================================
    # EASY Q&A (10) - Simple factual questions
    # ============================================
    EvalQuery(
        id=1,
        query="What is the speed of light in a vacuum?",
        complexity=ComplexityLevel.SIMPLE,
        expected_sources=1,
        requires_current_info=False,
        domain="physics",
        max_time_seconds=10,
        query_type="qa"
    ),
    EvalQuery(
        id=2,
        query="Who invented the telephone?",
        complexity=ComplexityLevel.SIMPLE,
        expected_sources=1,
        requires_current_info=False,
        domain="history",
        max_time_seconds=10,
        query_type="qa"
    ),
    EvalQuery(
        id=3,
        query="What is the chemical formula for water?",
        complexity=ComplexityLevel.SIMPLE,
        expected_sources=1,
        requires_current_info=False,
        domain="chemistry",
        max_time_seconds=10,
        query_type="qa"
    ),
    EvalQuery(
        id=4,
        query="How many continents are there on Earth?",
        complexity=ComplexityLevel.SIMPLE,
        expected_sources=1,
        requires_current_info=False,
        domain="geography",
        max_time_seconds=10,
        query_type="qa"
    ),
    EvalQuery(
        id=5,
        query="What is the largest planet in our solar system?",
        complexity=ComplexityLevel.SIMPLE,
        expected_sources=1,
        requires_current_info=False,
        domain="astronomy",
        max_time_seconds=10,
        query_type="qa"
    ),
    EvalQuery(
        id=6,
        query="Who wrote '1984'?",
        complexity=ComplexityLevel.SIMPLE,
        expected_sources=1,
        requires_current_info=False,
        domain="literature",
        max_time_seconds=10,
        query_type="qa"
    ),
    EvalQuery(
        id=7,
        query="What is the boiling point of water at sea level?",
        complexity=ComplexityLevel.SIMPLE,
        expected_sources=1,
        requires_current_info=False,
        domain="physics",
        max_time_seconds=10,
        query_type="qa"
    ),
    EvalQuery(
        id=8,
        query="What year did World War II end?",
        complexity=ComplexityLevel.SIMPLE,
        expected_sources=1,
        requires_current_info=False,
        domain="history",
        max_time_seconds=10,
        query_type="qa"
    ),
    EvalQuery(
        id=9,
        query="What is the primary function of red blood cells?",
        complexity=ComplexityLevel.SIMPLE,
        expected_sources=1,
        requires_current_info=False,
        domain="biology",
        max_time_seconds=10,
        query_type="qa"
    ),
    EvalQuery(
        id=10,
        query="What programming language is known for its use in web browsers?",
        complexity=ComplexityLevel.SIMPLE,
        expected_sources=1,
        requires_current_info=False,
        domain="technology",
        max_time_seconds=10,
        query_type="qa"
    ),

    # ============================================
    # MEDIUM Q&A (10) - Multi-faceted questions requiring explanation
    # ============================================
    EvalQuery(
        id=11,
        query="How does photosynthesis convert light energy into chemical energy?",
        complexity=ComplexityLevel.MODERATE,
        expected_sources=3,
        requires_current_info=False,
        domain="biology",
        max_time_seconds=25,
        query_type="qa"
    ),
    EvalQuery(
        id=12,
        query="What are the key differences between supervised and unsupervised learning in machine learning?",
        complexity=ComplexityLevel.MODERATE,
        expected_sources=3,
        requires_current_info=False,
        domain="technology",
        max_time_seconds=25,
        query_type="qa"
    ),
    EvalQuery(
        id=13,
        query="Explain the concept of supply and demand in economics with examples.",
        complexity=ComplexityLevel.MODERATE,
        expected_sources=3,
        requires_current_info=False,
        domain="economics",
        max_time_seconds=25,
        query_type="qa"
    ),
    EvalQuery(
        id=14,
        query="How does the human immune system respond to viral infections?",
        complexity=ComplexityLevel.MODERATE,
        expected_sources=3,
        requires_current_info=False,
        domain="medicine",
        max_time_seconds=25,
        query_type="qa"
    ),
    EvalQuery(
        id=15,
        query="What are the main causes and effects of the greenhouse effect?",
        complexity=ComplexityLevel.MODERATE,
        expected_sources=3,
        requires_current_info=False,
        domain="environment",
        max_time_seconds=25,
        query_type="qa"
    ),
    EvalQuery(
        id=16,
        query="How do neural networks learn patterns through backpropagation?",
        complexity=ComplexityLevel.MODERATE,
        expected_sources=3,
        requires_current_info=False,
        domain="ai",
        max_time_seconds=25,
        query_type="qa"
    ),
    EvalQuery(
        id=17,
        query="What are the key principles of object-oriented programming and why are they important?",
        complexity=ComplexityLevel.MODERATE,
        expected_sources=3,
        requires_current_info=False,
        domain="technology",
        max_time_seconds=25,
        query_type="qa"
    ),
    EvalQuery(
        id=18,
        query="Explain how DNA replication ensures genetic continuity in cells.",
        complexity=ComplexityLevel.MODERATE,
        expected_sources=3,
        requires_current_info=False,
        domain="biology",
        max_time_seconds=25,
        query_type="qa"
    ),
    EvalQuery(
        id=19,
        query="What are the main types of renewable energy and their advantages/disadvantages?",
        complexity=ComplexityLevel.MODERATE,
        expected_sources=4,
        requires_current_info=True,
        domain="energy",
        max_time_seconds=25,
        query_type="qa"
    ),
    EvalQuery(
        id=20,
        query="How does blockchain technology ensure security and immutability of transactions?",
        complexity=ComplexityLevel.MODERATE,
        expected_sources=3,
        requires_current_info=False,
        domain="technology",
        max_time_seconds=25,
        query_type="qa"
    ),

    # ============================================
    # HIGH COMPLEXITY Q&A (10) - Complex analytical questions
    # ============================================
    EvalQuery(
        id=21,
        query="What are the latest breakthroughs in quantum error correction and how do they impact the feasibility of practical quantum computers?",
        complexity=ComplexityLevel.COMPLEX,
        expected_sources=5,
        requires_current_info=True,
        domain="quantum_computing",
        max_time_seconds=40,
        query_type="qa"
    ),
    EvalQuery(
        id=22,
        query="How are transformer models like GPT revolutionizing natural language processing and what are their current limitations?",
        complexity=ComplexityLevel.COMPLEX,
        expected_sources=5,
        requires_current_info=True,
        domain="ai",
        max_time_seconds=40,
        query_type="qa"
    ),
    EvalQuery(
        id=23,
        query="What are the most promising approaches to fusion energy and what technical challenges remain before commercial viability?",
        complexity=ComplexityLevel.COMPLEX,
        expected_sources=5,
        requires_current_info=True,
        domain="energy",
        max_time_seconds=40,
        query_type="qa"
    ),
    EvalQuery(
        id=24,
        query="How is CRISPR-Cas9 gene editing being used in current medical trials and what ethical considerations are being debated?",
        complexity=ComplexityLevel.COMPLEX,
        expected_sources=5,
        requires_current_info=True,
        domain="biotechnology",
        max_time_seconds=40,
        query_type="qa"
    ),
    EvalQuery(
        id=25,
        query="What are the competing theories about dark matter and dark energy, and what recent observations support or challenge them?",
        complexity=ComplexityLevel.COMPLEX,
        expected_sources=5,
        requires_current_info=True,
        domain="astrophysics",
        max_time_seconds=40,
        query_type="qa"
    ),
    EvalQuery(
        id=26,
        query="How are central banks using digital currencies (CBDCs) and what implications do they have for monetary policy and privacy?",
        complexity=ComplexityLevel.COMPLEX,
        expected_sources=5,
        requires_current_info=True,
        domain="finance",
        max_time_seconds=40,
        query_type="qa"
    ),
    EvalQuery(
        id=27,
        query="What are the latest developments in mRNA vaccine technology beyond COVID-19 and their potential applications in cancer treatment?",
        complexity=ComplexityLevel.COMPLEX,
        expected_sources=5,
        requires_current_info=True,
        domain="medicine",
        max_time_seconds=40,
        query_type="qa"
    ),
    EvalQuery(
        id=28,
        query="How are autonomous vehicles solving the edge cases and ethical dilemmas in real-world deployment scenarios?",
        complexity=ComplexityLevel.COMPLEX,
        expected_sources=5,
        requires_current_info=True,
        domain="autonomous_systems",
        max_time_seconds=40,
        query_type="qa"
    ),
    EvalQuery(
        id=29,
        query="What are the current approaches to achieving artificial general intelligence (AGI) and what are the key technical and philosophical challenges?",
        complexity=ComplexityLevel.COMPLEX,
        expected_sources=5,
        requires_current_info=True,
        domain="ai",
        max_time_seconds=40,
        query_type="qa"
    ),
    EvalQuery(
        id=30,
        query="How is synthetic biology being used to create sustainable materials and what are the environmental and safety considerations?",
        complexity=ComplexityLevel.COMPLEX,
        expected_sources=5,
        requires_current_info=True,
        domain="biotechnology",
        max_time_seconds=40,
        query_type="qa"
    ),

    # ============================================
    # DEEP RESEARCH (10) - Comprehensive 2-page reports
    # ============================================
    EvalQuery(
        id=31,
        query="Provide a comprehensive analysis of the current state of climate change mitigation strategies globally, including carbon capture technologies, renewable energy transitions, policy frameworks, and their effectiveness. Include specific country examples, recent COP agreements, technological breakthroughs, and projections for meeting Paris Agreement targets.",
        complexity=ComplexityLevel.COMPLEX,
        expected_sources=10,
        requires_current_info=True,
        domain="climate_environment",
        max_time_seconds=60,
        query_type="research"
    ),
    EvalQuery(
        id=32,
        query="Create a detailed report on the current landscape of artificial intelligence development, covering the latest large language models, computer vision breakthroughs, AI governance initiatives, major industry players, investment trends, ethical challenges, regulatory approaches across different countries, and predictions for AI's impact on employment and society over the next decade.",
        complexity=ComplexityLevel.COMPLEX,
        expected_sources=10,
        requires_current_info=True,
        domain="ai_technology",
        max_time_seconds=60,
        query_type="research"
    ),
    EvalQuery(
        id=33,
        query="Analyze the global semiconductor industry crisis and recovery, including supply chain vulnerabilities exposed during COVID-19, the geopolitics of chip manufacturing, major investments in new fabs, the role of Taiwan and TSMC, US CHIPS Act implications, China's semiconductor ambitions, and the impact on electric vehicles, AI development, and consumer electronics.",
        complexity=ComplexityLevel.COMPLEX,
        expected_sources=10,
        requires_current_info=True,
        domain="technology_geopolitics",
        max_time_seconds=60,
        query_type="research"
    ),
    EvalQuery(
        id=34,
        query="Provide an in-depth analysis of the space economy revolution, including commercial space ventures, satellite constellations for global internet, space tourism developments, lunar and Mars exploration plans, space mining prospects, international space law challenges, the role of SpaceX and other private companies, and projections for the space economy's growth through 2035.",
        complexity=ComplexityLevel.COMPLEX,
        expected_sources=10,
        requires_current_info=True,
        domain="space_technology",
        max_time_seconds=60,
        query_type="research"
    ),
    EvalQuery(
        id=35,
        query="Create a comprehensive report on the global energy transition, analyzing the shift from fossil fuels to renewables, the role of nuclear power including SMRs, energy storage solutions, grid modernization challenges, hydrogen economy developments, the impact on oil-producing nations, investment flows, policy mechanisms, and realistic timelines for achieving net-zero emissions in major economies.",
        complexity=ComplexityLevel.COMPLEX,
        expected_sources=10,
        requires_current_info=True,
        domain="energy_environment",
        max_time_seconds=60,
        query_type="research"
    ),
    EvalQuery(
        id=36,
        query="Analyze the biotechnology revolution in medicine, covering gene therapy advances, personalized medicine using genomics, organ regeneration and 3D bioprinting, the microbiome's role in health, aging research and longevity treatments, AI-driven drug discovery, regulatory challenges, ethical considerations, cost implications for healthcare systems, and projections for how medicine will transform by 2040.",
        complexity=ComplexityLevel.COMPLEX,
        expected_sources=10,
        requires_current_info=True,
        domain="biotechnology_medicine",
        max_time_seconds=60,
        query_type="research"
    ),
    EvalQuery(
        id=37,
        query="Provide a detailed assessment of the global financial system's digital transformation, including central bank digital currencies, cryptocurrency adoption and regulation, DeFi ecosystem growth, traditional banking disruption, cross-border payment innovations, financial inclusion initiatives, cybersecurity challenges, the role of stablecoins, NFTs and tokenization of assets, and predictions for the future of money and banking.",
        complexity=ComplexityLevel.COMPLEX,
        expected_sources=10,
        requires_current_info=True,
        domain="finance_technology",
        max_time_seconds=60,
        query_type="research"
    ),
    EvalQuery(
        id=38,
        query="Create a comprehensive analysis of quantum computing's progress toward practical applications, covering current qubit technologies, error correction breakthroughs, quantum algorithms development, the quantum advantage demonstrations, major players (IBM, Google, IonQ, etc.), quantum networking and cryptography, potential impacts on drug discovery, materials science, and cryptography, timeline for fault-tolerant quantum computers, and investment trends in the quantum sector.",
        complexity=ComplexityLevel.COMPLEX,
        expected_sources=10,
        requires_current_info=True,
        domain="quantum_technology",
        max_time_seconds=60,
        query_type="research"
    ),
    EvalQuery(
        id=39,
        query="Analyze the future of work and automation, examining how AI and robotics are transforming industries, which jobs are most at risk and which are emerging, reskilling and education initiatives, the four-day workweek experiments, remote work's permanent changes to business, the gig economy evolution, universal basic income debates, demographic shifts impact on labor markets, and policy responses to technological unemployment across different countries.",
        complexity=ComplexityLevel.COMPLEX,
        expected_sources=10,
        requires_current_info=True,
        domain="economics_society",
        max_time_seconds=60,
        query_type="research"
    ),
    EvalQuery(
        id=40,
        query="Provide an in-depth report on global food security and agricultural innovation, covering climate change impacts on crop yields, precision agriculture and AI in farming, vertical farming and alternative proteins, genetic modification and CRISPR in crops, water scarcity solutions, supply chain resilience, food waste reduction technologies, the role of insects and lab-grown meat, policy approaches to ensure food security, and projections for feeding 10 billion people by 2050.",
        complexity=ComplexityLevel.COMPLEX,
        expected_sources=10,
        requires_current_info=True,
        domain="agriculture_environment",
        max_time_seconds=60,
        query_type="research"
    ),
]

def get_queries_by_complexity(complexity: ComplexityLevel) -> List[EvalQuery]:
    """Get all queries of a specific complexity level"""
    return [q for q in EVALUATION_QUERIES if q.complexity == complexity]

def get_queries_by_type(query_type: str) -> List[EvalQuery]:
    """Get all queries of a specific type (qa or research)"""
    return [q for q in EVALUATION_QUERIES if q.query_type == query_type]

def get_query_by_id(query_id: int) -> Optional[EvalQuery]:
    """Get a specific query by ID"""
    return next((q for q in EVALUATION_QUERIES if q.id == query_id), None)

def get_research_queries() -> List[EvalQuery]:
    """Get all deep research queries requiring comprehensive reports"""
    return [q for q in EVALUATION_QUERIES if q.query_type == "research"]

def get_qa_queries() -> List[EvalQuery]:
    """Get all Q&A style queries"""
    return [q for q in EVALUATION_QUERIES if q.query_type == "qa"]

def to_pandas() -> pd.DataFrame:
    """Convert evaluation queries to pandas DataFrame for analysis"""
    data = []
    for query in EVALUATION_QUERIES:
        data.append({
            'id': query.id,
            'query': query.query,
            'complexity': query.complexity.value,
            'expected_sources': query.expected_sources,
            'requires_current_info': query.requires_current_info,
            'domain': query.domain,
            'max_time_seconds': query.max_time_seconds,
            'query_type': query.query_type
        })
    
    return pd.DataFrame(data)

def to_csv(filepath: str) -> None:
    """Export evaluation dataset as CSV file for Jupyter notebook import"""
    df = to_pandas()
    df.to_csv(filepath, index=False)
    print(f"Dataset exported to {filepath}")

def to_arize_format() -> pd.DataFrame:
    """Convert to Arize Phoenix compatible format for evaluations"""
    df = to_pandas()
    
    # Rename columns for Arize compatibility
    arize_df = df.rename(columns={
        'query': 'input',
        'id': 'query_id'
    })
    
    # Add placeholder columns for evaluation workflow
    arize_df['output'] = ''  # Agent responses will go here
    arize_df['reference'] = ''  # Reference answers (if available)
    arize_df['evaluation_score'] = None  # Evaluation results
    arize_df['evaluation_label'] = ''  # Evaluation labels
    arize_df['evaluation_explanation'] = ''  # Evaluation explanations
    
    return arize_df

def create_evaluation_template() -> pd.DataFrame:
    """Create template DataFrame for storing agent responses and evaluations"""
    template_df = to_arize_format()
    
    # Add additional columns for comprehensive evaluation tracking
    template_df['agent_response_time_ms'] = None
    template_df['tokens_used'] = None
    template_df['model_used'] = ''
    template_df['sources_found'] = None
    template_df['citations_count'] = None
    template_df['hallucination_score'] = None
    template_df['relevance_score'] = None
    template_df['qa_correctness_score'] = None
    template_df['response_length'] = None
    template_df['timestamp'] = pd.NaT
    
    return template_df

def print_dataset_summary():
    """Print summary statistics of the evaluation dataset"""
    print("Evaluation Dataset Summary")
    print("=" * 60)
    
    by_complexity = {}
    by_domain = {}
    by_type = {}
    
    for query in EVALUATION_QUERIES:
        # Count by complexity
        complexity_key = query.complexity.value
        by_complexity[complexity_key] = by_complexity.get(complexity_key, 0) + 1
        
        # Count by domain
        by_domain[query.domain] = by_domain.get(query.domain, 0) + 1
        
        # Count by type
        by_type[query.query_type] = by_type.get(query.query_type, 0) + 1
    
    print("By Query Type:")
    for query_type, count in by_type.items():
        queries = [q for q in EVALUATION_QUERIES if q.query_type == query_type]
        complexity_breakdown = {}
        for q in queries:
            complexity_breakdown[q.complexity.value] = complexity_breakdown.get(q.complexity.value, 0) + 1
        print(f"  {query_type}: {count} queries")
        for complexity, c_count in complexity_breakdown.items():
            print(f"    - {complexity}: {c_count}")
    
    print("\nBy Complexity:")
    for complexity, count in by_complexity.items():
        print(f"  {complexity}: {count} queries")
    
    print("\nBy Domain (top 10):")
    sorted_domains = sorted(by_domain.items(), key=lambda x: x[1], reverse=True)[:10]
    for domain, count in sorted_domains:
        print(f"  {domain}: {count} queries")
    
    print(f"\nTotal queries: {len(EVALUATION_QUERIES)}")
    print(f"Requiring current info: {sum(1 for q in EVALUATION_QUERIES if q.requires_current_info)}")
    print(f"Average expected sources: {sum(q.expected_sources for q in EVALUATION_QUERIES) / len(EVALUATION_QUERIES):.1f}")
    print(f"Research queries (2-page reports): {len(get_research_queries())}")
    print(f"Q&A queries: {len(get_qa_queries())}")

if __name__ == "__main__":
    print_dataset_summary()
    
    # Show example queries from each category
    print("\n" + "=" * 60)
    print("EXAMPLE QUERIES FROM EACH CATEGORY")
    print("=" * 60)
    
    print("\n📝 Easy Q&A Example:")
    easy_qa = get_query_by_id(1)
    print(f"  Q: {easy_qa.query}")
    print(f"  Sources: {easy_qa.expected_sources}, Time: {easy_qa.max_time_seconds}s")
    
    print("\n📊 Medium Q&A Example:")
    medium_qa = get_query_by_id(11)
    print(f"  Q: {medium_qa.query}")
    print(f"  Sources: {medium_qa.expected_sources}, Time: {medium_qa.max_time_seconds}s")
    
    print("\n🔬 Complex Q&A Example:")
    complex_qa = get_query_by_id(21)
    print(f"  Q: {complex_qa.query[:100]}...")
    print(f"  Sources: {complex_qa.expected_sources}, Time: {complex_qa.max_time_seconds}s")
    
    print("\n📚 Deep Research Example:")
    research = get_query_by_id(31)
    print(f"  Q: {research.query[:150]}...")
    print(f"  Sources: {research.expected_sources}, Time: {research.max_time_seconds}s")
    
    # Show usage examples for DataFrame/CSV export
    print("\n" + "=" * 60)
    print("USAGE EXAMPLES")
    print("=" * 60)
    
    print("\n💻 For Jupyter Notebook Analysis:")
    print("  from evaluation.evaluation_dataset import to_pandas, to_csv")
    print("  df = to_pandas()")
    print("  df.head()")
    print("  # or export to CSV:")
    print("  to_csv('evaluation_queries.csv')")
    
    print("\n🔍 For Arize Phoenix Evaluations:")
    print("  from evaluation.evaluation_dataset import to_arize_format, create_evaluation_template")
    print("  arize_df = to_arize_format()")
    print("  # Fill agent responses in 'output' column, then:")
    print("  # phoenix.log_evaluations(arize_df)")
    
    print("\n📊 For Full Evaluation Tracking:")
    print("  template = create_evaluation_template()")
    print("  # Fill with agent responses and evaluation results")
    print("  template.to_csv('agent_evaluation_results.csv')")
    
    # Demo DataFrame structure
    print(f"\n📈 Dataset Structure Preview:")
    df_preview = to_pandas().head(3)
    print(f"  Shape: {df_preview.shape}")
    print(f"  Columns: {list(df_preview.columns)}")
    
    print(f"\n🎯 Arize Format Preview:")
    arize_preview = to_arize_format().head(1)
    print(f"  Shape: {arize_preview.shape}")
    print(f"  Columns: {list(arize_preview.columns)}")