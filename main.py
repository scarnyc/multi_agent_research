#!/usr/bin/env python3
"""
Multi-Agent Research System - Main Entry Point

This is the primary command-line interface for the multi-agent research system.
Provides access to both simple and complex research capabilities.

Usage:
    python main.py --help                    # Show help
    python main.py simple "What is AI?"      # Simple research query
    python main.py multi "Analyze AI trends" # Multi-agent research
    python main.py eval                      # Run evaluation suite
    python main.py setup                     # Setup Phoenix integration
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent))

from config.settings import settings

def print_banner():
    """Print system banner"""
    print("🤖 Multi-Agent Research System")
    print("=" * 40)
    print("Production-ready AI research with intelligent model routing")
    print()

def print_system_info():
    """Print system configuration info"""
    print("📊 System Configuration:")
    print(f"  GPT-5 Models: {settings.gpt5_nano_model}, {settings.gpt5_mini_model}, {settings.gpt5_regular_model}")
    print(f"  Responses API: {'Enabled' if settings.use_responses_api else 'Disabled'}")
    print(f"  Phoenix Tracing: {'Enabled' if settings.phoenix_api_key else 'Disabled'}")
    print()

async def run_simple_research(query: str):
    """Run simple research using the single-agent system"""
    print("🔍 Running Simple Research Agent...")
    print(f"Query: {query}")
    print()
    
    try:
        from agents.research_agent import ResearchAgent
        
        agent = ResearchAgent()
        result = agent.research(query)
        
        print("📋 Results:")
        print(f"  Model Used: {result.model_used}")
        print(f"  Complexity: {result.complexity_detected.value}")
        print(f"  Execution Time: {result.execution_time:.2f}s")
        print(f"  Token Usage: {result.token_usage['total_tokens']}")
        print(f"  Sources Found: {len(result.sources)}")
        print()
        print("📄 Response:")
        print(result.response)
        print()
        if result.sources:
            print("🔗 Sources:")
            for i, source in enumerate(result.sources, 1):
                print(f"  {i}. {source}")
        
    except Exception as e:
        print(f"❌ Error in simple research: {str(e)}")
        return False
    
    return True

async def run_multi_agent_research(query: str):
    """Run complex research using the multi-agent system"""
    print("🏗️ Running Multi-Agent Research System...")
    print(f"Query: {query}")
    print()
    
    try:
        from agents.multi_agents import initialize_system
        
        # Initialize the complete system
        system = initialize_system()
        
        # Process the query
        result = await system.process_query(query)
        
        print("📋 Results:")
        print(f"  Status: {result.get('status', 'unknown')}")
        print(f"  Session ID: {result.get('session_id', 'N/A')}")
        print(f"  Agents Used: {', '.join(result.get('agents_used', []))}")
        print(f"  Total Tokens: {result.get('total_tokens', 'N/A')}")
        print(f"  Execution Time: {result.get('execution_time', 'N/A'):.2f}s" if result.get('execution_time') else "  Execution Time: N/A")
        print()
        print("📄 Response:")
        print(result.get('response', 'No response available'))
        print()
        
        citations = result.get('citations', [])
        if citations:
            print("📚 Citations:")
            for i, citation in enumerate(citations, 1):
                if hasattr(citation, 'url'):
                    print(f"  {i}. {citation.url}")
                else:
                    print(f"  {i}. {str(citation)}")
        
    except Exception as e:
        print(f"❌ Error in multi-agent research: {str(e)}")
        return False
    
    return True

async def run_evaluation():
    """Run the evaluation suite"""
    print("📊 Running Evaluation Suite...")
    print()
    
    try:
        from evaluation.evaluation_dataset import print_dataset_summary
        
        print("📈 Evaluation Dataset Summary:")
        print_dataset_summary()
        print()
        
        print("🚀 To run full evaluation, use the Jupyter notebook:")
        print("  cd evaluation/")
        print("  jupyter notebook agent_evaluation_notebook.ipynb")
        print()
        
        print("🔥 To setup Phoenix monitoring:")
        print("  python evaluation/setup_phoenix_mcp.py")
        
    except Exception as e:
        print(f"❌ Error running evaluation: {str(e)}")
        return False
    
    return True

def setup_phoenix():
    """Setup Phoenix integration"""
    print("🔥 Setting up Phoenix Integration...")
    print()
    
    try:
        print("📋 Phoenix setup options:")
        print("  1. Manual setup: python evaluation/setup_phoenix_mcp.py")
        print("  2. Quick start: Set PHOENIX_API_KEY in .env file")
        print("  3. Local Phoenix: phoenix serve")
        print()
        print("📚 For detailed setup instructions, see:")
        print("  - README.md section on Phoenix integration")
        print("  - evaluation/setup_phoenix_mcp.py for automated setup")
        
    except Exception as e:
        print(f"❌ Error in Phoenix setup: {str(e)}")
        return False
    
    return True

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Research System",
        epilog="Examples:\n"
               "  python main.py simple 'What is machine learning?'\n"
               "  python main.py multi 'Analyze climate change trends'\n"
               "  python main.py eval\n"
               "  python main.py setup",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Simple research command
    simple_parser = subparsers.add_parser('simple', help='Run simple research query')
    simple_parser.add_argument('query', type=str, help='Research query to process')
    
    # Multi-agent research command
    multi_parser = subparsers.add_parser('multi', help='Run multi-agent research query')
    multi_parser.add_argument('query', type=str, help='Complex research query to process')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('eval', help='Run evaluation suite')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup Phoenix integration')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show system information')
    
    args = parser.parse_args()
    
    print_banner()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'info':
        print_system_info()
        return
    
    # Validate environment
    if not settings.openai_api_key:
        print("❌ Error: OPENAI_API_KEY not found in environment")
        print("   Please set your OpenAI API key in .env file or environment variables")
        return
    
    # Run the appropriate command
    try:
        if args.command == 'simple':
            asyncio.run(run_simple_research(args.query))
        elif args.command == 'multi':
            asyncio.run(run_multi_agent_research(args.query))
        elif args.command == 'eval':
            asyncio.run(run_evaluation())
        elif args.command == 'setup':
            setup_phoenix()
        else:
            print(f"❌ Unknown command: {args.command}")
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        if '--debug' in sys.argv:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()