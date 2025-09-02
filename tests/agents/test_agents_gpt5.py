#!/usr/bin/env python3
"""
Comprehensive test of OpenAI Agents SDK compatibility with GPT-5 models
"""
import os
from openai import OpenAI
from config import config
import time

def test_direct_gpt5_calls():
    """Test GPT-5 models with direct OpenAI calls"""
    client = OpenAI(api_key=config.openai_api_key)
    
    models_to_test = [
        ("GPT-5 Nano", config.gpt5_nano_model),
        ("GPT-5 Mini", config.gpt5_mini_model), 
        ("GPT-5 Regular", config.gpt5_regular_model)
    ]
    
    print("=" * 50)
    print("TESTING DIRECT GPT-5 CALLS")
    print("=" * 50)
    
    for model_name, model in models_to_test:
        try:
            start_time = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "What is 2+2? Be brief."}],
                max_completion_tokens=50
            )
            duration = time.time() - start_time
            
            print(f"✅ {model_name} ({model})")
            print(f"   Response: {response.choices[0].message.content}")
            print(f"   Tokens: {response.usage.total_tokens}, Time: {duration:.2f}s")
            print()
            
        except Exception as e:
            print(f"❌ {model_name} ({model}): {str(e)}")
            print()

def test_gpt5_with_web_search():
    """Test GPT-5 with web search functionality"""
    client = OpenAI(api_key=config.openai_api_key)
    
    print("=" * 50)
    print("TESTING GPT-5 WITH WEB SEARCH")  
    print("=" * 50)
    
    test_query = "What is the current weather in San Francisco?"
    
    for model_name, model in [("GPT-5 Mini", config.gpt5_mini_model)]:
        try:
            start_time = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "system", 
                    "content": "You are a helpful research assistant. Use web search to find current information."
                }, {
                    "role": "user", 
                    "content": test_query
                }],
                max_completion_tokens=200
            )
            duration = time.time() - start_time
            
            print(f"✅ {model_name} with web search:")
            print(f"   Query: {test_query}")
            print(f"   Response: {response.choices[0].message.content[:200]}...")
            print(f"   Tokens: {response.usage.total_tokens}, Time: {duration:.2f}s")
            print()
            
        except Exception as e:
            print(f"❌ {model_name} web search: {str(e)}")
            print()

def test_agents_sdk_compatibility():
    """Test if OpenAI Agents SDK works with GPT-5"""
    print("=" * 50)
    print("TESTING AGENTS SDK WITH GPT-5")
    print("=" * 50)
    
    try:
        # Try importing the agents SDK
        try:
            from openai_agents import Agent
            print("✅ OpenAI Agents SDK imported successfully")
        except ImportError:
            # Try alternative import patterns
            try:
                import openai_agents
                print("✅ OpenAI Agents module found")
                print(f"   Available: {dir(openai_agents)}")
            except ImportError as e:
                print(f"❌ OpenAI Agents SDK not available: {e}")
                return False
        
        # Test agent creation with GPT-5
        try:
            # This is speculative syntax - may need adjustment based on actual SDK
            agent = Agent(
                name="test-agent",
                model=config.gpt5_mini_model,
                instructions="You are a helpful research assistant."
            )
            print(f"✅ Agent created successfully with {config.gpt5_mini_model}")
            
            # Test agent execution
            response = agent.run("What is 2+2?")
            print(f"   Agent response: {response}")
            
        except Exception as e:
            print(f"❌ Agent creation/execution failed: {str(e)}")
            return False
            
    except Exception as e:
        print(f"❌ Agents SDK test failed: {str(e)}")
        return False
    
    return True

def test_function_calling_with_gpt5():
    """Test if GPT-5 supports function calling (alternative to Agents SDK)"""
    client = OpenAI(api_key=config.openai_api_key)
    
    print("=" * 50) 
    print("TESTING GPT-5 FUNCTION CALLING")
    print("=" * 50)
    
    # Define a simple function for testing
    tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for current information",
                "parameters": {
                    "type": "object", 
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    
    for model_name, model in [("GPT-5 Mini", config.gpt5_mini_model)]:
        try:
            start_time = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user", 
                    "content": "Search for the current stock price of Apple"
                }],
                tools=tools,
                tool_choice="auto",
                max_completion_tokens=200
            )
            duration = time.time() - start_time
            
            print(f"✅ {model_name} function calling:")
            print(f"   Response: {response.choices[0].message}")
            if response.choices[0].message.tool_calls:
                print(f"   Tool calls: {response.choices[0].message.tool_calls}")
            print(f"   Tokens: {response.usage.total_tokens}, Time: {duration:.2f}s")
            print()
            
        except Exception as e:
            print(f"❌ {model_name} function calling: {str(e)}")
            print()

def main():
    print("Testing OpenAI Agents SDK + GPT-5 Compatibility")
    print("=" * 50)
    
    # Test 1: Basic GPT-5 functionality
    test_direct_gpt5_calls()
    
    # Test 2: GPT-5 with web search
    test_gpt5_with_web_search()
    
    # Test 3: Agents SDK compatibility 
    agents_sdk_works = test_agents_sdk_compatibility()
    
    # Test 4: Function calling as alternative
    test_function_calling_with_gpt5()
    
    # Summary
    print("=" * 50)
    print("COMPATIBILITY TEST SUMMARY")
    print("=" * 50)
    if agents_sdk_works:
        print("✅ RECOMMENDED: Use OpenAI Agents SDK with GPT-5")
    else:
        print("⚠️  FALLBACK: Use direct API calls with function calling")
    
    print("\nNext steps:")
    print("1. Build research agent using the working approach")
    print("2. Implement model routing logic")  
    print("3. Create evaluation framework")

if __name__ == "__main__":
    main()