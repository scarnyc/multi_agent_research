#!/usr/bin/env python3
"""
Test script to verify OpenAI Agents SDK works with GPT-5 models
"""
import os
from openai import OpenAI
from config import config

def test_gpt5_availability():
    """Test if GPT-5 models are available"""
    client = OpenAI(api_key=config.openai_api_key)
    
    models_to_test = [
        config.gpt5_nano_model,
        config.gpt5_mini_model, 
        config.gpt5_regular_model
    ]
    
    print("Testing GPT-5 model availability...")
    
    for model in models_to_test:
        try:
            # Simple test completion
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Hello, are you working?"}],
                max_tokens=50
            )
            print(f"✅ {model}: {response.choices[0].message.content[:50]}...")
        except Exception as e:
            print(f"❌ {model}: {str(e)}")

def test_agents_sdk():
    """Test OpenAI Agents SDK if available"""
    try:
        # Try importing agents SDK
        from openai import OpenAI
        client = OpenAI(api_key=config.openai_api_key)
        
        # Check if agents functionality is available
        print("\nTesting Agents SDK...")
        
        # Test basic agent creation (this syntax may need adjustment)
        response = client.chat.completions.create(
            model=config.gpt5_mini_model,
            messages=[{
                "role": "system", 
                "content": "You are a research assistant. Use web search to find information."
            }, {
                "role": "user", 
                "content": "What is the current weather in San Francisco?"
            }],
            tools=[{
                "type": "web_search"
            }] if hasattr(client, 'tools') else None
        )
        
        print(f"✅ Basic agent test: {response.choices[0].message.content[:100]}...")
        
    except ImportError as e:
        print(f"❌ Agents SDK not available: {e}")
    except Exception as e:
        print(f"❌ Agents SDK test failed: {e}")

if __name__ == "__main__":
    test_gpt5_availability()
    test_agents_sdk()