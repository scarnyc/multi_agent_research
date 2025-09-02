import pytest
import asyncio
from typing import Generator
import os
from dotenv import load_dotenv

load_dotenv()

@pytest.fixture(scope="session")
def event_loop() -> Generator:
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def api_key():
    return os.getenv("OPENAI_API_KEY")

@pytest.fixture
def mock_search_results():
    return [
        {
            "title": "Test Result 1",
            "url": "https://example.com/1",
            "snippet": "This is a test search result",
            "score": 0.95
        },
        {
            "title": "Test Result 2", 
            "url": "https://example.com/2",
            "snippet": "Another test search result",
            "score": 0.87
        }
    ]