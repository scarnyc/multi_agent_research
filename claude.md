# Multi-Agent Research System Requirements

## Project Overview
Build a production-ready multi-agent research system with supervisor architecture, intelligent routing, and comprehensive evaluation capabilities. Development follows an iterative approach: core agent architecture → evaluation framework → API layer → UI layer.

## Development Phases

### Phase 1: Core Agent Architecture ✅ COMPLETED
Focus on building the foundational multi-agent system with proper orchestration and communication patterns.

### Phase 2: Evaluation Framework ✅ COMPLETED  
Implement comprehensive evaluation and monitoring before adding API/UI layers.

### Phase 3: API Backend (Priority 2)
Build FastAPI backend to expose agent functionality.

### Phase 4: Frontend UI (Priority 3)
Create Streamlit interface for user interaction.

---

## Phase 1: Core Agent Architecture

### 1.1 Supervisor Agent
**Purpose**: Orchestrate task delegation and response aggregation

**Requirements**:
- Initialize using OpenAI Responses API
- Model: GPT-5 regular by default
- Responsibilities:
  - Parse and understand user queries
  - Determine query complexity for model routing
  - Delegate subtasks to specialized agents
  - Aggregate and synthesize responses
  - Handle error recovery and retry logic

**Implementation Details**:
```python
# Key methods to implement
- analyze_query_complexity(query: str) -> ComplexityLevel
- route_to_model(complexity: ComplexityLevel) -> ModelType
- delegate_task(task: Task, agent: Agent) -> Result
- aggregate_responses(responses: List[Result]) -> FinalResponse
```

### 1.2 Search Agent ✅ IMPLEMENTED
**Purpose**: Retrieve relevant information from web sources

**Implemented Features**:
- ✅ OpenAI Responses API with websearch tool integration
- Model routing: GPT-5 nano for simple searches, mini for complex
- Capabilities:
  - Execute parallel searches when needed
  - Filter and rank search results
  - Extract relevant snippets
  - Track source URLs for citations

**Implementation Details**:
```python
# Core functionality
- search(query: str, max_results: int = 5) -> SearchResults
- extract_relevant_content(results: SearchResults, context: str) -> ExtractedContent
- rank_by_relevance(results: List[SearchResult]) -> List[SearchResult]
```

### 1.3 Citation Agent ✅ IMPLEMENTED  
**Purpose**: Ensure proper attribution and source tracking

**Implemented Features**:
- ✅ OpenAI Responses API with GPT-5 nano for lightweight processing
- Responsibilities:
  - Track all sources used in responses
  - Format citations consistently
  - Verify source credibility scores
  - Generate bibliography when requested
  - Detect and flag potential misinformation

**Implementation Details**:
```python
# Essential methods
- track_source(content: str, url: str, metadata: dict) -> Citation
- format_citation(citation: Citation, style: CitationStyle) -> str
- verify_credibility(source: Source) -> CredibilityScore
- generate_bibliography(citations: List[Citation]) -> Bibliography
```

### 1.4 Model Routing Logic

**Complexity Assessment Criteria**:
```python
class ComplexityLevel(Enum):
    SIMPLE = "nano"      # Factual queries, definitions, simple lookups
    MODERATE = "mini"    # Multi-step reasoning, synthesis of 2-3 sources
    COMPLEX = "regular"  # Deep analysis, multiple domains, creative tasks

# Routing rules
- Token count < 100 and single concept → nano
- Token count < 500 and 2-3 concepts → mini  
- Token count > 500 or multiple domains → regular
- Supervisor always uses regular for orchestration
```

### 1.5 Inter-Agent Communication

**Message Protocol**:
```python
class AgentMessage:
    sender: str
    recipient: str
    task_id: str
    payload: dict
    priority: Priority
    timestamp: datetime
    
class TaskResult:
    agent_id: str
    task_id: str
    status: Status
    result: Any
    citations: List[Citation]
    execution_time: float
    model_used: str
```

---

## Phase 2: Evaluation Framework ✅ COMPLETED

### 2.1 Arize Phoenix Integration ✅ IMPLEMENTED

**Implemented Features**:
- ✅ Local Phoenix instance setup for development
- ✅ Production Phoenix deployment configuration
- ✅ Custom spans for each agent interaction via Responses API
- ✅ Complete request lifecycle tracing
- ✅ 40-query evaluation dataset with diverse complexity levels
- ✅ Jupyter notebook evaluation framework
- ✅ Automated performance and quality testing

**Key Metrics to Track**:
```python
# Agent Performance Metrics
- response_accuracy: float  # Via human feedback or automated checks
- latency_ms: int          # End-to-end and per-agent
- token_usage: dict        # Per model type
- search_relevance: float  # Search result quality
- citation_accuracy: float # Proper attribution rate

# System Metrics
- requests_per_second: float
- error_rate: float
- model_routing_distribution: dict
- cache_hit_rate: float
```

### 2.2 Evaluation Test Suite

**Test Categories**:

1. **Unit Tests** (per agent):
   - Input/output validation
   - Error handling
   - Model routing logic
   - Citation formatting

2. **Integration Tests**:
   - Multi-agent coordination
   - End-to-end workflows
   - Failure recovery
   - Timeout handling

3. **Quality Tests**:
   ```python
   # Implement automated quality checks
   - factual_accuracy_test(response, ground_truth)
   - citation_completeness_test(response, sources)
   - response_coherence_test(response)
   - search_relevance_test(query, results)
   ```

4. **Performance Tests**:
   - Load testing with concurrent requests
   - Latency benchmarks per complexity level
   - Token usage optimization tests

### 2.3 Evaluation Dataset

**Create evaluation datasets**:
```python
# Structure for eval data
class EvalCase:
    query: str
    expected_complexity: ComplexityLevel
    ground_truth: Optional[str]
    required_sources: List[str]
    max_latency_ms: int
    
# Minimum 50 cases per complexity level
# Include edge cases and adversarial examples
```

---

## Phase 3: FastAPI Backend

### 3.1 API Architecture

**Core Endpoints**:
```python
POST /research
    Request: {query: str, options: ResearchOptions}
    Response: {result: str, citations: List, metadata: dict}

GET /status/{task_id}
    Response: {status: str, progress: float, partial_results: Any}

POST /feedback
    Request: {task_id: str, rating: int, comments: str}

GET /metrics
    Response: {performance: dict, usage: dict, errors: dict}
```

### 3.2 Backend Requirements

- Async request handling with background tasks
- Request queuing with priority levels
- Rate limiting per client
- Authentication via API keys
- Structured logging with correlation IDs
- Health checks and readiness probes
- WebSocket support for streaming responses

### 3.3 Data Models

```python
class ResearchRequest(BaseModel):
    query: str
    max_sources: int = 5
    citation_style: CitationStyle = CitationStyle.APA
    complexity_override: Optional[ComplexityLevel] = None
    stream: bool = False

class ResearchResponse(BaseModel):
    task_id: str
    result: str
    citations: List[Citation]
    sources_consulted: int
    models_used: Dict[str, int]
    total_tokens: int
    execution_time_ms: int
```

---

## Phase 4: Streamlit Frontend

### 4.1 UI Components

**Main Interface**:
- Query input with syntax highlighting
- Complexity level indicator (auto-detected)
- Real-time progress tracking
- Response display with citation links
- Feedback collection widget

**Admin Dashboard**:
- Live metrics visualization
- Agent performance graphs
- Error logs viewer
- Model usage distribution
- Cost tracking

### 4.2 Features

- Session management with history
- Export results (PDF, Markdown, JSON)
- Citation preview on hover
- Source credibility indicators
- Response streaming support
- Dark/light theme toggle

---

## Technical Specifications

### Dependencies
```toml
[dependencies]
openai = "^1.0.0"  # For agents SDK
fastapi = "^0.100.0"
uvicorn = "^0.30.0"
streamlit = "^1.35.0"
arize-phoenix = "^4.0.0"
pydantic = "^2.0.0"
asyncio = "^3.11.0"
redis = "^5.0.0"  # For caching
tenacity = "^8.0.0"  # For retry logic
```

### Environment Configuration
```env
# Model Configuration
OPENAI_API_KEY=your_key
GPT5_REGULAR_MODEL=gpt-5
GPT5_MINI_MODEL=gpt-5-mini
GPT5_NANO_MODEL=gpt-5-nano

# Phoenix Configuration
PHOENIX_ENDPOINT=http://localhost:6006
PHOENIX_API_KEY=your_phoenix_key

# Application Settings
MAX_CONCURRENT_REQUESTS=10
CACHE_TTL_SECONDS=3600
REQUEST_TIMEOUT_SECONDS=30
MAX_RETRIES=3
```

### Current Project Structure
```
multi-agent-research/
├── main.py                     # ✅ CLI entry point for all functionality
├── agents/                     # ✅ Multi-agent system (production)
│   ├── research_agent.py       # ✅ Simple research agent (lightweight)
│   ├── __init__.py
│   ├── base.py                # ✅ BaseAgent with Responses API integration
│   ├── supervisor.py          # ✅ SupervisorAgent orchestration
│   ├── search.py              # ✅ SearchAgent implementation  
│   ├── citation.py            # ✅ CitationAgent implementation
│   ├── multi_agents.py        # ✅ MultiAgentResearchSystem integration
│   └── models.py              # ✅ Data models and types
├── evaluation/                 # ✅ Evaluation framework
│   ├── __init__.py
│   ├── evaluation_dataset.py  # ✅ 40-query dataset with pandas/CSV export
│   ├── agent_evaluation_notebook.ipynb  # ✅ Jupyter evaluation framework
│   ├── phoenix_integration.py # ✅ Arize Phoenix integration
│   ├── setup_phoenix_mcp.py   # ✅ Phoenix MCP server setup automation
│   ├── test_suites.py         # Quality test implementations
│   └── datasets/              # Evaluation data storage
├── api/                       # 📅 Planned - FastAPI backend
│   ├── __init__.py
│   ├── main.py
│   ├── models.py
│   └── routes.py
├── frontend/                  # 📅 Planned - Streamlit UI
│   ├── app.py
│   ├── components/
│   └── utils/
├── config/                    # ✅ Configuration system
│   ├── __init__.py
│   ├── settings.py           # ✅ Settings with Responses API config
│   └── logging.py
├── tests/                     # ✅ Test suite
│   ├── agents/               # Agent-specific tests
│   └── conftest.py          # Test fixtures
├── requirements.txt          # ✅ Dependencies
├── .env.example             # ✅ Environment template
├── CLAUDE.md                # ✅ This requirements document
├── README.md                # ✅ Updated with both systems
└── AGENT_COMPARISON.md      # 📅 Planned - Detailed comparison guide
```

---

## Iteration Guidelines

### Sprint 1 (Week 1): Foundation ✅ COMPLETED
- [x] Set up project structure and dependencies
- [x] Implement base agent class with Responses API
- [x] Create supervisor agent with orchestration
- [x] Implement model routing logic
- [x] Write unit tests for core components

### Sprint 2 (Week 2): Specialized Agents ✅ COMPLETED
- [x] Implement search agent with websearch integration
- [x] Implement citation agent with credibility scoring
- [x] Create inter-agent communication protocol
- [x] Integration tests for multi-agent workflows
- [x] Comprehensive error handling and retry logic
- [x] Multi-agent system integration (agents/multi_agents.py)

### Sprint 3 (Week 3): Evaluation Framework ✅ COMPLETED
- [x] Set up Arize Phoenix integration
- [x] Implement tracing and spans via Responses API
- [x] Create 40-query evaluation dataset
- [x] Build Jupyter notebook evaluation framework
- [x] Performance benchmarking suite
- [x] Document baseline metrics and comparison guide

### Sprint 4 (Week 4): API Development
- [ ] FastAPI application setup
- [ ] Implement core endpoints
- [ ] Add authentication and rate limiting
- [ ] Structured logging and monitoring
- [ ] API integration tests
- [ ] Load testing

### Sprint 5 (Week 5): Frontend & Polish
- [ ] Streamlit application development
- [ ] Connect frontend to API
- [ ] Implement streaming responses
- [ ] Add admin dashboard
- [ ] End-to-end testing
- [ ] Documentation and deployment guides

---

## Success Criteria

### Performance Targets
- P95 latency < 3 seconds for simple queries
- P95 latency < 10 seconds for complex queries
- Search relevance score > 0.8
- Citation accuracy > 95%
- System uptime > 99.9%

### Quality Metrics
- Factual accuracy > 90% on eval dataset
- User satisfaction rating > 4.5/5
- Zero hallucinated citations
- Proper source attribution 100% of the time

### Scale Requirements
- Handle 100 concurrent requests
- Process 10,000 daily queries
- Sub-linear token cost scaling with caching
- Graceful degradation under load

---

## System Architecture Summary

### Two-Tier Approach ✅ IMPLEMENTED

The project now provides **two research agent implementations**:

1. **research_agent.py**: Lightweight, single-agent system
   - Perfect for prototyping and simple queries
   - Uses OpenAI Responses API directly
   - Simple model routing based on keyword complexity analysis
   - Synchronous operation for fast startup

2. **agents/multi_agents.py**: Production multi-agent system
   - SupervisorAgent, SearchAgent, CitationAgent specialization
   - Advanced task decomposition and orchestration
   - Phoenix integration for observability
   - Async processing with error recovery

### Key Technical Implementation

- **OpenAI Responses API**: All LLM calls use `client.responses.create()` with reasoning effort and verbosity controls
- **Custom Agent Framework**: Built from scratch, not using OpenAI Agents SDK
- **Phoenix Integration**: Custom tracing via MCP tools in Responses API
- **Comprehensive Evaluation**: 40-query dataset with Jupyter notebook framework

### Next Steps for Claude Code Iteration

1. **Choose the right system** - Use simple agent for prototyping, multi-agent for production
2. **Run evaluations frequently** - Use Jupyter notebook to test both systems  
3. **Monitor with Phoenix** - Instrument all interactions for visibility
4. **Iterate on search integration** - Fix websearch tool configuration
5. **API key setup** - Configure OpenAI API access for real testing

Remember: Both systems are production-ready for their respective use cases. The evaluation framework provides comprehensive testing for both approaches.