# 📊 Multi-Agent Evaluation Notebook

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install jupyter ipywidgets matplotlib seaborn pandas notebook
```

### 2. Launch Notebook
```bash
# Option 1: Use the launcher script
python3 launch_notebook.py

# Option 2: Direct Jupyter launch
cd evaluation
jupyter notebook multi_agent_evaluation_notebook.ipynb
```

### 3. Enable Widget Extensions (if needed)
```bash
jupyter nbextension enable --py widgetsnbextension
```

## 📋 Features

### 🎮 Interactive Controls
- **Query Limits**: Control how many queries to evaluate
- **Complexity Filtering**: Choose which model types to test
- **Domain Filtering**: Focus on specific research domains
- **Real-time Progress**: Live progress tracking during evaluation

### 📊 Comprehensive Analysis
- **Performance Metrics**: Success rate, execution time, token usage
- **Visual Analytics**: Charts and graphs for result analysis
- **Complexity Breakdown**: Performance by model complexity level
- **Domain Analysis**: Results grouped by research domain

### 🔥 Phoenix Integration
- **Real-time Tracing**: OpenTelemetry spans for all agent interactions
- **Session Management**: Organized evaluation sessions
- **Observability UI**: Access Phoenix UI at http://localhost:6006

### 💾 Export Capabilities
- **CSV Export**: Tabular data for spreadsheet analysis
- **JSON Export**: Structured data for programmatic use
- **Visualization Export**: Save charts and graphs

### 🧪 Custom Testing
- **Interactive Query Testing**: Test individual queries
- **Real-time Results**: Immediate feedback on query performance
- **Debug Interface**: Detailed execution information

## 📈 Usage Workflow

1. **Initialize System**: Run setup cells to start multi-agent system
2. **Configure Phoenix**: Start evaluation session with tracing
3. **Set Parameters**: Use interactive widgets to configure evaluation
4. **Run Evaluation**: Execute batch evaluation with progress tracking
5. **Analyze Results**: Review comprehensive metrics and visualizations
6. **Export Data**: Save results for further analysis
7. **Custom Testing**: Test specific queries interactively

## 🔧 Troubleshooting

### Common Issues

**Widget not displaying:**
```bash
jupyter nbextension enable --py widgetsnbextension
jupyter notebook --generate-config
```

**Module import errors:**
- Ensure all requirements are installed: `pip install -r requirements.txt`
- Check Python environment: `which python3`

**Phoenix connection issues:**
- Phoenix integration gracefully degrades if unavailable
- Check environment variables: `PHOENIX_API_KEY`, `PHOENIX_ENDPOINT`

**Evaluation timeouts:**
- Adjust `request_timeout_seconds` in settings
- Reduce batch size for large evaluations
- Monitor API rate limits

## 📚 Evaluation Dataset

The notebook uses `EVALUATION_QUERIES` from `evaluation_dataset.py`:

- **40+ queries** across different complexity levels
- **Multiple domains**: Technology, Biology, History, etc.
- **Varied types**: Q&A, Research, Analysis
- **Current info flags**: Some queries require recent information

## 🎯 Best Practices

### For Reliable Evaluations
1. **Start Small**: Begin with 5-10 queries to test functionality
2. **Monitor Resources**: Watch token usage and execution time
3. **Use Filters**: Focus on specific complexity levels or domains
4. **Regular Exports**: Save results frequently to avoid data loss

### For Analysis
1. **Compare Across Runs**: Track performance over time
2. **Analyze by Complexity**: Understand model routing effectiveness
3. **Domain-Specific Insights**: Look for domain performance patterns
4. **Quality vs Speed**: Balance response quality with execution time

## 🔗 Integration Points

### With Main System
- Uses `MultiAgentResearchSystem` for consistent behavior
- Leverages same configuration from `config/settings.py`
- Compatible with all CLI and API interfaces

### With Phoenix
- Automatic OpenTelemetry instrumentation
- Session-based trace organization
- Real-time observability dashboard

### With Evaluation Framework
- Integrates with existing evaluation datasets
- Compatible with batch evaluation runner
- Supports custom quality metrics

## 🚀 Advanced Usage

### Batch Processing
```python
# Process multiple queries concurrently
results = await system.batch_process_queries(
    queries=custom_query_list,
    max_concurrent=3
)
```

### Custom Metrics
```python
# Add custom evaluation metrics
def custom_quality_metric(response, expected):
    # Your custom evaluation logic
    return score
```

### Phoenix Custom Traces
```python
# Add custom spans for detailed tracing
with tracer.start_as_current_span("custom_analysis") as span:
    # Your analysis code
    span.set_attribute("custom_metric", value)
```

---

🎉 **Ready to evaluate your multi-agent system comprehensively!**