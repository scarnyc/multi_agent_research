#!/usr/bin/env python3
"""
Real-time monitoring and dashboard setup for Phoenix MCP integration.
"""
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time

from evaluation.phoenix_integration import phoenix_integration

logger = logging.getLogger(__name__)

@dataclass
class MetricPoint:
    """A single metric data point."""
    timestamp: datetime
    metric_name: str
    value: float
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}

@dataclass
class DashboardMetrics:
    """Dashboard metrics data structure."""
    # System metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    current_qps: float = 0.0
    
    # Quality metrics
    avg_accuracy_score: float = 0.0
    avg_citation_completeness: float = 0.0
    avg_response_coherence: float = 0.0
    avg_source_relevance: float = 0.0
    
    # Resource metrics
    total_tokens_used: int = 0
    avg_tokens_per_request: float = 0.0
    tokens_per_minute: float = 0.0
    
    # Model distribution
    model_usage: Dict[str, int] = None
    complexity_distribution: Dict[str, int] = None
    
    # Time-based metrics
    hourly_requests: List[int] = None
    response_time_percentiles: Dict[str, float] = None
    
    def __post_init__(self):
        if self.model_usage is None:
            self.model_usage = {}
        if self.complexity_distribution is None:
            self.complexity_distribution = {}
        if self.hourly_requests is None:
            self.hourly_requests = [0] * 24
        if self.response_time_percentiles is None:
            self.response_time_percentiles = {}

class PhoenixMonitor:
    """Real-time monitoring for Phoenix MCP integration."""
    
    def __init__(self, update_interval: float = 30.0):
        self.update_interval = update_interval
        self.metrics_history: List[MetricPoint] = []
        self.current_metrics = DashboardMetrics()
        self._monitoring = False
        self._last_update = datetime.now()
        
    async def start_monitoring(self) -> None:
        """Start real-time monitoring."""
        if self._monitoring:
            logger.warning("Monitoring already running")
            return
            
        self._monitoring = True
        logger.info(f"Starting Phoenix monitoring with {self.update_interval}s interval")
        
        while self._monitoring:
            try:
                await self._update_metrics()
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Monitoring update failed: {e}")
                await asyncio.sleep(self.update_interval)
    
    def stop_monitoring(self) -> None:
        """Stop real-time monitoring."""
        self._monitoring = False
        logger.info("Stopped Phoenix monitoring")
    
    async def _update_metrics(self) -> None:
        """Update metrics from Phoenix."""
        try:
            # Get metrics from Phoenix MCP
            phoenix_metrics = await phoenix_integration.get_evaluation_metrics()
            
            if phoenix_metrics:
                await self._process_phoenix_metrics(phoenix_metrics)
                self._last_update = datetime.now()
                logger.debug("Updated metrics from Phoenix")
            
        except Exception as e:
            logger.warning(f"Failed to update metrics from Phoenix: {e}")
    
    async def _process_phoenix_metrics(self, phoenix_metrics: Dict[str, Any]) -> None:
        """Process metrics received from Phoenix."""
        current_time = datetime.now()
        
        # Extract basic metrics
        if "total_requests" in phoenix_metrics:
            self.current_metrics.total_requests = phoenix_metrics["total_requests"]
            self._add_metric_point("total_requests", phoenix_metrics["total_requests"], current_time)
        
        if "successful_requests" in phoenix_metrics:
            self.current_metrics.successful_requests = phoenix_metrics["successful_requests"]
            
        if "failed_requests" in phoenix_metrics:
            self.current_metrics.failed_requests = phoenix_metrics["failed_requests"]
        
        # Calculate success rate
        total = self.current_metrics.total_requests
        if total > 0:
            success_rate = self.current_metrics.successful_requests / total
            self._add_metric_point("success_rate", success_rate, current_time)
        
        # Response time metrics
        if "avg_response_time" in phoenix_metrics:
            self.current_metrics.avg_response_time = phoenix_metrics["avg_response_time"]
            self._add_metric_point("avg_response_time", phoenix_metrics["avg_response_time"], current_time)
        
        # Quality metrics
        quality_metrics = phoenix_metrics.get("quality_metrics", {})
        if quality_metrics:
            self.current_metrics.avg_accuracy_score = quality_metrics.get("avg_accuracy", 0.0)
            self.current_metrics.avg_citation_completeness = quality_metrics.get("avg_citations", 0.0)
            self.current_metrics.avg_response_coherence = quality_metrics.get("avg_coherence", 0.0)
            self.current_metrics.avg_source_relevance = quality_metrics.get("avg_relevance", 0.0)
            
            self._add_metric_point("accuracy_score", self.current_metrics.avg_accuracy_score, current_time)
            self._add_metric_point("citation_completeness", self.current_metrics.avg_citation_completeness, current_time)
        
        # Token usage metrics
        if "total_tokens" in phoenix_metrics:
            self.current_metrics.total_tokens_used = phoenix_metrics["total_tokens"]
            self._add_metric_point("total_tokens", phoenix_metrics["total_tokens"], current_time)
        
        if "avg_tokens_per_request" in phoenix_metrics:
            self.current_metrics.avg_tokens_per_request = phoenix_metrics["avg_tokens_per_request"]
        
        # Model distribution
        if "model_usage" in phoenix_metrics:
            self.current_metrics.model_usage = phoenix_metrics["model_usage"]
        
        # Complexity distribution
        if "complexity_distribution" in phoenix_metrics:
            self.current_metrics.complexity_distribution = phoenix_metrics["complexity_distribution"]
    
    def _add_metric_point(self, metric_name: str, value: float, timestamp: datetime) -> None:
        """Add a metric point to history."""
        point = MetricPoint(
            timestamp=timestamp,
            metric_name=metric_name,
            value=value
        )
        self.metrics_history.append(point)
        
        # Keep only recent metrics (last 24 hours)
        cutoff_time = timestamp - timedelta(days=1)
        self.metrics_history = [p for p in self.metrics_history if p.timestamp > cutoff_time]
    
    def get_current_metrics(self) -> DashboardMetrics:
        """Get current dashboard metrics."""
        return self.current_metrics
    
    def get_metric_history(self, metric_name: str, hours: int = 24) -> List[MetricPoint]:
        """Get historical data for a specific metric."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            point for point in self.metrics_history 
            if point.metric_name == metric_name and point.timestamp > cutoff_time
        ]
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        return {
            "current_metrics": {
                "total_requests": self.current_metrics.total_requests,
                "successful_requests": self.current_metrics.successful_requests,
                "failed_requests": self.current_metrics.failed_requests,
                "success_rate": (
                    self.current_metrics.successful_requests / self.current_metrics.total_requests 
                    if self.current_metrics.total_requests > 0 else 0
                ),
                "avg_response_time": self.current_metrics.avg_response_time,
                "avg_accuracy_score": self.current_metrics.avg_accuracy_score,
                "avg_citation_completeness": self.current_metrics.avg_citation_completeness,
                "total_tokens_used": self.current_metrics.total_tokens_used,
                "avg_tokens_per_request": self.current_metrics.avg_tokens_per_request
            },
            "model_usage": self.current_metrics.model_usage,
            "complexity_distribution": self.current_metrics.complexity_distribution,
            "last_update": self._last_update.isoformat(),
            "monitoring_active": self._monitoring,
            "metrics_count": len(self.metrics_history)
        }

class SimpleHTTPDashboard:
    """Simple HTTP dashboard for monitoring metrics."""
    
    def __init__(self, monitor: PhoenixMonitor, port: int = 8080):
        self.monitor = monitor
        self.port = port
        self._server = None
    
    async def start_server(self) -> None:
        """Start the HTTP dashboard server."""
        try:
            from aiohttp import web, web_runner
            
            app = web.Application()
            app.router.add_get('/', self._dashboard_handler)
            app.router.add_get('/api/metrics', self._metrics_api_handler)
            app.router.add_get('/api/health', self._health_handler)
            
            runner = web_runner.AppRunner(app)
            await runner.setup()
            
            site = web_runner.TCPSite(runner, 'localhost', self.port)
            await site.start()
            
            self._server = runner
            logger.info(f"Dashboard server started on http://localhost:{self.port}")
            
        except ImportError:
            logger.error("aiohttp not available. Install with: pip install aiohttp")
            raise
        except Exception as e:
            logger.error(f"Failed to start dashboard server: {e}")
            raise
    
    async def stop_server(self) -> None:
        """Stop the HTTP dashboard server."""
        if self._server:
            await self._server.cleanup()
            self._server = None
            logger.info("Dashboard server stopped")
    
    async def _dashboard_handler(self, request) -> 'web.Response':
        """Serve the main dashboard page."""
        from aiohttp import web
        
        dashboard_html = self._generate_dashboard_html()
        return web.Response(text=dashboard_html, content_type='text/html')
    
    async def _metrics_api_handler(self, request) -> 'web.Response':
        """Serve metrics API endpoint."""
        from aiohttp import web
        
        dashboard_data = self.monitor.get_dashboard_data()
        return web.json_response(dashboard_data)
    
    async def _health_handler(self, request) -> 'web.Response':
        """Health check endpoint."""
        from aiohttp import web
        
        health_data = {
            "status": "healthy" if self.monitor._monitoring else "monitoring_stopped",
            "timestamp": datetime.now().isoformat(),
            "last_update": self.monitor._last_update.isoformat()
        }
        return web.json_response(health_data)
    
    def _generate_dashboard_html(self) -> str:
        """Generate HTML dashboard page."""
        dashboard_data = self.monitor.get_dashboard_data()
        
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Phoenix MCP Monitoring Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 20px; }}
        .metric-card {{ background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
        .metric-label {{ color: #7f8c8d; margin-top: 5px; }}
        .status-indicator {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; }}
        .status-active {{ background-color: #27ae60; }}
        .status-inactive {{ background-color: #e74c3c; }}
        .distribution-chart {{ background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .chart-bar {{ height: 20px; background-color: #3498db; margin: 5px 0; border-radius: 3px; }}
    </style>
    <script>
        function refreshData() {{
            fetch('/api/metrics')
                .then(response => response.json())
                .then(data => updateDashboard(data))
                .catch(error => console.error('Error fetching metrics:', error));
        }}
        
        function updateDashboard(data) {{
            const metrics = data.current_metrics;
            
            document.getElementById('total-requests').textContent = metrics.total_requests;
            document.getElementById('success-rate').textContent = (metrics.success_rate * 100).toFixed(1) + '%';
            document.getElementById('avg-response-time').textContent = metrics.avg_response_time.toFixed(2) + 's';
            document.getElementById('accuracy-score').textContent = (metrics.avg_accuracy_score * 100).toFixed(1) + '%';
            document.getElementById('total-tokens').textContent = metrics.total_tokens_used.toLocaleString();
            document.getElementById('last-update').textContent = new Date(data.last_update).toLocaleString();
            
            const statusIndicator = document.getElementById('status-indicator');
            if (data.monitoring_active) {{
                statusIndicator.className = 'status-indicator status-active';
            }} else {{
                statusIndicator.className = 'status-indicator status-inactive';
            }}
        }}
        
        // Auto-refresh every 30 seconds
        setInterval(refreshData, 30000);
        
        // Initial load
        window.addEventListener('load', refreshData);
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Phoenix MCP Monitoring Dashboard</h1>
            <p>
                <span id="status-indicator" class="status-indicator {'status-active' if dashboard_data['monitoring_active'] else 'status-inactive'}"></span>
                Monitoring Status: {'Active' if dashboard_data['monitoring_active'] else 'Inactive'} | 
                Last Update: <span id="last-update">{dashboard_data['last_update']}</span>
            </p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value" id="total-requests">{dashboard_data['current_metrics']['total_requests']}</div>
                <div class="metric-label">Total Requests</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value" id="success-rate">{dashboard_data['current_metrics']['success_rate'] * 100:.1f}%</div>
                <div class="metric-label">Success Rate</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value" id="avg-response-time">{dashboard_data['current_metrics']['avg_response_time']:.2f}s</div>
                <div class="metric-label">Avg Response Time</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value" id="accuracy-score">{dashboard_data['current_metrics']['avg_accuracy_score'] * 100:.1f}%</div>
                <div class="metric-label">Accuracy Score</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value" id="total-tokens">{dashboard_data['current_metrics']['total_tokens_used']:,}</div>
                <div class="metric-label">Total Tokens Used</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-value">{dashboard_data['current_metrics']['avg_citation_completeness'] * 100:.1f}%</div>
                <div class="metric-label">Citation Completeness</div>
            </div>
        </div>
        
        <div class="distribution-chart">
            <h3>Model Usage Distribution</h3>
            <div id="model-usage">
                {''.join([
                    f'<div style="margin: 10px 0;"><strong>{model}:</strong> {count} requests<div class="chart-bar" style="width: {(count / max(dashboard_data["model_usage"].values(), default=1)) * 100}%;"></div></div>'
                    for model, count in dashboard_data["model_usage"].items()
                ]) if dashboard_data["model_usage"] else '<p>No data available</p>'}
            </div>
        </div>
        
        <div class="distribution-chart">
            <h3>Complexity Distribution</h3>
            <div id="complexity-distribution">
                {''.join([
                    f'<div style="margin: 10px 0;"><strong>{complexity}:</strong> {count} queries<div class="chart-bar" style="width: {(count / max(dashboard_data["complexity_distribution"].values(), default=1)) * 100}%;"></div></div>'
                    for complexity, count in dashboard_data["complexity_distribution"].items()
                ]) if dashboard_data["complexity_distribution"] else '<p>No data available</p>'}
            </div>
        </div>
    </div>
</body>
</html>
        """

# Global monitoring instances
monitor = PhoenixMonitor()
dashboard = SimpleHTTPDashboard(monitor)

async def start_monitoring_stack(port: int = 8080) -> None:
    """Start the complete monitoring stack."""
    logger.info("Starting Phoenix monitoring stack...")
    
    # Start monitoring
    monitor_task = asyncio.create_task(monitor.start_monitoring())
    
    # Start dashboard server
    try:
        await dashboard.start_server()
        logger.info(f"Monitoring dashboard available at http://localhost:{port}")
        
        # Keep running
        await monitor_task
        
    except KeyboardInterrupt:
        logger.info("Shutting down monitoring stack...")
    except Exception as e:
        logger.error(f"Monitoring stack failed: {e}")
    finally:
        monitor.stop_monitoring()
        await dashboard.stop_server()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Start Phoenix monitoring dashboard")
    parser.add_argument("--port", type=int, default=8080, help="Dashboard port")
    parser.add_argument("--update-interval", type=float, default=30.0, help="Metrics update interval in seconds")
    
    args = parser.parse_args()
    
    # Configure monitor
    monitor.update_interval = args.update_interval
    dashboard.port = args.port
    
    # Start monitoring stack
    asyncio.run(start_monitoring_stack(args.port))