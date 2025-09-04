#!/usr/bin/env python3
"""
Setup script for Phoenix MCP server integration.
Handles Phoenix server setup, MCP configuration, and initial testing.
"""
import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PhoenixMCPSetup:
    """Setup and configuration manager for Phoenix MCP integration."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.config_dir = self.project_root / "config"
        self.env_file = self.project_root / ".env"
        
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are installed."""
        logger.info("Checking dependencies...")
        
        required_packages = [
            "openai>=1.50.0",
            "arize-phoenix>=4.0.0",
            "pydantic>=2.0.0",
            "httpx>=0.24.0",
            "tenacity>=8.0.0"
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                import importlib
                package_name = package.split(">=")[0].replace("-", "_")
                importlib.import_module(package_name)
                logger.info(f"✓ {package}")
            except ImportError:
                missing_packages.append(package)
                logger.warning(f"✗ {package} - Missing")
        
        if missing_packages:
            logger.error(f"Missing packages: {', '.join(missing_packages)}")
            logger.info("Install missing packages with: pip install " + " ".join(missing_packages))
            return False
        
        logger.info("All dependencies satisfied")
        return True
    
    def setup_phoenix_server(self, method: str = "local") -> bool:
        """Set up Phoenix server (local or remote)."""
        logger.info(f"Setting up Phoenix server using method: {method}")
        
        if method == "local":
            return self._setup_local_phoenix()
        elif method == "docker":
            return self._setup_docker_phoenix()
        elif method == "remote":
            return self._setup_remote_phoenix()
        else:
            logger.error(f"Unknown setup method: {method}")
            return False
    
    def _setup_local_phoenix(self) -> bool:
        """Set up local Phoenix server."""
        try:
            # Check if Phoenix is already running
            import httpx
            
            try:
                response = httpx.get("http://localhost:6006/health", timeout=5.0)
                if response.status_code == 200:
                    logger.info("✓ Phoenix server already running on localhost:6006")
                    return True
            except httpx.RequestError:
                pass
            
            logger.info("Starting local Phoenix server...")
            
            # Start Phoenix server
            import phoenix as px
            
            # Launch Phoenix server
            session = px.launch_app()
            logger.info(f"✓ Phoenix server started at {session.url}")
            
            # Update environment configuration
            self._update_env_config({
                "PHOENIX_BASE_URL": "http://localhost:6006",
                "PHOENIX_MCP_SERVER_URL": "http://localhost:6006/mcp"
            })
            
            return True
            
        except ImportError:
            logger.error("Phoenix package not found. Install with: pip install arize-phoenix")
            return False
        except Exception as e:
            logger.error(f"Failed to start local Phoenix server: {e}")
            return False
    
    def _setup_docker_phoenix(self) -> bool:
        """Set up Phoenix server using Docker."""
        try:
            # Check if Docker is available
            result = subprocess.run(["docker", "--version"], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("Docker not found. Please install Docker first.")
                return False
            
            logger.info("Starting Phoenix server with Docker...")
            
            # Create Phoenix Docker command
            docker_cmd = [
                "docker", "run", "-d",
                "--name", "phoenix-server",
                "-p", "6006:6006",
                "-e", "PHOENIX_PORT=6006",
                "arizephoenix/phoenix:latest"
            ]
            
            # Stop existing container if running
            subprocess.run(["docker", "stop", "phoenix-server"], 
                         capture_output=True)
            subprocess.run(["docker", "rm", "phoenix-server"], 
                         capture_output=True)
            
            # Start new container
            result = subprocess.run(docker_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✓ Phoenix server started in Docker container")
                
                # Wait for server to be ready
                import time
                import httpx
                
                for i in range(30):  # Wait up to 30 seconds
                    try:
                        response = httpx.get("http://localhost:6006/health", timeout=2.0)
                        if response.status_code == 200:
                            logger.info("✓ Phoenix server is ready")
                            break
                    except httpx.RequestError:
                        pass
                    
                    time.sleep(1)
                else:
                    logger.warning("Phoenix server may not be fully ready")
                
                # Update environment configuration
                self._update_env_config({
                    "PHOENIX_BASE_URL": "http://localhost:6006",
                    "PHOENIX_MCP_SERVER_URL": "http://localhost:6006/mcp"
                })
                
                return True
            else:
                logger.error(f"Failed to start Phoenix Docker container: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to setup Phoenix with Docker: {e}")
            return False
    
    def _setup_remote_phoenix(self) -> bool:
        """Set up connection to remote Phoenix server."""
        logger.info("Configuring remote Phoenix server connection...")
        
        # Prompt for remote server details
        phoenix_url = input("Enter Phoenix server URL (e.g., https://phoenix.example.com): ")
        api_key = input("Enter Phoenix API key: ")
        
        if not phoenix_url or not api_key:
            logger.error("Phoenix URL and API key are required")
            return False
        
        # Test connection
        try:
            import httpx
            
            headers = {"Authorization": f"Bearer {api_key}"}
            response = httpx.get(f"{phoenix_url}/health", headers=headers, timeout=10.0)
            
            if response.status_code == 200:
                logger.info("✓ Successfully connected to remote Phoenix server")
                
                # Update environment configuration
                self._update_env_config({
                    "PHOENIX_BASE_URL": phoenix_url,
                    "PHOENIX_API_KEY": api_key,
                    "PHOENIX_MCP_SERVER_URL": f"{phoenix_url}/mcp"
                })
                
                return True
            else:
                logger.error(f"Failed to connect to Phoenix server: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to test remote Phoenix connection: {e}")
            return False
    
    def configure_mcp_server(self) -> bool:
        """Configure Phoenix MCP server settings."""
        logger.info("Configuring Phoenix MCP server...")
        
        try:
            # Read current environment
            config = self._read_env_config()
            
            # Set MCP-specific configuration
            mcp_config = {
                "PHOENIX_MCP_REQUIRE_APPROVAL": "never",  # For development
                "PHOENIX_MCP_SERVER_LABEL": "phoenix",
            }
            
            # Prompt for approval settings
            approval_choice = input("Require approval for MCP tool calls? (y/n, default=n): ").lower()
            if approval_choice in ['y', 'yes']:
                mcp_config["PHOENIX_MCP_REQUIRE_APPROVAL"] = "always"
            
            # Update configuration
            config.update(mcp_config)
            self._write_env_config(config)
            
            logger.info("✓ Phoenix MCP server configuration updated")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure MCP server: {e}")
            return False
    
    async def test_integration(self) -> bool:
        """Test Phoenix MCP integration."""
        logger.info("Testing Phoenix MCP integration...")
        
        try:
            # Import after setup
            from evaluation.phoenix_integration import phoenix_integration
            from config.settings import settings
            
            # Test basic connection
            session_id = await phoenix_integration.start_evaluation_session("test_session")
            logger.info(f"✓ Created test session: {session_id}")
            
            # Test trace creation
            trace_id = await phoenix_integration.start_trace("test_trace", 
                {"test": True, "timestamp": "now"})
            logger.info(f"✓ Created test trace: {trace_id}")
            
            # Test span creation
            span_id = await phoenix_integration.create_span(
                trace_id=trace_id,
                span_name="test_span",
                span_type="test",
                metadata={"test_data": "integration_test"}
            )
            logger.info(f"✓ Created test span: {span_id}")
            
            # Test span completion
            await phoenix_integration.end_span(
                trace_id=trace_id,
                span_id=span_id,
                status="success",
                result="Integration test completed",
                metrics={"test_metric": 1.0}
            )
            logger.info("✓ Completed test span")
            
            # Test session closure
            final_metrics = await phoenix_integration.close_session()
            logger.info(f"✓ Closed test session with metrics: {len(final_metrics)} items")
            
            logger.info("✅ Phoenix MCP integration test passed!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Phoenix MCP integration test failed: {e}")
            return False
    
    async def test_evaluation_runner(self) -> bool:
        """Test the evaluation runner with Phoenix integration."""
        logger.info("Testing evaluation runner...")
        
        try:
            from evaluation.runner import EvaluationRunner
            from agents.supervisor import SupervisorAgent
            
            # Create test supervisor
            supervisor = SupervisorAgent()
            
            # Create evaluation runner
            runner = EvaluationRunner(
                supervisor_agent=supervisor,
                phoenix_enabled=True,
                output_dir="test_results"
            )
            
            # Test with a simple query (ID 1 from dataset)
            result = await runner.run_single_query_evaluation(
                query_id=1,
                save_results=False,
                run_quality_tests=False  # Skip quality tests for basic integration test
            )
            
            if result and result["evaluation_result"]["success"]:
                logger.info("✓ Evaluation runner test passed")
                return True
            else:
                logger.warning("⚠️ Evaluation runner test completed with issues")
                return False
                
        except Exception as e:
            logger.error(f"❌ Evaluation runner test failed: {e}")
            return False
    
    def _update_env_config(self, updates: Dict[str, str]) -> None:
        """Update environment configuration."""
        config = self._read_env_config()
        config.update(updates)
        self._write_env_config(config)
    
    def _read_env_config(self) -> Dict[str, str]:
        """Read environment configuration."""
        config = {}
        if self.env_file.exists():
            with open(self.env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        key, _, value = line.partition('=')
                        config[key.strip()] = value.strip()
        return config
    
    def _write_env_config(self, config: Dict[str, str]) -> None:
        """Write environment configuration."""
        with open(self.env_file, 'w') as f:
            f.write("# Multi-Agent Research System Configuration\n")
            f.write("# Generated by setup_phoenix_mcp.py\n\n")
            
            for key, value in config.items():
                f.write(f"{key}={value}\n")
        
        logger.info(f"Environment configuration updated: {self.env_file}")
    
    def create_startup_script(self) -> bool:
        """Create startup script for Phoenix MCP system."""
        startup_script = self.project_root / "start_phoenix_system.py"
        
        script_content = '''#!/usr/bin/env python3
"""
Startup script for Phoenix MCP evaluation system.
"""
import asyncio
import logging
from evaluation.monitoring import start_monitoring_stack
from evaluation.runner import EvaluationRunner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Start the complete Phoenix MCP system."""
    logger.info("Starting Phoenix MCP evaluation system...")
    
    try:
        # Start monitoring stack
        await start_monitoring_stack(port=8080)
    except KeyboardInterrupt:
        logger.info("System shutdown requested")
    except Exception as e:
        logger.error(f"System startup failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
'''
        
        with open(startup_script, 'w') as f:
            f.write(script_content)
        
        # Make executable
        startup_script.chmod(0o755)
        
        logger.info(f"Created startup script: {startup_script}")
        return True

async def main():
    """Main setup function."""
    setup = PhoenixMCPSetup()
    
    print("Phoenix MCP Integration Setup")
    print("=" * 40)
    
    # Check dependencies
    if not setup.check_dependencies():
        print("Please install missing dependencies before continuing.")
        sys.exit(1)
    
    # Setup Phoenix server
    print("\nPhoenix Server Setup:")
    print("1. Local (using Python package)")
    print("2. Docker (using Docker container)")
    print("3. Remote (connect to existing server)")
    
    choice = input("Choose setup method (1-3, default=1): ").strip()
    
    method_map = {"1": "local", "2": "docker", "3": "remote", "": "local"}
    method = method_map.get(choice, "local")
    
    if not setup.setup_phoenix_server(method):
        print("Phoenix server setup failed.")
        sys.exit(1)
    
    # Configure MCP server
    if not setup.configure_mcp_server():
        print("MCP server configuration failed.")
        sys.exit(1)
    
    # Test integration
    print("\nTesting Phoenix MCP integration...")
    if await setup.test_integration():
        print("✅ Phoenix MCP integration test passed!")
    else:
        print("❌ Phoenix MCP integration test failed!")
        sys.exit(1)
    
    # Optional: Test evaluation runner
    test_eval = input("\nTest evaluation runner? (y/n, default=n): ").lower()
    if test_eval in ['y', 'yes']:
        if await setup.test_evaluation_runner():
            print("✅ Evaluation runner test passed!")
        else:
            print("⚠️ Evaluation runner test had issues")
    
    # Create startup script
    setup.create_startup_script()
    
    print("\n" + "=" * 50)
    print("✅ Phoenix MCP integration setup complete!")
    print("\nNext steps:")
    print("1. Start the monitoring dashboard: python -m evaluation.monitoring")
    print("2. Run a single evaluation: python -m evaluation.runner --query-id 1")
    print("3. Run full evaluation: python -m evaluation.runner --full")
    print("4. Start complete system: python start_phoenix_system.py")
    print("\nDashboard will be available at: http://localhost:8080")

if __name__ == "__main__":
    asyncio.run(main())