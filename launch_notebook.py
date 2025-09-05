#!/usr/bin/env python3
"""
Launch script for the Multi-Agent Evaluation Notebook
Provides easy access to the comprehensive evaluation interface.
"""
import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed."""
    required_packages = [
        'jupyter',
        'ipywidgets',
        'matplotlib',
        'seaborn',
        'pandas'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("❌ Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\n📦 Install missing packages with:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    return True

def launch_notebook():
    """Launch the Jupyter notebook."""
    notebook_path = Path(__file__).parent / "evaluation" / "multi_agent_evaluation_notebook.ipynb"
    
    if not notebook_path.exists():
        print(f"❌ Notebook not found: {notebook_path}")
        return False
    
    print("🚀 Launching Multi-Agent Evaluation Notebook...")
    print(f"📁 Notebook: {notebook_path}")
    print("🌐 Opening in your default browser...")
    
    try:
        # Launch Jupyter with the specific notebook
        subprocess.run([
            sys.executable, "-m", "jupyter", "notebook", 
            str(notebook_path),
            "--NotebookApp.open_browser=True"
        ], check=True)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch notebook: {e}")
        print("💡 Try installing Jupyter: pip install jupyter")
        return False
    except KeyboardInterrupt:
        print("\n🛑 Notebook server stopped")
        return True

def main():
    print("🤖 Multi-Agent Research System - Evaluation Notebook Launcher")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    print("✅ All requirements satisfied")
    
    # Launch notebook
    if not launch_notebook():
        sys.exit(1)
    
    print("🎉 Notebook session ended")

if __name__ == "__main__":
    main()