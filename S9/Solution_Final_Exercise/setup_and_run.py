#!/usr/bin/env python3
"""
Setup and Demo Script for Knowledge Graph Exercise
Automatically installs dependencies and runs the demo
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages."""
    print("📦 Installing required packages...")
    
    requirements = [
        "langchain-google-genai>=2.0.0",
        "pydantic>=2.0.0", 
        "networkx>=3.0",
        "matplotlib>=3.6.0",
        "plotly>=5.15.0",
        "langchain>=0.1.0",
        "langchain-core>=0.1.0",
        "google-generativeai>=0.3.0",
        "kaleido>=0.2.1"
    ]
    
    for package in requirements:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    print("✅ All packages installed successfully!")
    return True

def check_api_key():
    """Check and set up API key."""
    if "GOOGLE_API_KEY" not in os.environ:
        print("\n🔑 Google AI API Key Setup")
        print("=" * 40)
        print("You need a Google AI API key to run this demo.")
        print("Get one for free at: https://aistudio.google.com/")
        print()
        
        api_key = input("Enter your Google AI API key (or press Enter to skip): ").strip()
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            print("✅ API key configured!")
            return True
        else:
            print("⚠️ Skipping API key setup. You can run the demo later with your key.")
            return False
    else:
        print("✅ Google AI API key found!")
        return True

def run_demo():
    """Run the knowledge graph demo."""
    print("\n🚀 Running Knowledge Graph Demo...")
    print("=" * 40)
    
    try:
        from simple_demo import demo_with_sample_text
        demo_with_sample_text()
    except ImportError:
        print("❌ Could not import demo module. Make sure all files are in the current directory.")
    except Exception as e:
        print(f"❌ Error running demo: {e}")

def main():
    print("🎓 Knowledge Graph Exercise - Auto Setup")
    print("=" * 50)
    print("This script will:")
    print("1. Install required Python packages")
    print("2. Set up your Google AI API key") 
    print("3. Run a demonstration with sample data")
    print()
    
    # Step 1: Install packages
    if not install_requirements():
        print("❌ Failed to install packages. Please install manually with:")
        print("pip install -r requirements.txt")
        return
    
    # Step 2: API key setup
    has_api_key = check_api_key()
    
    # Step 3: Run demo
    if has_api_key:
        run_demo()
    else:
        print("\n💡 To run the demo later:")
        print("1. Set your API key: export GOOGLE_API_KEY='your_key_here'")
        print("2. Run: python simple_demo.py")
    
    print("\n🎉 Setup complete!")
    print("📁 Available files to run:")
    print("  • simple_demo.py - Quick demo with sample text")
    print("  • knowledge_graph_creator.py - Full audio processing")
    print("  • README.md - Complete documentation")

if __name__ == "__main__":
    main() 