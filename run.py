#!/usr/bin/env python3
"""
Embedding Visualizer - Quick Start Script
"""

import subprocess
import sys


def main():
    print("=" * 60)
    print("Embedding Visualizer - Starting...")
    print("=" * 60)

    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure Streamlit is installed:")
        print("  pip install -r requirements.txt")


if __name__ == "__main__":
    main()
