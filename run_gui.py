    #!/usr/bin/env python3
"""
Simple launcher for the Frequency Analysis GUI
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from frequency_analysis_gui import main

if __name__ == "__main__":
    print("=" * 60)
    print("Mosquito Frequency Range Analysis - GUI")
    print("=" * 60)
    print("Starting graphical interface...")
    print()
    main()
