#!/usr/bin/env python3
"""
GUI Launcher for Maze Challenge System
Simple script to launch the GUI visualizer
"""

import sys
import os

def main():
    """Launch the GUI visualizer"""
    print("🎮 Maze Challenge System - GUI Launcher")
    print("=" * 50)
    
    try:
        # Import and run the GUI
        from maze_gui import main as gui_main
        print("✅ Starting GUI visualizer...")
        gui_main()
        
    except ImportError as e:
        print(f"❌ Error importing GUI: {e}")
        print("\nPossible solutions:")
        print("1. Install required packages:")
        print("   pip install -r requirements.txt")
        print("2. Make sure you're in the correct directory")
        print("3. Check that all Python files are present")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error launching GUI: {e}")
        print("\nIf the error persists, try running:")
        print("  python main_demo.py gui")
        sys.exit(1)

if __name__ == "__main__":
    main() 