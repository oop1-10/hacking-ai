"""
Main Demo Program
Demonstrates all four maze programs working together

This program runs through the complete workflow:
1. Generate the maze
2. Solve all possible paths
3. Analyze and rank the paths
4. Visualize the results
"""

import sys
import os
from maze_generator import MazeGenerator
from maze_solver_answer import MazeSolver
from path_analyzer_answer import PathAnalyzer
from maze_visualizer import MazeVisualizer

def main():
    """
    Run the complete maze challenge workflow
    """
    print("🎮" + "="*78 + "🎮")
    print("                          MAZE CHALLENGE SYSTEM")
    print("🎮" + "="*78 + "🎮")
    print()
    print("This demonstration will:")
    print("1. 🏗️  Generate a maze with hardcoded key")
    print("2. 🤖 Solve all possible paths and save to CSV")
    print("3. 📊 Analyze paths and rank the top 5")
    print("4. 🎨 Visualize the maze and best paths")
    print()
    
    try:
        # Step 1: Demonstrate maze generation
        print("🏗️  STEP 1: MAZE GENERATION")
        print("-" * 60)
        maze = MazeGenerator()
        print(f"✅ Maze generated with key: '{maze.seed_key}'")
        print(f"✅ Maze has {len(maze.maze_structure)} decision points")
        
        # Show a quick preview
        print("\nQuick preview of first 3 positions:")
        for i in range(min(3, len(maze.maze_structure))):
            node = maze.maze_structure[i]
            print(f"  Position {i}: {len(node['available_directions'])} directions available")
        
        input("\nPress Enter to continue to solving phase...")
        
        # Step 2: Solve the maze
        print("\n🤖 STEP 2: MAZE SOLVING")
        print("-" * 60)
        solver = MazeSolver()
        solver.solve_maze(use_smart_generation=True)
        
        input("\nPress Enter to continue to analysis phase...")
        
        # Step 3: Analyze paths
        print("\n📊 STEP 3: PATH ANALYSIS")
        print("-" * 60)
        analyzer = PathAnalyzer()
        analyzer.run_analysis()
        
        input("\nPress Enter to continue to visualization phase...")
        
        # Step 4: Visualize results
        print("\n🎨 STEP 4: VISUALIZATION")
        print("-" * 60)
        visualizer = MazeVisualizer()
        
        print("Choose visualization option:")
        print("1. Full visualization (unsolved + solved maze)")
        print("2. Just show unsolved maze")
        print("3. Just show solved maze with paths")
        print("4. Interactive walkthrough only")
        print("5. Launch GUI Visualizer")
        
        while True:
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                visualizer.run_visualization()
                break
            elif choice == '2':
                visualizer.display_unsolved_maze()
                break
            elif choice == '3':
                visualizer.display_solved_maze()
                break
            elif choice == '4':
                visualizer.show_interactive_maze_solution()
                break
            elif choice == '5':
                print("Launching GUI Visualizer...")
                try:
                    from maze_gui import main as gui_main
                    gui_main()
                except ImportError as e:
                    print(f"Error launching GUI: {e}")
                    print("Please ensure all GUI dependencies are installed.")
                break
            else:
                print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")
        
        # Summary
        print("\n🎉 DEMONSTRATION COMPLETE!")
        print("="*80)
        print("Files created:")
        print("  📄 maze_results.csv - Complete solving results")
        print("  📄 top_paths.csv - Top 10 ranked paths")
        print("  🖼️  unsolved_maze.png - Maze structure visualization")
        print("  🖼️  solved_maze.png - Maze with best paths highlighted")
        print("  🖼️  path_comparison.png - Path comparison charts")
        print()
        print("You can now:")
        print("  • Run individual programs separately")
        print("  • Modify the maze key in maze_generator.py")
        print("  • Use the template files for student assignments")
        print("  • Analyze the CSV results with your own tools")
        
    except KeyboardInterrupt:
        print("\n\n❌ Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        print("Please check that all required packages are installed:")
        print("  pip install matplotlib networkx pandas numpy")
        sys.exit(1)

def quick_test():
    """
    Quick test to verify all programs work
    """
    print("🔧 Running quick system test...")
    
    try:
        # Test maze generation
        maze = MazeGenerator()
        assert len(maze.maze_structure) > 0
        print("✅ Maze generation: OK")
        
        # Test path simulation
        test_result = maze.simulate_path(['right', 'left'])
        assert 'type' in test_result
        print("✅ Path simulation: OK")
        
        # Test solver instantiation
        solver = MazeSolver()
        assert solver.maze is not None
        print("✅ Maze solver: OK")
        
        # Test analyzer instantiation
        analyzer = PathAnalyzer()
        assert analyzer.csv_filename is not None
        print("✅ Path analyzer: OK")
        
        # Test visualizer instantiation
        visualizer = MazeVisualizer()
        assert visualizer.maze is not None
        print("✅ Maze visualizer: OK")
        
        print("\n🎉 All systems functional!")
        return True
        
    except Exception as e:
        print(f"\n❌ System test failed: {e}")
        return False

def show_student_info():
    """
    Show information for students
    """
    print("\n📚 STUDENT INFORMATION")
    print("="*60)
    print()
    print("Template Files (for students to complete):")
    print("  📝 maze_solver_template.py")
    print("  📝 path_analyzer_template.py")
    print()
    print("Answer Key Files (complete implementations):")
    print("  ✅ maze_solver_answer.py")
    print("  ✅ path_analyzer_answer.py")
    print()
    print("Core System Files:")
    print("  🏗️  maze_generator.py - Maze generation engine")
    print("  🎨 maze_visualizer.py - Visualization system")
    print()
    print("Assignment Instructions:")
    print("1. Students should work with the template files")
    print("2. Implement the TODO methods in each template")
    print("3. Test their implementations against the maze")
    print("4. Compare results with the answer key")
    print("5. Use the visualizer to see their solutions")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            quick_test()
        elif sys.argv[1] == "student":
            show_student_info()
        elif sys.argv[1] == "gui":
            print("Launching GUI Visualizer directly...")
            try:
                from maze_gui import main as gui_main
                gui_main()
            except ImportError as e:
                print(f"Error launching GUI: {e}")
                print("Please ensure all GUI dependencies are installed.")
        elif sys.argv[1] == "help":
            print("Usage:")
            print("  python main_demo.py          - Run full demonstration")
            print("  python main_demo.py test     - Run quick system test")
            print("  python main_demo.py student  - Show student information")
            print("  python main_demo.py gui      - Launch GUI visualizer directly")
            print("  python main_demo.py help     - Show this help message")
        else:
            print("Unknown option. Use 'help' for usage information.")
    else:
        # Run quick test first
        if quick_test():
            print()
            main()
        else:
            print("\n❌ System test failed. Please fix issues before running demo.") 