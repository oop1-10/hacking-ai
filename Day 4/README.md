# Maze Challenge System

A complete maze generation and solving system with visualization capabilities, designed for educational purposes.

## Overview

This system consists of 4 main programs that work together to create, solve, analyze, and visualize maze challenges:

1. **🏗️ Maze Generator** (`maze_generator.py`) - Creates deterministic mazes with hardcoded keys
2. **🤖 Maze Solver** (`maze_solver_answer.py` + `maze_solver_template.py`) - Finds all possible paths through the maze
3. **📊 Path Analyzer** (`path_analyzer_answer.py` + `path_analyzer_template.py`) - Ranks and analyzes the best paths
4. **🎨 Maze Visualizer** (`maze_visualizer.py` + `maze_gui.py`) - Creates visual representations of the maze and solutions

## Quick Start

### Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the GUI Visualizer (Recommended):**
   ```bash
   python launch_gui.py
   # OR
   python main_demo.py gui
   ```

3. **Run the complete demonstration:**
   ```bash
   python main_demo.py
   ```

4. **Or run individual components:**
   ```bash
   # Generate and explore the maze structure
   python maze_generator.py
   
   # Solve all possible paths
   python maze_solver_answer.py
   
   # Analyze the results
   python path_analyzer_answer.py
   
   # Visualize everything (command line)
   python maze_visualizer.py
   ```

### Quick Test
```bash
python main_demo.py test
```

## 🎮 GUI Visualizer Features

The new **GUI Visualizer** (`maze_gui.py`) provides a comprehensive interactive interface:

### **Main Features:**
- **🗺️ Interactive Maze Visualization** - Real-time graph display with NetworkX
- **🎯 Manual Navigation** - Click buttons to explore paths manually
- **🚀 Automated Solving** - Load and animate best path solutions
- **📊 Statistics Dashboard** - Real-time analysis and metrics
- **💾 File Management** - Load different CSV files, export images
- **🎬 Path Animation** - Watch solutions play out step-by-step

### **GUI Components:**
- **Control Panel** - Navigate maze, load paths, control animations
- **Visualization Panel** - Interactive graph with zoom and pan capabilities
- **Status & Statistics** - Tabbed interface showing real-time information
- **Export Tools** - Save visualizations as high-quality images

### **Navigation Options:**
- Use position spinner to jump to any decision point
- Click direction buttons to make moves manually
- Load best paths from CSV and watch them animate
- Reset to starting position anytime

## System Architecture

### Program #1: Maze Generator
- **File:** `maze_generator.py`
- **Purpose:** Creates a deterministic maze structure based on a hardcoded key
- **Features:**
  - Each decision point has 1-3 available directions (left, middle, right)
  - Outcomes can be: continue to next position, success (exit found), or failure (dead end)
  - Consistent maze generation using seed-based randomization
  - Callable by other programs for path simulation

### Program #2: Maze Solver
- **Template:** `maze_solver_template.py` (for students)
- **Answer Key:** `maze_solver_answer.py` (complete implementation)
- **Purpose:** Systematically tests all possible path combinations
- **Features:**
  - Smart path generation to reduce search space
  - Brute force fallback for completeness
  - CSV output with columns: RESULT, SCORE, PATH
  - Progress tracking and performance optimization

### Program #3: Path Analyzer
- **Template:** `path_analyzer_template.py` (for students)
- **Answer Key:** `path_analyzer_answer.py` (complete implementation)
- **Purpose:** Analyzes maze results and ranks the best paths
- **Features:**
  - Multi-criteria ranking system (score, efficiency, path length, complexity)
  - Comprehensive statistics calculation
  - Top 5 path identification
  - Export capabilities for further analysis

### Program #4: Maze Visualizer
- **Command Line:** `maze_visualizer.py` (text + matplotlib plots)
- **GUI Version:** `maze_gui.py` (interactive tkinter interface)
- **Purpose:** Creates visual representations of the maze and solutions
- **Features:**
  - Text-based maze structure display
  - Graph visualization using NetworkX and Matplotlib
  - Path highlighting for best solutions
  - Interactive walkthrough of optimal paths
  - Comparison charts for top paths

## Educational Use

### For Instructors

This system is designed as a complete educational package:

- **Templates** provide scaffolding for student implementation
- **Answer keys** offer complete reference implementations
- **GUI Visualization** helps students understand their solutions interactively
- **Modular design** allows incremental learning

### For Students

Work with the template files:

1. **Start with** `maze_solver_template.py`
   - Implement path generation algorithms
   - Learn about exhaustive search techniques
   - Practice CSV file handling

2. **Continue with** `path_analyzer_template.py`
   - Implement data analysis and ranking
   - Learn about multi-criteria decision making
   - Practice statistical calculations

3. **Test and visualize** your solutions
   - Use the GUI to see your paths interactively
   - Compare with answer key implementations
   - Iterate and improve your algorithms

### Assignment Workflow

```bash
# 1. Student implements template
python maze_solver_template.py     # (after implementing TODO methods)

# 2. Student implements analyzer
python path_analyzer_template.py   # (after implementing TODO methods)

# 3. Student visualizes results with GUI
python launch_gui.py

# 4. Compare with reference implementation
python maze_solver_answer.py
python path_analyzer_answer.py
```

## File Structure

```
├── maze_generator.py              # Core maze generation engine
├── maze_solver_template.py        # Student template for solver
├── maze_solver_answer.py          # Complete solver implementation
├── path_analyzer_template.py      # Student template for analyzer
├── path_analyzer_answer.py        # Complete analyzer implementation
├── maze_visualizer.py             # Command-line visualization system
├── maze_gui.py                    # GUI visualization system ⭐
├── launch_gui.py                  # Simple GUI launcher ⭐
├── main_demo.py                   # Complete system demonstration
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Generated Files

When you run the system, it creates:

- **`maze_results.csv`** - Complete solving results
- **`top_paths.csv`** - Top 10 ranked paths
- **`unsolved_maze.png`** - Maze structure visualization
- **`solved_maze.png`** - Maze with best paths highlighted
- **`path_comparison.png`** - Path comparison charts

## Maze Structure

The maze consists of decision points where you can choose directions:

- **Directions:** left, middle, right
- **Outcomes:** 
  - Continue to next position
  - Success (exit found) - earns points
  - Failure (dead end) - minimal points
- **Scoring:** Based on path efficiency and decision count

## Advanced Features

### Smart Path Generation
The solver uses intelligent path generation to reduce search space while ensuring completeness.

### Multi-Criteria Ranking
Paths are ranked using a composite score considering:
- Raw score (40%)
- Efficiency (30%)
- Path length (20%)
- Complexity (10%)

### Interactive GUI Visualization
The GUI system provides multiple interaction modes:
- Manual maze exploration with instant feedback
- Automated path animation with timing controls
- Real-time statistics and performance metrics
- Export capabilities for presentations and reports

## Usage Examples

### Quick GUI Launch
```bash
# Fastest way to start exploring
python launch_gui.py
```

### Complete Workflow
```bash
# 1. Generate maze and solve all paths
python maze_solver_answer.py

# 2. Launch GUI to explore results
python launch_gui.py

# 3. In GUI: Click "Load Best Paths" → Select a path → Click "Play"
```

### Educational Sequence
```bash
# 1. Show students the maze structure
python maze_generator.py

# 2. Let them implement the solver template
# (Students work on maze_solver_template.py)

# 3. Test their implementation
python maze_solver_template.py

# 4. Visualize their results
python launch_gui.py
```

## Troubleshooting

### Common Issues

1. **Import errors:** Ensure all dependencies are installed via `pip install -r requirements.txt`
2. **No CSV file:** Run the solver before the analyzer or GUI
3. **GUI doesn't start:** Check that tkinter is available (comes with most Python installations)
4. **Visualization errors:** Check that matplotlib backend is properly configured
5. **Memory issues:** Reduce `max_depth` parameter in path generation

### Getting Help

```bash
python main_demo.py help     # Show usage information
python main_demo.py student  # Show student-specific guidance
python main_demo.py test     # Run system diagnostics
python launch_gui.py         # Direct GUI access
```

## Customization

### Changing the Maze
Modify the `seed_key` in `maze_generator.py` to create different mazes:

```python
maze = MazeGenerator(seed_key="YOUR_CUSTOM_KEY_HERE")
```

Or use the GUI's "Generate New Maze" button to create different mazes interactively.

### Adjusting Difficulty
- Modify `max_depth` in solver for longer/shorter path searches
- Adjust scoring weights in the analyzer
- Change maze size by modifying the range in `_generate_maze_structure()`

## GUI Keyboard Shortcuts

While using the GUI:
- **Enter** - Confirm dialogs
- **Escape** - Cancel operations
- **Space** - Reset to starting position
- **Arrow Keys** - Navigate in matplotlib view (when focused)

## License

This educational software is provided as-is for learning purposes. Feel free to modify and adapt for your educational needs.

---

**Happy Maze Solving! 🎮🧩**

**NEW: Try the interactive GUI visualizer for the best experience! 🎨** 