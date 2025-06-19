"""
Maze GUI Visualizer
A comprehensive graphical user interface for the maze challenge system

Features:
- Interactive maze exploration
- Real-time path visualization  
- Step-by-step solution walkthrough
- Statistics dashboard
- Export capabilities
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import networkx as nx
import pandas as pd
import os
from typing import List, Dict, Optional
import threading

from maze_generator import MazeGenerator
from path_analyzer_answer import PathAnalyzer
from maze_visualizer import MazeVisualizer

class MazeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Maze Challenge System - GUI Visualizer")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize components
        self.maze = MazeGenerator(seed_key="MAZE_2025_CHALLENGE")
        self.analyzer = PathAnalyzer()
        self.visualizer = MazeVisualizer()
        
        # GUI state variables
        self.current_position = 0
        self.current_path = []
        self.best_paths = []
        self.maze_graph = None
        self.graph_pos = None
        
        # Create GUI components
        self.setup_gui()
        self.load_initial_data()
        
    def setup_gui(self):
        """Setup the main GUI layout"""
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Create left panel (controls)
        self.create_control_panel(main_frame)
        
        # Create right panel (visualization)
        self.create_visualization_panel(main_frame)
        
        # Create bottom panel (status and statistics)
        self.create_status_panel(main_frame)
        
    def create_control_panel(self, parent):
        """Create the left control panel"""
        control_frame = ttk.LabelFrame(parent, text="Control Panel", padding="10")
        control_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Maze Info Section
        info_frame = ttk.LabelFrame(control_frame, text="Maze Information", padding="5")
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.maze_info_label = ttk.Label(info_frame, text="Loading maze...", wraplength=250)
        self.maze_info_label.pack()
        
        # Navigation Section
        nav_frame = ttk.LabelFrame(control_frame, text="Manual Navigation", padding="5")
        nav_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Position controls
        pos_frame = ttk.Frame(nav_frame)
        pos_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(pos_frame, text="Position:").pack(side=tk.LEFT)
        self.position_var = tk.StringVar(value="0")
        position_spin = ttk.Spinbox(pos_frame, from_=0, to=14, textvariable=self.position_var, 
                                   width=5, command=self.on_position_change)
        position_spin.pack(side=tk.LEFT, padx=(5, 0))
        
        # Direction buttons
        self.direction_frame = ttk.Frame(nav_frame)
        self.direction_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Path display
        path_frame = ttk.LabelFrame(control_frame, text="Current Path", padding="5")
        path_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.path_display = tk.Text(path_frame, height=4, width=30, wrap=tk.WORD)
        self.path_display.pack(fill=tk.BOTH, expand=True)
        
        # Auto-solve Section
        solve_frame = ttk.LabelFrame(control_frame, text="Auto Solve", padding="5")
        solve_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(solve_frame, text="Load Best Paths", command=self.load_best_paths).pack(fill=tk.X, pady=(0, 5))
        
        # Best path selection
        self.path_var = tk.StringVar()
        self.path_combo = ttk.Combobox(solve_frame, textvariable=self.path_var, state="readonly")
        self.path_combo.pack(fill=tk.X, pady=(0, 5))
        self.path_combo.bind('<<ComboboxSelected>>', self.on_path_select)
        
        # Animation controls
        anim_frame = ttk.Frame(solve_frame)
        anim_frame.pack(fill=tk.X)
        
        ttk.Button(anim_frame, text="Play", command=self.play_animation).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(anim_frame, text="Step", command=self.step_animation).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(anim_frame, text="Reset", command=self.reset_path).pack(side=tk.LEFT)
        
        # File operations
        file_frame = ttk.LabelFrame(control_frame, text="File Operations", padding="5")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(file_frame, text="Load CSV", command=self.load_csv).pack(fill=tk.X, pady=(0, 2))
        ttk.Button(file_frame, text="Export Image", command=self.export_image).pack(fill=tk.X, pady=(0, 2))
        ttk.Button(file_frame, text="Generate New Maze", command=self.generate_new_maze).pack(fill=tk.X)
        
    def create_visualization_panel(self, parent):
        """Create the main visualization panel"""
        viz_frame = ttk.LabelFrame(parent, text="Maze Visualization", padding="5")
        viz_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        viz_frame.columnconfigure(0, weight=1)
        viz_frame.rowconfigure(0, weight=1)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Add toolbar
        toolbar_frame = ttk.Frame(viz_frame)
        toolbar_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()
        
    def create_status_panel(self, parent):
        """Create the bottom status and statistics panel"""
        status_frame = ttk.LabelFrame(parent, text="Status & Statistics", padding="5")
        status_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Create notebook for tabs
        notebook = ttk.Notebook(status_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Status tab
        status_tab = ttk.Frame(notebook)
        notebook.add(status_tab, text="Status")
        
        self.status_text = tk.Text(status_tab, height=6, wrap=tk.WORD)
        status_scroll = ttk.Scrollbar(status_tab, orient=tk.VERTICAL, command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=status_scroll.set)
        
        self.status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        status_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Statistics tab
        stats_tab = ttk.Frame(notebook)
        notebook.add(stats_tab, text="Statistics")
        
        self.stats_text = tk.Text(stats_tab, height=6, wrap=tk.WORD)
        stats_scroll = ttk.Scrollbar(stats_tab, orient=tk.VERTICAL, command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=stats_scroll.set)
        
        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        stats_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
    def load_initial_data(self):
        """Load initial maze data and setup visualization"""
        self.log_status("Initializing maze system...")
        
        # Setup maze info
        maze_info = self.maze.get_maze_info()
        info_text = f"Seed Key: {maze_info['seed_key']}\n"
        info_text += f"Decision Points: {maze_info['total_positions']}\n"
        info_text += f"Generated: Ready"
        self.maze_info_label.config(text=info_text)
        
        # Create initial visualization
        self.create_maze_graph()
        self.update_visualization()
        self.update_position_info()
        
        self.log_status("Maze system initialized successfully!")
        
    def create_maze_graph(self):
        """Create the NetworkX graph representation of the maze"""
        self.maze_graph = nx.DiGraph()
        maze_info = self.maze.get_maze_info()
        
        # Add nodes and edges
        for i, node in enumerate(maze_info['maze_structure']):
            self.maze_graph.add_node(i, label=f"P{i}", pos=i)
            
            for direction, outcome in node['outcomes'].items():
                if outcome['type'] == 'next':
                    self.maze_graph.add_edge(i, outcome['next_position'], 
                                           direction=direction, color='blue')
                elif outcome['type'] == 'success':
                    success_node = f"SUCCESS_{i}_{direction}"
                    self.maze_graph.add_node(success_node, label="WIN", node_type='success')
                    self.maze_graph.add_edge(i, success_node, direction=direction, color='green')
                elif outcome['type'] == 'failure':
                    failure_node = f"FAILURE_{i}_{direction}"
                    self.maze_graph.add_node(failure_node, label="LOSE", node_type='failure')
                    self.maze_graph.add_edge(i, failure_node, direction=direction, color='red')
        
        # Create layout
        self.graph_pos = self.create_hierarchical_layout()
        
    def create_hierarchical_layout(self):
        """Create a hierarchical layout for the maze graph"""
        pos = {}
        regular_nodes = [n for n in self.maze_graph.nodes() if isinstance(n, int)]
        
        # Arrange decision nodes in rows
        cols = 5
        for i, node in enumerate(sorted(regular_nodes)):
            row = i // cols
            col = i % cols
            pos[node] = (col * 3, -row * 2)
        
        # Position terminal nodes around their parents
        for node in self.maze_graph.nodes():
            if isinstance(node, str):
                parts = node.split('_')
                parent_node = int(parts[1])
                direction = parts[2]
                
                if parent_node in pos:
                    parent_x, parent_y = pos[parent_node]
                    
                    if direction == 'left':
                        pos[node] = (parent_x - 1.2, parent_y - 0.5)
                    elif direction == 'right':
                        pos[node] = (parent_x + 1.2, parent_y - 0.5)
                    else:  # middle
                        pos[node] = (parent_x, parent_y - 1)
        
        return pos
        
    def update_visualization(self):
        """Update the main maze visualization"""
        self.ax.clear()
        
        if not self.maze_graph:
            return
            
        # Draw different node types
        regular_nodes = [n for n in self.maze_graph.nodes() if isinstance(n, int)]
        success_nodes = [n for n in self.maze_graph.nodes() if isinstance(n, str) and 'SUCCESS' in n]
        failure_nodes = [n for n in self.maze_graph.nodes() if isinstance(n, str) and 'FAILURE' in n]
        
        # Draw regular nodes
        nx.draw_networkx_nodes(self.maze_graph, self.graph_pos, nodelist=regular_nodes,
                              node_color='lightblue', node_size=800, ax=self.ax)
        
        # Highlight current position
        if self.current_position in regular_nodes:
            nx.draw_networkx_nodes(self.maze_graph, self.graph_pos, 
                                 nodelist=[self.current_position],
                                 node_color='yellow', node_size=1000, ax=self.ax)
        
        # Draw terminal nodes
        if success_nodes:
            nx.draw_networkx_nodes(self.maze_graph, self.graph_pos, nodelist=success_nodes,
                                  node_color='lightgreen', node_size=600, node_shape='s', ax=self.ax)
        
        if failure_nodes:
            nx.draw_networkx_nodes(self.maze_graph, self.graph_pos, nodelist=failure_nodes,
                                  node_color='lightcoral', node_size=600, node_shape='X', ax=self.ax)
        
        # Draw edges
        for edge in self.maze_graph.edges(data=True):
            start, end, data = edge
            color = data.get('color', 'gray')
            nx.draw_networkx_edges(self.maze_graph, self.graph_pos, [(start, end)],
                                 edge_color=color, width=2, alpha=0.7, ax=self.ax,
                                 arrows=True, arrowsize=15)
        
        # Highlight current path
        if len(self.current_path) > 1:
            path_edges = [(self.current_path[i], self.current_path[i+1]) 
                         for i in range(len(self.current_path)-1)]
            nx.draw_networkx_edges(self.maze_graph, self.graph_pos, path_edges,
                                 edge_color='orange', width=4, alpha=0.8, ax=self.ax)
        
        # Add labels
        node_labels = {}
        for node in self.maze_graph.nodes():
            if isinstance(node, int):
                node_labels[node] = f"P{node}"
            elif 'SUCCESS' in str(node):
                node_labels[node] = "WIN"
            elif 'FAILURE' in str(node):
                node_labels[node] = "LOSE"
        
        nx.draw_networkx_labels(self.maze_graph, self.graph_pos, node_labels,
                               font_size=8, font_weight='bold', ax=self.ax)
        
        # Add edge labels (directions)
        edge_labels = {}
        for edge in self.maze_graph.edges(data=True):
            start, end, data = edge
            direction = data.get('direction', '')
            if direction and isinstance(start, int):
                edge_labels[(start, end)] = direction[0].upper()
        
        nx.draw_networkx_edge_labels(self.maze_graph, self.graph_pos, edge_labels,
                                   font_size=6, ax=self.ax)
        
        self.ax.set_title("Maze Structure - Current Position Highlighted in Yellow", 
                         fontsize=12, fontweight='bold')
        self.ax.axis('off')
        self.canvas.draw()
        
    def update_position_info(self):
        """Update the position information and direction buttons"""
        # Clear previous direction buttons
        for widget in self.direction_frame.winfo_children():
            widget.destroy()
            
        # Get current node info
        maze_info = self.maze.get_maze_info()
        if self.current_position < len(maze_info['maze_structure']):
            node = maze_info['maze_structure'][self.current_position]
            
            ttk.Label(self.direction_frame, text="Available directions:").pack()
            
            # Create direction buttons
            for direction in node['available_directions']:
                btn = ttk.Button(self.direction_frame, text=direction.title(),
                               command=lambda d=direction: self.make_move(d))
                btn.pack(side=tk.LEFT, padx=2)
        
        # Update path display
        self.path_display.delete(1.0, tk.END)
        if self.current_path:
            path_str = " → ".join([str(p) for p in self.current_path])
            self.path_display.insert(1.0, f"Path: {path_str}\nLength: {len(self.current_path)-1} steps")
        else:
            self.path_display.insert(1.0, "No path selected")
            
    def make_move(self, direction):
        """Make a move in the specified direction"""
        result = self.maze.make_choice(direction)
        
        if result['type'] == 'continue':
            self.current_position = result['new_position']
            self.current_path.append(self.current_position)
            self.log_status(f"Moved {direction} to position {self.current_position}")
        elif result['type'] == 'success':
            self.log_status(f"SUCCESS! {result['message']} Score: {result['score']}")
            messagebox.showinfo("Success!", f"Congratulations! You found the exit!\nScore: {result['score']}")
        elif result['type'] == 'failure':
            self.log_status(f"FAILURE! {result['message']}")
            messagebox.showwarning("Dead End", "You hit a dead end!")
        else:
            self.log_status(f"Error: {result['message']}")
            
        self.position_var.set(str(self.current_position))
        self.update_position_info()
        self.update_visualization()
        
    def on_position_change(self):
        """Handle manual position change"""
        try:
            new_pos = int(self.position_var.get())
            if 0 <= new_pos < len(self.maze.maze_structure):
                self.current_position = new_pos
                self.current_path = [new_pos]  # Reset path to just current position
                self.maze.current_position = new_pos
                self.update_position_info()
                self.update_visualization()
                self.log_status(f"Moved to position {new_pos}")
        except ValueError:
            pass
            
    def reset_path(self):
        """Reset to starting position"""
        self.current_position = 0
        self.current_path = [0]
        self.maze.reset_maze()
        self.position_var.set("0")
        self.update_position_info()
        self.update_visualization()
        self.log_status("Reset to starting position")
        
    def load_best_paths(self):
        """Load best paths from CSV"""
        if not os.path.exists('maze_results.csv'):
            messagebox.showwarning("No Results", "No maze results found. Please run the solver first!")
            return
            
        try:
            self.analyzer.load_csv_data()
            self.best_paths = self.analyzer.get_top_paths(10)
            
            if self.best_paths:
                # Update combo box
                path_options = []
                for i, path in enumerate(self.best_paths):
                    option = f"#{i+1}: {path['PATH'][:30]}... (Score: {path['SCORE']})"
                    path_options.append(option)
                
                self.path_combo['values'] = path_options
                self.log_status(f"Loaded {len(self.best_paths)} best paths")
                
                # Update statistics
                self.update_statistics()
            else:
                messagebox.showinfo("No Paths", "No successful paths found in results!")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load paths: {e}")
            
    def on_path_select(self, event=None):
        """Handle path selection from combo box"""
        selection = self.path_combo.current()
        if selection >= 0 and selection < len(self.best_paths):
            selected_path = self.best_paths[selection]
            self.log_status(f"Selected path #{selection+1}: Score {selected_path['SCORE']}")
            
    def play_animation(self):
        """Play animation of selected path"""
        selection = self.path_combo.current()
        if selection < 0 or selection >= len(self.best_paths):
            messagebox.showwarning("No Path", "Please select a path first!")
            return
            
        path = self.best_paths[selection]['PATH'].split(',')
        self.animate_path(path)
        
    def animate_path(self, directions):
        """Animate a path step by step"""
        def animate_step(step):
            if step >= len(directions):
                self.log_status("Animation complete!")
                return
                
            direction = directions[step]
            self.make_move(direction)
            
            # Schedule next step
            self.root.after(1000, lambda: animate_step(step + 1))
            
        self.reset_path()
        self.log_status(f"Starting animation of path: {' → '.join(directions)}")
        animate_step(0)
        
    def step_animation(self):
        """Step through animation manually"""
        # Implementation for manual stepping
        pass
        
    def load_csv(self):
        """Load a different CSV file"""
        filename = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.analyzer.csv_filename = filename
                self.load_best_paths()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load CSV: {e}")
                
    def export_image(self):
        """Export current visualization as image"""
        filename = filedialog.asksaveasfilename(
            title="Save visualization",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.fig.savefig(filename, dpi=300, bbox_inches='tight')
                self.log_status(f"Exported visualization to {filename}")
                messagebox.showinfo("Success", f"Saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {e}")
                
    def generate_new_maze(self):
        """Generate a new maze with different seed"""
        seed = tk.simpledialog.askstring("New Maze", "Enter seed key:", 
                                        initialvalue="MAZE_2025_CHALLENGE")
        if seed:
            self.maze = MazeGenerator(seed_key=seed)
            self.visualizer = MazeVisualizer()
            self.reset_path()
            self.create_maze_graph()
            self.load_initial_data()
            self.log_status(f"Generated new maze with seed: {seed}")
            
    def update_statistics(self):
        """Update the statistics display"""
        if not self.analyzer.results_data:
            return
            
        stats = self.analyzer.calculate_statistics()
        
        stats_text = "MAZE SOLVING STATISTICS\n"
        stats_text += "=" * 40 + "\n\n"
        stats_text += f"Total Paths Tested: {stats['total_paths_tested']:,}\n"
        stats_text += f"Successful Paths: {stats['successful_paths']:,}\n"
        stats_text += f"Failed Paths: {stats['failed_paths']:,}\n"
        stats_text += f"Success Rate: {stats['success_rate_percent']}%\n\n"
        
        stats_text += f"Average Score (All): {stats['average_score_all']}\n"
        stats_text += f"Average Score (Success): {stats['average_score_successful']}\n"
        stats_text += f"Highest Score: {stats['highest_score']}\n"
        stats_text += f"Lowest Score: {stats['lowest_score']}\n\n"
        
        stats_text += f"Most Used Direction: {stats['most_used_direction'][0]}\n"
        stats_text += f"Average Path Length: {stats['average_path_length']} steps\n"
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats_text)
        
    def log_status(self, message):
        """Log a status message"""
        self.status_text.insert(tk.END, f"[{tk.time.strftime('%H:%M:%S')}] {message}\n")
        self.status_text.see(tk.END)
        self.root.update_idletasks()

def main():
    """Main function to run the GUI"""
    # Import required modules for dialog
    try:
        import tkinter.simpledialog
        tk.simpledialog = tkinter.simpledialog
        import time
        tk.time = time
    except ImportError:
        pass
    
    root = tk.Tk()
    app = MazeGUI(root)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()

if __name__ == "__main__":
    main()