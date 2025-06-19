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
        self.show_best_paths = False
        
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
        
        # Path display controls
        display_frame = ttk.Frame(solve_frame)
        display_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.show_paths_var = tk.BooleanVar(value=False)
        self.show_paths_checkbox = ttk.Checkbutton(display_frame, text="Highlight Best Paths", 
                                                  variable=self.show_paths_var,
                                                  command=self.toggle_path_display)
        self.show_paths_checkbox.pack(side=tk.LEFT)
        
        # Camera following control
        self.follow_camera_var = tk.BooleanVar(value=True)
        self.follow_camera_checkbox = ttk.Checkbutton(display_frame, text="Follow Camera", 
                                                     variable=self.follow_camera_var,
                                                     command=self.toggle_camera_follow)
        self.follow_camera_checkbox.pack(side=tk.LEFT, padx=(20, 0))
        
        # File operations
        file_frame = ttk.LabelFrame(control_frame, text="File Operations", padding="5")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(file_frame, text="Load CSV", command=self.load_csv).pack(fill=tk.X, pady=(0, 2))
        ttk.Button(file_frame, text="Export Image", command=self.export_image).pack(fill=tk.X, pady=(0, 2))
        ttk.Button(file_frame, text="Generate New Maze", command=self.generate_new_maze).pack(fill=tk.X)
        
    def create_visualization_panel(self, parent):
        """Create the main visualization panel with enhanced interactivity"""
        viz_frame = ttk.LabelFrame(parent, text="Interactive Maze Visualization", padding="5")
        viz_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        viz_frame.columnconfigure(0, weight=1)
        viz_frame.rowconfigure(0, weight=1)
        
        # Create matplotlib figure with larger size for better visibility
        self.fig = Figure(figsize=(12, 10), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # Enable interactive features
        self.fig.patch.set_facecolor('white')
        self.ax.set_facecolor('white')
        
        # Create scrollable frame for the canvas
        canvas_frame = ttk.Frame(viz_frame)
        canvas_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        
        # Create canvas with scrollbars
        self.canvas = FigureCanvasTkAgg(self.fig, canvas_frame)
        self.canvas.draw()
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Add scrollbars
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Enhanced toolbar with custom controls
        toolbar_frame = ttk.Frame(viz_frame)
        toolbar_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # Standard matplotlib navigation toolbar
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        
        # Custom control buttons
        custom_controls = ttk.Frame(toolbar_frame)
        custom_controls.pack(side=tk.RIGHT, padx=(10, 0))
        
        ttk.Button(custom_controls, text="Fit to Window", command=self.fit_maze_to_window).pack(side=tk.LEFT, padx=2)
        ttk.Button(custom_controls, text="Reset View", command=self.reset_view).pack(side=tk.LEFT, padx=2)
        ttk.Button(custom_controls, text="Center Current", command=self.center_on_current).pack(side=tk.LEFT, padx=2)
        ttk.Button(custom_controls, text="Zoom In", command=self.zoom_in).pack(side=tk.LEFT, padx=2)
        ttk.Button(custom_controls, text="Zoom Out", command=self.zoom_out).pack(side=tk.LEFT, padx=2)
        
        # View options
        view_frame = ttk.LabelFrame(viz_frame, text="View Options", padding="3")
        view_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # Layout options
        layout_frame = ttk.Frame(view_frame)
        layout_frame.pack(fill=tk.X)
        
        ttk.Label(layout_frame, text="Layout:").pack(side=tk.LEFT)
        
        self.layout_var = tk.StringVar(value="hierarchical")
        layout_combo = ttk.Combobox(layout_frame, textvariable=self.layout_var, 
                                  values=["hierarchical", "spring", "circular", "grid"], 
                                  state="readonly", width=12)
        layout_combo.pack(side=tk.LEFT, padx=(5, 10))
        layout_combo.bind('<<ComboboxSelected>>', self.on_layout_change)
        
        # Node size control
        ttk.Label(layout_frame, text="Node Size:").pack(side=tk.LEFT, padx=(10, 5))
        self.node_size_var = tk.IntVar(value=800)
        node_size_scale = ttk.Scale(layout_frame, from_=200, to=1500, 
                                  variable=self.node_size_var, orient=tk.HORIZONTAL,
                                  length=100, command=self.on_node_size_change)
        node_size_scale.pack(side=tk.LEFT, padx=2)
        
        # Enable mouse interaction events
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('scroll_event', self.on_mouse_scroll)
        
        # Initialize interaction state
        self.dragging = False
        self.drag_start = None
        
        # Camera following state
        self.follow_camera = True  # Auto-follow during animation
        
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
        
        # Fit maze to window initially
        self.root.after(100, self.fit_maze_to_window)  # Delay to ensure proper rendering
        
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
        
        # Get dynamic node size
        base_node_size = self.node_size_var.get() if hasattr(self, 'node_size_var') else 800
        
        # Draw regular nodes
        nx.draw_networkx_nodes(self.maze_graph, self.graph_pos, nodelist=regular_nodes,
                              node_color='lightblue', node_size=base_node_size, ax=self.ax)
        
        # Highlight current position
        if self.current_position in regular_nodes:
            nx.draw_networkx_nodes(self.maze_graph, self.graph_pos, 
                                 nodelist=[self.current_position],
                                 node_color='yellow', node_size=base_node_size + 200, ax=self.ax)
        
        # Draw terminal nodes
        terminal_size = max(base_node_size * 0.75, 400)
        if success_nodes:
            nx.draw_networkx_nodes(self.maze_graph, self.graph_pos, nodelist=success_nodes,
                                  node_color='lightgreen', node_size=terminal_size, node_shape='s', ax=self.ax)
        
        if failure_nodes:
            nx.draw_networkx_nodes(self.maze_graph, self.graph_pos, nodelist=failure_nodes,
                                  node_color='lightcoral', node_size=terminal_size, node_shape='X', ax=self.ax)
        
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
        
        # Highlight best paths if loaded
        self._highlight_best_paths_on_graph()
        
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
        
        # Update title based on whether we're showing paths
        title = "Maze Structure - Current Position Highlighted in Yellow"
        if hasattr(self, 'show_best_paths') and self.show_best_paths and self.best_paths:
            title += f" | Best {len(self.best_paths)} Paths Highlighted"
        
        self.ax.set_title(title, fontsize=12, fontweight='bold')
        self.ax.axis('off')
        self.canvas.draw()
        
    def _highlight_best_paths_on_graph(self):
        """Highlight best paths on the graph"""
        if not hasattr(self, 'show_best_paths') or not self.show_best_paths or not self.best_paths:
            return
            
        colors = ['gold', 'orange', 'purple', 'cyan', 'magenta']
        
        for i, path_data in enumerate(self.best_paths[:5]):  # Show top 5 paths
            if i >= len(colors):
                break
                
            path_str = path_data['PATH']
            directions = path_str.split(',')
            current_pos = 0
            color = colors[i]
            
            # Trace the path through the maze
            for direction in directions:
                if current_pos >= len(self.maze.maze_structure):
                    break
                
                node = self.maze.maze_structure[current_pos]
                
                if direction in node['outcomes']:
                    outcome = node['outcomes'][direction]
                    
                    if outcome['type'] == 'next':
                        next_pos = outcome['next_position']
                        # Check if edge exists in graph
                        if self.maze_graph.has_edge(current_pos, next_pos):
                            nx.draw_networkx_edges(self.maze_graph, self.graph_pos, 
                                                 [(current_pos, next_pos)],
                                                 edge_color=color, width=3, alpha=0.9, ax=self.ax)
                        current_pos = next_pos
                    else:
                        # Terminal node
                        terminal_node = f"{outcome['type'].upper()}_{current_pos}_{direction}"
                        if terminal_node in self.maze_graph.nodes():
                            nx.draw_networkx_edges(self.maze_graph, self.graph_pos, 
                                                 [(current_pos, terminal_node)],
                                                 edge_color=color, width=3, alpha=0.9, ax=self.ax)
                        break
        
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
        
        # Auto-center camera if following is enabled
        if hasattr(self, 'follow_camera_var') and self.follow_camera_var.get():
            self.center_camera_on_node(self.current_position)
        
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
        
        # Center camera on starting position if following is enabled
        if hasattr(self, 'follow_camera_var') and self.follow_camera_var.get():
            self.center_camera_on_node(self.current_position)
            
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
                
                # Automatically enable path highlighting when paths are loaded
                self.show_paths_var.set(True)
                self.show_best_paths = True
                self.update_visualization()
                
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
        
        # Enable camera following for animation
        self.follow_camera_var.set(True)
        self.follow_camera = True
        
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
        
    def center_camera_on_node(self, node):
        """Center the camera view on a specific node"""
        if self.maze_graph and self.graph_pos and node in self.graph_pos:
            node_x, node_y = self.graph_pos[node]
            
            # Get current view range
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            
            x_range = xlim[1] - xlim[0]
            y_range = ylim[1] - ylim[0]
            
            # Center on the node
            new_xlim = [node_x - x_range/2, node_x + x_range/2]
            new_ylim = [node_y - y_range/2, node_y + y_range/2]
            
            self.ax.set_xlim(new_xlim)
            self.ax.set_ylim(new_ylim)
            self.canvas.draw_idle()
            
    def toggle_camera_follow(self):
        """Toggle camera following during animation"""
        self.follow_camera = self.follow_camera_var.get()
        
        if self.follow_camera:
            self.log_status("Camera following enabled - will center on current position")
            # Immediately center on current position if enabled
            self.center_camera_on_node(self.current_position)
        else:
            self.log_status("Camera following disabled")
            
    def toggle_path_display(self):
        """Toggle showing/hiding best paths"""
        self.show_best_paths = self.show_paths_var.get()
        self.update_visualization()
        
        if self.show_best_paths and self.best_paths:
            self.log_status(f"Showing {len(self.best_paths)} best paths")
        else:
            self.log_status("Hidden path highlighting")
        
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
        
    def on_mouse_press(self, event):
        """Handle mouse press for dragging"""
        if event.button == 1 and event.inaxes == self.ax:  # Left mouse button
            self.dragging = True
            self.drag_start = (event.xdata, event.ydata)
            
    def on_mouse_release(self, event):
        """Handle mouse release"""
        self.dragging = False
        self.drag_start = None
        
    def on_mouse_move(self, event):
        """Handle mouse movement for panning"""
        if self.dragging and self.drag_start and event.inaxes == self.ax:
            dx = event.xdata - self.drag_start[0]
            dy = event.ydata - self.drag_start[1]
            
            # Get current axis limits
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            
            # Update limits (reverse direction for natural drag feel)
            self.ax.set_xlim([xlim[0] - dx, xlim[1] - dx])
            self.ax.set_ylim([ylim[0] - dy, ylim[1] - dy])
            
            self.canvas.draw_idle()
            
    def on_mouse_scroll(self, event):
        """Handle mouse scroll for zooming"""
        if event.inaxes == self.ax:
            # Get current axis limits
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            
            # Calculate zoom factor
            zoom_factor = 1.1 if event.step > 0 else 0.9
            
            # Get mouse position in data coordinates
            xdata, ydata = event.xdata, event.ydata
            
            # Calculate new limits centered on mouse position
            x_range = (xlim[1] - xlim[0]) * zoom_factor
            y_range = (ylim[1] - ylim[0]) * zoom_factor
            
            new_xlim = [xdata - x_range/2, xdata + x_range/2]
            new_ylim = [ydata - y_range/2, ydata + y_range/2]
            
            self.ax.set_xlim(new_xlim)
            self.ax.set_ylim(new_ylim)
            
            self.canvas.draw_idle()
            
    def fit_maze_to_window(self):
        """Fit the entire maze to the window"""
        if self.maze_graph and self.graph_pos:
            # Get all node positions
            x_coords = [pos[0] for pos in self.graph_pos.values()]
            y_coords = [pos[1] for pos in self.graph_pos.values()]
            
            if x_coords and y_coords:
                # Add padding around the maze
                padding = 1.0
                x_min, x_max = min(x_coords) - padding, max(x_coords) + padding
                y_min, y_max = min(y_coords) - padding, max(y_coords) + padding
                
                self.ax.set_xlim([x_min, x_max])
                self.ax.set_ylim([y_min, y_max])
                self.canvas.draw()
                
                self.log_status("Fitted maze to window")
                
    def reset_view(self):
        """Reset view to original state"""
        self.ax.autoscale()
        self.canvas.draw()
        self.log_status("Reset view to original")
        
    def center_on_current(self):
        """Center view on current position"""
        self.center_camera_on_node(self.current_position)
        self.log_status(f"Centered camera on position {self.current_position}")
        
    def zoom_in(self):
        """Zoom in by 20%"""
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2
        
        x_range = (xlim[1] - xlim[0]) * 0.8
        y_range = (ylim[1] - ylim[0]) * 0.8
        
        self.ax.set_xlim([x_center - x_range/2, x_center + x_range/2])
        self.ax.set_ylim([y_center - y_range/2, y_center + y_range/2])
        self.canvas.draw()
        
    def zoom_out(self):
        """Zoom out by 20%"""
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2
        
        x_range = (xlim[1] - xlim[0]) * 1.2
        y_range = (ylim[1] - ylim[0]) * 1.2
        
        self.ax.set_xlim([x_center - x_range/2, x_center + x_range/2])
        self.ax.set_ylim([y_center - y_range/2, y_center + y_range/2])
        self.canvas.draw()
        
    def on_layout_change(self, event=None):
        """Handle layout change"""
        layout_type = self.layout_var.get()
        
        if layout_type == "hierarchical":
            self.graph_pos = self.create_hierarchical_layout()
        elif layout_type == "spring":
            self.graph_pos = nx.spring_layout(self.maze_graph, k=3, iterations=50)
        elif layout_type == "circular":
            # Only use regular nodes for circular layout
            regular_nodes = [n for n in self.maze_graph.nodes() if isinstance(n, int)]
            pos = nx.circular_layout(self.maze_graph.subgraph(regular_nodes))
            # Add terminal nodes around their parents
            self.graph_pos = self._add_terminal_positions(pos)
        elif layout_type == "grid":
            # Create a grid layout for regular nodes
            regular_nodes = [n for n in self.maze_graph.nodes() if isinstance(n, int)]
            cols = int(len(regular_nodes)**0.5) + 1
            pos = {}
            for i, node in enumerate(sorted(regular_nodes)):
                row = i // cols
                col = i % cols
                pos[node] = (col * 2, -row * 2)
            self.graph_pos = self._add_terminal_positions(pos)
        
        self.update_visualization()
        self.log_status(f"Changed layout to {layout_type}")
        
    def _add_terminal_positions(self, base_pos):
        """Add terminal node positions around their parent nodes"""
        pos = base_pos.copy()
        
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
        
    def on_node_size_change(self, value):
        """Handle node size change"""
        self.update_visualization()
        
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