"""
Program #4: Maze Visualizer
Visualizes the unsolved maze and then the maze with all the best paths highlighted

This program:
1. Displays the maze structure in a readable format
2. Shows available directions at each decision point
3. Highlights the best paths through the maze
4. Creates both text-based and graphical representations
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import networkx as nx
from maze_generator import MazeGenerator
from path_analyzer_answer import PathAnalyzer
import numpy as np
from typing import List, Dict, Tuple
import os

class MazeVisualizer:
    def __init__(self):
        self.maze = MazeGenerator()
        self.analyzer = PathAnalyzer()
        self.maze_info = self.maze.get_maze_info()
        
    def print_maze_structure(self):
        """
        Print the complete maze structure in text format
        """
        print("\n" + "="*100)
        print("                                MAZE STRUCTURE")
        print("="*100)
        
        for i, node in enumerate(self.maze_info['maze_structure']):
            print(f"\n📍 POSITION {i}")
            print("-" * 80)
            print(f"Description: {node['description']}")
            print(f"Available Directions: {', '.join(node['available_directions'])}")
            
            print("\nOutcomes:")
            for direction, outcome in node['outcomes'].items():
                if outcome['type'] == 'next':
                    print(f"  {direction.upper()}: → Continue to position {outcome['next_position']}")
                elif outcome['type'] == 'success':
                    print(f"  {direction.upper()}: ✅ SUCCESS! (Score: {outcome['score']})")
                elif outcome['type'] == 'failure':
                    print(f"  {direction.upper()}: ❌ Dead end (Score: {outcome['score']})")
        
        print("\n" + "="*100)
    
    def visualize_maze_graph(self, highlight_paths: List[str] = None, save_file: str = None):
        """
        Create a graph visualization of the maze using matplotlib and networkx
        """
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes and edges
        for i, node in enumerate(self.maze_info['maze_structure']):
            G.add_node(i, label=f"P{i}")
            
            for direction, outcome in node['outcomes'].items():
                if outcome['type'] == 'next':
                    G.add_edge(i, outcome['next_position'], 
                             direction=direction, color='blue', style='solid')
                elif outcome['type'] == 'success':
                    success_node = f"SUCCESS_{i}_{direction}"
                    G.add_node(success_node, label="SUCCESS", node_type='success')
                    G.add_edge(i, success_node, 
                             direction=direction, color='green', style='solid')
                elif outcome['type'] == 'failure':
                    failure_node = f"FAILURE_{i}_{direction}"
                    G.add_node(failure_node, label="DEAD END", node_type='failure')
                    G.add_edge(i, failure_node, 
                             direction=direction, color='red', style='solid')
        
        # Create matplotlib figure
        plt.figure(figsize=(16, 12))
        
        # Use hierarchical layout
        pos = self._create_hierarchical_layout(G)
        
        # Draw nodes
        regular_nodes = [n for n in G.nodes() if isinstance(n, int)]
        success_nodes = [n for n in G.nodes() if isinstance(n, str) and 'SUCCESS' in n]
        failure_nodes = [n for n in G.nodes() if isinstance(n, str) and 'FAILURE' in n]
        
        # Draw regular decision nodes
        nx.draw_networkx_nodes(G, pos, nodelist=regular_nodes, 
                              node_color='lightblue', node_size=1000, 
                              node_shape='o', alpha=0.8)
        
        # Draw success nodes
        nx.draw_networkx_nodes(G, pos, nodelist=success_nodes, 
                              node_color='lightgreen', node_size=800, 
                              node_shape='s', alpha=0.8)
        
        # Draw failure nodes
        nx.draw_networkx_nodes(G, pos, nodelist=failure_nodes, 
                              node_color='lightcoral', node_size=800, 
                              node_shape='X', alpha=0.8)
        
        # Draw edges with different colors
        for edge in G.edges(data=True):
            start, end, data = edge
            color = data.get('color', 'gray')
            direction = data.get('direction', '')
            
            nx.draw_networkx_edges(G, pos, [(start, end)], 
                                 edge_color=color, width=2, alpha=0.7,
                                 arrows=True, arrowsize=20)
        
        # Add node labels
        node_labels = {}
        for node in G.nodes():
            if isinstance(node, int):
                node_labels[node] = f"P{node}"
            elif 'SUCCESS' in str(node):
                node_labels[node] = "WIN"
            elif 'FAILURE' in str(node):
                node_labels[node] = "LOSE"
        
        nx.draw_networkx_labels(G, pos, node_labels, font_size=8, font_weight='bold')
        
        # Add edge labels (directions)
        edge_labels = {}
        for edge in G.edges(data=True):
            start, end, data = edge
            direction = data.get('direction', '')
            if direction:
                edge_labels[(start, end)] = direction[0].upper()  # First letter
        
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)
        
        # Highlight best paths if provided
        if highlight_paths:
            self._highlight_paths_on_graph(G, pos, highlight_paths)
        
        plt.title("Maze Structure Visualization\n(Blue=Decision Point, Green=Success, Red=Failure)", 
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if save_file:
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            print(f"Maze visualization saved to {save_file}")
        
        plt.show()
    
    def _create_hierarchical_layout(self, G) -> Dict:
        """
        Create a hierarchical layout for the maze graph
        """
        pos = {}
        
        # Place regular nodes in a grid pattern
        regular_nodes = [n for n in G.nodes() if isinstance(n, int)]
        
        # Arrange in rows
        cols = 5
        for i, node in enumerate(sorted(regular_nodes)):
            row = i // cols
            col = i % cols
            pos[node] = (col * 3, -row * 2)
        
        # Place terminal nodes around their parent nodes
        for node in G.nodes():
            if isinstance(node, str):
                # Extract parent node number
                parts = node.split('_')
                parent_node = int(parts[1])
                direction = parts[2]
                
                if parent_node in pos:
                    parent_x, parent_y = pos[parent_node]
                    
                    # Position terminal nodes around parent
                    if direction == 'left':
                        pos[node] = (parent_x - 1.2, parent_y - 0.5)
                    elif direction == 'right':
                        pos[node] = (parent_x + 1.2, parent_y - 0.5)
                    else:  # middle
                        pos[node] = (parent_x, parent_y - 1)
        
        return pos
    
    def _highlight_paths_on_graph(self, G, pos, paths: List[str]):
        """
        Highlight the best paths on the graph
        """
        colors = ['gold', 'orange', 'purple', 'cyan', 'magenta']
        
        for i, path_str in enumerate(paths[:5]):  # Top 5 paths
            if i >= len(colors):
                break
                
            directions = path_str.split(',')
            current_pos = 0
            color = colors[i]
            
            # Trace the path through the maze
            for direction in directions:
                if current_pos >= len(self.maze_info['maze_structure']):
                    break
                
                node = self.maze_info['maze_structure'][current_pos]
                
                if direction in node['outcomes']:
                    outcome = node['outcomes'][direction]
                    
                    if outcome['type'] == 'next':
                        next_pos = outcome['next_position']
                        # Highlight edge
                        nx.draw_networkx_edges(G, pos, [(current_pos, next_pos)], 
                                             edge_color=color, width=4, alpha=0.9)
                        current_pos = next_pos
                    else:
                        # Terminal node
                        terminal_node = f"{outcome['type'].upper()}_{current_pos}_{direction}"
                        if terminal_node in G.nodes():
                            nx.draw_networkx_edges(G, pos, [(current_pos, terminal_node)], 
                                                 edge_color=color, width=4, alpha=0.9)
                        break
    
    def create_path_comparison_chart(self, top_paths: List[Dict]):
        """
        Create a comparison chart of the top paths
        """
        if not top_paths:
            print("No paths to visualize!")
            return
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Extract data
        paths = [p['PATH'].replace(',', '→') for p in top_paths]
        scores = [p['SCORE'] for p in top_paths]
        lengths = [p['PATH_LENGTH'] for p in top_paths]
        ranks = [f"{p['RANK']}" for p in top_paths]
        
        # Truncate long paths for display
        display_paths = []
        for path in paths:
            if len(path) > 20:
                display_paths.append(path[:17] + "...")
            else:
                display_paths.append(path)
        
        # Chart 1: Scores
        bars1 = ax1.bar(ranks, scores, color=['gold', 'silver', '#CD7F32', 'lightblue', 'lightgreen'])
        ax1.set_title('Path Scores', fontweight='bold')
        ax1.set_xlabel('Rank')
        ax1.set_ylabel('Score')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars1, scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{score}', ha='center', va='bottom', fontweight='bold')
        
        # Chart 2: Path Lengths
        bars2 = ax2.bar(ranks, lengths, color=['gold', 'silver', '#CD7F32', 'lightblue', 'lightgreen'])
        ax2.set_title('Path Lengths', fontweight='bold')
        ax2.set_xlabel('Rank')
        ax2.set_ylabel('Steps')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, length in zip(bars2, lengths):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{length}', ha='center', va='bottom', fontweight='bold')
        
        # Chart 3: Path visualization
        ax3.barh(range(len(display_paths)), [1]*len(display_paths), 
                color=['gold', 'silver', '#CD7F32', 'lightblue', 'lightgreen'])
        ax3.set_yticks(range(len(display_paths)))
        ax3.set_yticklabels([f"#{rank}: {path}" for rank, path in zip(ranks, display_paths)])
        ax3.set_title('Path Sequences', fontweight='bold')
        ax3.set_xlabel('Rank Order')
        
        # Remove x-axis for path chart
        ax3.set_xticks([])
        
        plt.tight_layout()
        plt.savefig('path_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Path comparison chart saved as 'path_comparison.png'")
    
    def show_interactive_maze_solution(self):
        """
        Show an interactive solution where user can trace through the best path
        """
        # Load best paths
        if os.path.exists('maze_results.csv'):
            self.analyzer.load_csv_data()
            top_paths = self.analyzer.get_top_paths(1)
            
            if top_paths:
                best_path = top_paths[0]['PATH'].split(',')
                print("\n" + "="*80)
                print("                    INTERACTIVE BEST PATH WALKTHROUGH")
                print("="*80)
                print(f"Best Path: {' → '.join(best_path)}")
                print(f"Score: {top_paths[0]['SCORE']}")
                print(f"Steps: {top_paths[0]['PATH_LENGTH']}")
                print("-" * 80)
                
                # Walk through the path step by step
                current_pos = 0
                for step, direction in enumerate(best_path):
                    print(f"\nStep {step + 1}: At Position {current_pos}")
                    
                    if current_pos < len(self.maze_info['maze_structure']):
                        node = self.maze_info['maze_structure'][current_pos]
                        print(f"Description: {node['description']}")
                        print(f"Available directions: {', '.join(node['available_directions'])}")
                        print(f"🎯 Choosing: {direction.upper()}")
                        
                        if direction in node['outcomes']:
                            outcome = node['outcomes'][direction]
                            
                            if outcome['type'] == 'next':
                                current_pos = outcome['next_position']
                                print(f"   ➡️  Moving to position {current_pos}")
                            elif outcome['type'] == 'success':
                                print(f"   🎉 SUCCESS! Found the exit! Score: {outcome['score']}")
                                break
                            elif outcome['type'] == 'failure':
                                print(f"   💀 Dead end reached")
                                break
                        
                        input("Press Enter to continue...")
                
                print("\n🏆 Walkthrough complete!")
            else:
                print("No successful paths found to demonstrate!")
        else:
            print("No maze results file found. Run the maze solver first!")
    
    def display_unsolved_maze(self):
        """
        Display the unsolved maze structure
        """
        print("🗺️  Displaying unsolved maze structure...")
        self.print_maze_structure()
        self.visualize_maze_graph(save_file='unsolved_maze.png')
    
    def display_solved_maze(self):
        """
        Display the maze with best paths highlighted
        """
        print("🎯 Displaying maze with best paths...")
        
        if os.path.exists('maze_results.csv'):
            self.analyzer.load_csv_data()
            top_paths = self.analyzer.get_top_paths(5)
            
            if top_paths:
                # Extract path strings
                path_strings = [p['PATH'] for p in top_paths]
                
                self.visualize_maze_graph(highlight_paths=path_strings, 
                                        save_file='solved_maze.png')
                self.create_path_comparison_chart(top_paths)
            else:
                print("No successful paths found to highlight!")
                self.visualize_maze_graph(save_file='unsolved_maze_only.png')
        else:
            print("No maze results found. Please run the maze solver first!")
            self.visualize_maze_graph(save_file='unsolved_maze_only.png')
    
    def run_visualization(self):
        """
        Run the complete visualization workflow
        """
        print("🎨 Starting maze visualization...")
        
        # Show unsolved maze
        print("\n" + "="*60)
        print("STEP 1: UNSOLVED MAZE")
        print("="*60)
        self.display_unsolved_maze()
        
        # Show solved maze with best paths
        print("\n" + "="*60)
        print("STEP 2: MAZE WITH BEST PATHS")
        print("="*60)
        self.display_solved_maze()
        
        # Interactive walkthrough
        print("\n" + "="*60)
        print("STEP 3: INTERACTIVE WALKTHROUGH")
        print("="*60)
        self.show_interactive_maze_solution()
        
        print("\n✅ Visualization complete!")

# Run the visualizer
if __name__ == "__main__":
    visualizer = MazeVisualizer()
    visualizer.run_visualization()