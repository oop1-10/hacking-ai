"""
Program #2: Maze Solver Answer Key
Complete implementation that finds all possible maze path combinations

This program:
1. Imports and uses the maze_generator.py
2. Finds all possible combinations of maze directions
3. Saves results to CSV with columns: RESULT, SCORE, PATH
4. Tests all possible paths through the maze
"""

import csv
import itertools
from maze_generator import MazeGenerator
from typing import List, Dict

class MazeSolver:
    def __init__(self):
        self.maze = MazeGenerator()
        self.results = []
    
    def generate_all_paths(self, max_depth: int = 10) -> List[List[str]]:
        """
        Generate all possible path combinations up to max_depth
        
        Args:
            max_depth: Maximum number of decisions to make in a path
            
        Returns:
            List of all possible paths (each path is a list of directions)
        """
        all_directions = ['left', 'middle', 'right']
        all_paths = []
        
        # Generate paths of different lengths (1 to max_depth)
        for depth in range(1, max_depth + 1):
            # Generate all combinations with replacement for this depth
            for path in itertools.product(all_directions, repeat=depth):
                all_paths.append(list(path))
        
        return all_paths
    
    def generate_smart_paths(self, max_depth: int = 15) -> List[List[str]]:
        """
        Generate paths more intelligently by considering maze structure
        This reduces the search space significantly
        """
        all_paths = []
        
        def explore_paths(current_path: List[str], position: int, depth: int):
            if depth >= max_depth:
                return
            
            # Test current path to see if it's terminal
            if current_path:
                result = self.maze.simulate_path(current_path)
                if result['type'] in ['success', 'failure']:
                    all_paths.append(current_path.copy())
                    return
            
            # Get available directions at current position
            maze_info = self.maze.get_maze_info()
            if position < len(maze_info['maze_structure']):
                available_dirs = maze_info['maze_structure'][position]['available_directions']
                
                for direction in available_dirs:
                    new_path = current_path + [direction]
                    # Simulate to get next position
                    sim_result = self.maze.simulate_path(new_path)
                    
                    if sim_result['type'] == 'continue':
                        # Continue exploring from next position
                        next_pos = position + 1  # Simplified position tracking
                        explore_paths(new_path, next_pos, depth + 1)
                    elif sim_result['type'] in ['success', 'failure']:
                        # Terminal state reached
                        all_paths.append(new_path)
        
        # Start exploration
        explore_paths([], 0, 0)
        
        # Also add some brute force paths for completeness
        brute_force_paths = self.generate_all_paths(max_depth=8)
        all_paths.extend(brute_force_paths)
        
        # Remove duplicates
        unique_paths = []
        seen = set()
        for path in all_paths:
            path_str = ','.join(path)
            if path_str not in seen:
                seen.add(path_str)
                unique_paths.append(path)
        
        return unique_paths
    
    def test_path(self, path: List[str]) -> Dict:
        """
        Test a single path through the maze
        
        Args:
            path: List of directions to follow
            
        Returns:
            Dictionary with result information
        """
        result = self.maze.simulate_path(path)
        
        return {
            'result': 'SUCCESS' if result['type'] == 'success' else 'FAILURE',
            'score': result.get('score', 0),
            'path': ','.join(path),
            'decisions_made': result.get('decisions_made', len(path)),
            'message': result.get('message', '')
        }
    
    def solve_maze(self, use_smart_generation: bool = True):
        """
        Main solving function that tests all possible paths
        """
        print("Starting maze solving...")
        
        # Generate all paths (use smart generation to reduce search space)
        if use_smart_generation:
            paths = self.generate_smart_paths()
            print(f"Generated {len(paths)} paths using smart generation")
        else:
            paths = self.generate_all_paths()
            print(f"Generated {len(paths)} paths using brute force")
        
        print(f"Testing {len(paths)} possible paths...")
        
        # Test each path and store results
        for i, path in enumerate(paths):
            if i % 100 == 0:  # Progress indicator
                print(f"Tested {i}/{len(paths)} paths...")
                
            result = self.test_path(path)
            self.results.append(result)
        
        # Save results to CSV
        self.save_results_to_csv()
        
        # Print summary
        successful_paths = [r for r in self.results if r['result'] == 'SUCCESS']
        failed_paths = [r for r in self.results if r['result'] == 'FAILURE']
        
        print(f"\nMaze solving complete!")
        print(f"Total paths tested: {len(self.results)}")
        print(f"Successful paths: {len(successful_paths)}")
        print(f"Failed paths: {len(failed_paths)}")
        
        if successful_paths:
            best_path = max(successful_paths, key=lambda x: x['score'])
            print(f"Best path found: {best_path['path']} (Score: {best_path['score']})")
    
    def save_results_to_csv(self, filename: str = "maze_results.csv"):
        """
        Save all results to a CSV file
        
        CSV columns: RESULT, SCORE, PATH
        """
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['RESULT', 'SCORE', 'PATH']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header
            writer.writeheader()
            
            # Write results
            for result in self.results:
                writer.writerow({
                    'RESULT': result['result'],
                    'SCORE': result['score'],
                    'PATH': result['path']
                })
        
        print(f"Results saved to {filename}")

# Run the solver
if __name__ == "__main__":
    solver = MazeSolver()
    solver.solve_maze(use_smart_generation=True)