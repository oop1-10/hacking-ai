"""
Program #2: Maze Solver Template
Student Template - Complete this code to solve the maze

This program should:
1. Import and use the maze_generator.py
2. Find all possible combinations of maze directions
3. Save results to CSV with columns: RESULT, SCORE, PATH
4. Test all possible paths through the maze
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
        TODO: Generate all possible path combinations
        
        Hint: You need to create all possible combinations of directions
        up to a maximum depth. Consider that each position might have
        different available directions.
        
        Args:
            max_depth: Maximum number of decisions to make in a path
            
        Returns:
            List of all possible paths (each path is a list of directions)
        """
        # TODO: Implement this method
        # You'll need to:
        # 1. Get all possible directions (left, middle, right)
        # 2. Generate combinations up to max_depth
        # 3. Return list of path combinations
        
        all_paths = []
        
        # YOUR CODE HERE
        
        return all_paths
    
    def test_path(self, path: List[str]) -> Dict:
        """
        TODO: Test a single path through the maze
        
        Args:
            path: List of directions to follow
            
        Returns:
            Dictionary with result information
        """
        # TODO: Use maze.simulate_path() to test the path
        # Return the result in the correct format
        
        # YOUR CODE HERE
        
        pass
    
    def solve_maze(self):
        """
        TODO: Main solving function
        
        This should:
        1. Generate all possible paths
        2. Test each path
        3. Store results
        4. Save to CSV
        """
        print("Starting maze solving...")
        
        # TODO: Generate all paths
        paths = self.generate_all_paths()
        
        print(f"Testing {len(paths)} possible paths...")
        
        # TODO: Test each path and store results
        for path in paths:
            # Test the path and store result
            pass
        
        # TODO: Save results to CSV
        self.save_results_to_csv()
        
        print(f"Maze solving complete! Tested {len(self.results)} paths.")
    
    def save_results_to_csv(self, filename: str = "maze_results.csv"):
        """
        TODO: Save all results to a CSV file
        
        CSV should have columns: RESULT, SCORE, PATH
        - RESULT: "SUCCESS" or "FAILURE"
        - SCORE: Numerical score from maze
        - PATH: String representation of path (e.g., "left,right,middle")
        """
        # TODO: Implement CSV saving
        # Use the csv module to write results
        
        # YOUR CODE HERE
        
        pass

# Test your implementation
if __name__ == "__main__":
    solver = MazeSolver()
    
    # TODO: Uncomment this line when you've implemented the methods
    # solver.solve_maze()
    
    print("Template created! Implement the TODO methods to solve the maze.") 