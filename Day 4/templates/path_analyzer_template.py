"""
Program #3: Path Analyzer Template
Student Template - Complete this code to analyze maze results

This program should:
1. Read the CSV file created by the maze solver
2. Analyze and rank the paths
3. Return the top 5 best paths
4. Display rankings and statistics
"""

import csv
import pandas as pd
from typing import List, Dict, Tuple

class PathAnalyzer:
    def __init__(self, csv_filename: str = "maze_results.csv"):
        self.csv_filename = csv_filename
        self.results_data = []
        self.successful_paths = []
        self.failed_paths = []
    
    def load_csv_data(self):
        """
        TODO: Load data from the CSV file
        
        Read the CSV file and populate self.results_data with the maze results
        Also separate successful and failed paths into different lists
        """
        # TODO: Implement CSV reading
        # Use csv module or pandas to read the file
        # Populate self.results_data, self.successful_paths, self.failed_paths
        
        # YOUR CODE HERE
        
        pass
    
    def analyze_successful_paths(self) -> List[Dict]:
        """
        TODO: Analyze successful paths and rank them
        
        Returns:
            List of successful paths sorted by ranking criteria
        """
        # TODO: Implement path analysis
        # Consider different ranking criteria:
        # - Highest score
        # - Shortest path length
        # - Combination of both
        
        # YOUR CODE HERE
        
        return []
    
    def get_top_paths(self, n: int = 5) -> List[Dict]:
        """
        TODO: Get the top N best paths
        
        Args:
            n: Number of top paths to return (default 5)
            
        Returns:
            List of top N paths with rankings
        """
        # TODO: Use analyze_successful_paths() to get ranked paths
        # Return only the top N paths
        
        # YOUR CODE HERE
        
        return []
    
    def calculate_statistics(self) -> Dict:
        """
        TODO: Calculate various statistics about the maze solving results
        
        Returns:
            Dictionary with statistics like success rate, average scores, etc.
        """
        # TODO: Calculate:
        # - Total paths tested
        # - Success rate (percentage)
        # - Average score of successful paths
        # - Average score of failed paths
        # - Most common path length
        # - Any other interesting statistics
        
        # YOUR CODE HERE
        
        return {}
    
    def display_rankings(self):
        """
        TODO: Display the top 5 paths in a nice format
        
        Show:
        1. Ranking (1st, 2nd, 3rd, etc.)
        2. Path
        3. Score
        4. Any other relevant information
        """
        # TODO: Get top 5 paths and display them nicely
        top_paths = self.get_top_paths(5)
        
        print("=== TOP 5 MAZE PATHS ===")
        
        # YOUR CODE HERE - Display the paths in a nice format
        
        pass
    
    def run_analysis(self):
        """
        TODO: Run the complete analysis
        
        This should:
        1. Load the CSV data
        2. Perform analysis
        3. Display results and statistics
        """
        print("Starting path analysis...")
        
        # TODO: Load data
        self.load_csv_data()
        
        # TODO: Display statistics
        stats = self.calculate_statistics()
        
        # TODO: Display top paths
        self.display_rankings()
        
        print("Analysis complete!")

# Test your implementation
if __name__ == "__main__":
    analyzer = PathAnalyzer()
    
    # TODO: Uncomment this line when you've implemented the methods
    # analyzer.run_analysis()
    
    print("Template created! Implement the TODO methods to analyze paths.") 