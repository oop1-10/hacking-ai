"""
Program #3: Path Analyzer Answer Key
Complete implementation that analyzes maze results and ranks paths

This program:
1. Reads the CSV file created by the maze solver
2. Analyzes and ranks the paths by multiple criteria
3. Returns the top 5 best paths
4. Displays comprehensive statistics
"""

import csv
import pandas as pd
from typing import List, Dict, Tuple
from collections import Counter
import os

class PathAnalyzer:
    def __init__(self, csv_filename: str = "maze_results.csv"):
        self.csv_filename = csv_filename
        self.results_data = []
        self.successful_paths = []
        self.failed_paths = []
    
    def load_csv_data(self):
        """
        Load data from the CSV file and separate successful/failed paths
        """
        if not os.path.exists(self.csv_filename):
            print(f"Error: CSV file '{self.csv_filename}' not found!")
            print("Please run the maze solver first to generate results.")
            return
        
        try:
            with open(self.csv_filename, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                
                for row in reader:
                    # Convert score to integer
                    row['SCORE'] = int(row['SCORE'])
                    row['PATH_LENGTH'] = len(row['PATH'].split(',')) if row['PATH'] else 0
                    
                    self.results_data.append(row)
                    
                    if row['RESULT'] == 'SUCCESS':
                        self.successful_paths.append(row)
                    else:
                        self.failed_paths.append(row)
            
            print(f"Loaded {len(self.results_data)} results from {self.csv_filename}")
            
        except Exception as e:
            print(f"Error reading CSV file: {e}")
    
    def analyze_successful_paths(self) -> List[Dict]:
        """
        Analyze successful paths and rank them by multiple criteria
        
        Returns:
            List of successful paths sorted by composite ranking
        """
        if not self.successful_paths:
            return []
        
        # Create enhanced data for ranking
        enhanced_paths = []
        
        for path_data in self.successful_paths:
            enhanced = path_data.copy()
            
            # Calculate additional metrics
            enhanced['EFFICIENCY'] = enhanced['SCORE'] / max(enhanced['PATH_LENGTH'], 1)
            enhanced['PATH_COMPLEXITY'] = self._calculate_path_complexity(enhanced['PATH'])
            
            # Calculate composite score for ranking
            # Weighting: 40% score, 30% efficiency, 20% path length (shorter better), 10% complexity (lower better)
            max_score = max(p['SCORE'] for p in self.successful_paths)
            max_length = max(p['PATH_LENGTH'] for p in self.successful_paths)
            max_complexity = max(self._calculate_path_complexity(p['PATH']) for p in self.successful_paths)
            
            normalized_score = enhanced['SCORE'] / max_score if max_score > 0 else 0
            normalized_length = 1 - (enhanced['PATH_LENGTH'] / max_length) if max_length > 0 else 1
            normalized_complexity = 1 - (enhanced['PATH_COMPLEXITY'] / max_complexity) if max_complexity > 0 else 1
            
            enhanced['COMPOSITE_SCORE'] = (
                0.4 * normalized_score +
                0.3 * enhanced['EFFICIENCY'] / 100 +  # Normalize efficiency
                0.2 * normalized_length +
                0.1 * normalized_complexity
            )
            
            enhanced_paths.append(enhanced)
        
        # Sort by composite score (descending)
        enhanced_paths.sort(key=lambda x: x['COMPOSITE_SCORE'], reverse=True)
        
        return enhanced_paths
    
    def _calculate_path_complexity(self, path_str: str) -> float:
        """
        Calculate path complexity based on direction changes and patterns
        """
        if not path_str:
            return 0
        
        directions = path_str.split(',')
        if len(directions) <= 1:
            return 0
        
        # Count direction changes
        changes = 0
        for i in range(1, len(directions)):
            if directions[i] != directions[i-1]:
                changes += 1
        
        # Complexity = changes / total_steps
        return changes / len(directions)
    
    def get_top_paths(self, n: int = 5) -> List[Dict]:
        """
        Get the top N best paths based on comprehensive analysis
        
        Args:
            n: Number of top paths to return
            
        Returns:
            List of top N paths with rankings
        """
        ranked_paths = self.analyze_successful_paths()
        
        # Add ranking information
        for i, path in enumerate(ranked_paths[:n]):
            path['RANK'] = i + 1
            path['RANK_SUFFIX'] = self._get_rank_suffix(i + 1)
        
        return ranked_paths[:n]
    
    def _get_rank_suffix(self, rank: int) -> str:
        """Get ordinal suffix for ranking (1st, 2nd, 3rd, etc.)"""
        if 10 <= rank % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(rank % 10, 'th')
        return f"{rank}{suffix}"
    
    def calculate_statistics(self) -> Dict:
        """
        Calculate comprehensive statistics about the maze solving results
        """
        if not self.results_data:
            return {}
        
        total_paths = len(self.results_data)
        successful_count = len(self.successful_paths)
        failed_count = len(self.failed_paths)
        
        # Basic statistics
        success_rate = (successful_count / total_paths) * 100 if total_paths > 0 else 0
        
        # Score statistics
        all_scores = [r['SCORE'] for r in self.results_data]
        successful_scores = [r['SCORE'] for r in self.successful_paths] if self.successful_paths else [0]
        failed_scores = [r['SCORE'] for r in self.failed_paths] if self.failed_paths else [0]
        
        # Path length statistics
        path_lengths = [r['PATH_LENGTH'] for r in self.results_data]
        path_length_counter = Counter(path_lengths)
        most_common_length = path_length_counter.most_common(1)[0] if path_length_counter else (0, 0)
        
        # Direction analysis
        all_directions = []
        for result in self.results_data:
            if result['PATH']:
                all_directions.extend(result['PATH'].split(','))
        
        direction_counter = Counter(all_directions)
        
        return {
            'total_paths_tested': total_paths,
            'successful_paths': successful_count,
            'failed_paths': failed_count,
            'success_rate_percent': round(success_rate, 2),
            'average_score_all': round(sum(all_scores) / len(all_scores), 2),
            'average_score_successful': round(sum(successful_scores) / len(successful_scores), 2),
            'average_score_failed': round(sum(failed_scores) / len(failed_scores), 2),
            'highest_score': max(all_scores),
            'lowest_score': min(all_scores),
            'most_common_path_length': most_common_length[0],
            'most_common_path_length_count': most_common_length[1],
            'average_path_length': round(sum(path_lengths) / len(path_lengths), 2),
            'direction_frequency': dict(direction_counter),
            'most_used_direction': direction_counter.most_common(1)[0] if direction_counter else ('none', 0)
        }
    
    def display_rankings(self):
        """
        Display the top 5 paths in a comprehensive format
        """
        top_paths = self.get_top_paths(5)
        
        if not top_paths:
            print("No successful paths found to display!")
            return
        
        print("\n" + "="*80)
        print("                           TOP 5 MAZE PATHS")
        print("="*80)
        
        for path in top_paths:
            print(f"\n🏆 {path['RANK_SUFFIX']} PLACE")
            print("-" * 60)
            print(f"Path: {path['PATH']}")
            print(f"Score: {path['SCORE']}")
            print(f"Path Length: {path['PATH_LENGTH']} steps")
            print(f"Efficiency: {path['EFFICIENCY']:.2f} points/step")
            print(f"Complexity: {path['PATH_COMPLEXITY']:.2f}")
            print(f"Composite Score: {path['COMPOSITE_SCORE']:.4f}")
        
        print("\n" + "="*80)
    
    def display_statistics(self):
        """
        Display comprehensive statistics in a formatted manner
        """
        stats = self.calculate_statistics()
        
        if not stats:
            print("No statistics available - no data loaded!")
            return
        
        print("\n" + "="*80)
        print("                        MAZE SOLVING STATISTICS")
        print("="*80)
        
        print(f"\n📊 OVERALL RESULTS:")
        print(f"   Total Paths Tested: {stats['total_paths_tested']:,}")
        print(f"   Successful Paths: {stats['successful_paths']:,}")
        print(f"   Failed Paths: {stats['failed_paths']:,}")
        print(f"   Success Rate: {stats['success_rate_percent']}%")
        
        print(f"\n🎯 SCORE ANALYSIS:")
        print(f"   Average Score (All): {stats['average_score_all']}")
        print(f"   Average Score (Successful): {stats['average_score_successful']}")
        print(f"   Average Score (Failed): {stats['average_score_failed']}")
        print(f"   Highest Score: {stats['highest_score']}")
        print(f"   Lowest Score: {stats['lowest_score']}")
        
        print(f"\n🗺️ PATH ANALYSIS:")
        print(f"   Average Path Length: {stats['average_path_length']} steps")
        print(f"   Most Common Path Length: {stats['most_common_path_length']} steps ({stats['most_common_path_length_count']} times)")
        print(f"   Most Used Direction: {stats['most_used_direction'][0]} ({stats['most_used_direction'][1]} times)")
        
        print(f"\n🧭 DIRECTION FREQUENCY:")
        for direction, count in stats['direction_frequency'].items():
            percentage = (count / sum(stats['direction_frequency'].values())) * 100
            print(f"   {direction.title()}: {count} times ({percentage:.1f}%)")
        
        print("\n" + "="*80)
    
    def export_top_paths(self, filename: str = "top_paths.csv"):
        """
        Export top paths to a separate CSV file
        """
        top_paths = self.get_top_paths(10)  # Export top 10
        
        if not top_paths:
            print("No successful paths to export!")
            return
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['RANK', 'PATH', 'SCORE', 'PATH_LENGTH', 'EFFICIENCY', 'COMPLEXITY', 'COMPOSITE_SCORE']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            
            for path in top_paths:
                writer.writerow({
                    'RANK': path['RANK'],
                    'PATH': path['PATH'],
                    'SCORE': path['SCORE'],
                    'PATH_LENGTH': path['PATH_LENGTH'],
                    'EFFICIENCY': round(path['EFFICIENCY'], 2),
                    'COMPLEXITY': round(path['PATH_COMPLEXITY'], 4),
                    'COMPOSITE_SCORE': round(path['COMPOSITE_SCORE'], 4)
                })
        
        print(f"Top paths exported to {filename}")
    
    def run_analysis(self):
        """
        Run the complete analysis workflow
        """
        print("🔍 Starting comprehensive path analysis...")
        
        # Load data
        self.load_csv_data()
        
        if not self.results_data:
            print("No data to analyze!")
            return
        
        # Display statistics
        self.display_statistics()
        
        # Display top paths
        self.display_rankings()
        
        # Export results
        self.export_top_paths()
        
        print("\n✅ Analysis complete!")

# Run the analysis
if __name__ == "__main__":
    analyzer = PathAnalyzer()
    analyzer.run_analysis()