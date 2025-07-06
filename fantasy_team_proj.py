import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import random
from collections import defaultdict, Counter
import time
import warnings
warnings.filterwarnings('ignore')

class FantasyTeamSimulator:
    def __init__(self, csv_file_path):
        """
        Initialize the Fantasy Team Simulator
        
        Parameters:
        csv_file_path (str): Path to the CSV file containing player data
        """
        self.csv_file_path = csv_file_path
        self.players_df = None
        self.teams = []
        self.player_selection_count = defaultdict(int)
        self.role_players = {}
        
    def load_data(self):
        """
        Load player data from CSV file and organize by roles
        """
        print("📊 Loading player data...")
        try:
            self.players_df = pd.read_csv(self.csv_file_path)
            print(f"✅ Successfully loaded {len(self.players_df)} players")
            
            # Display basic info about the dataset
            print("\n📋 Dataset Overview:")
            print(self.players_df.head())
            print(f"\n📈 Dataset Shape: {self.players_df.shape}")
            print(f"\n🎯 Roles Distribution:")
            print(self.players_df['role'].value_counts())
            
            # Organize players by role for efficient sampling
            self.role_players = {
                role: self.players_df[self.players_df['role'] == role].to_dict('records')
                for role in self.players_df['role'].unique()
            }
            
            print(f"\n🔍 Players by Role:")
            for role, players in self.role_players.items():
                print(f"  {role}: {len(players)} players")
                
        except FileNotFoundError:
            print(f"❌ Error: Could not find file '{self.csv_file_path}'")
            print("📁 Please ensure the CSV file is in the same directory as this script")
            raise
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            raise
    
    def weighted_random_selection(self, players_list, num_select, exclude_players=None):
        """
        Select players using weighted random selection based on perc_selection
        
        Parameters:
        players_list (list): List of player dictionaries
        num_select (int): Number of players to select
        exclude_players (set): Set of player codes to exclude
        
        Returns:
        list: Selected players
        """
        if exclude_players is None:
            exclude_players = set()
        
        # Filter out excluded players
        available_players = [p for p in players_list if p['player_code'] not in exclude_players]
        
        if len(available_players) < num_select:
            return available_players
        
        # Extract weights (perc_selection values)
        weights = [p['perc_selection'] for p in available_players]
        
        # Normalize weights to ensure they sum to 1
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1/len(available_players)] * len(available_players)
        
        # Use numpy's random choice for weighted selection
        selected_indices = np.random.choice(
            len(available_players), 
            size=num_select, 
            replace=False, 
            p=weights
        )
        
        return [available_players[i] for i in selected_indices]
    
    def generate_valid_team(self, max_attempts=1000):
        """
        Generate a single valid 11-player team meeting all constraints
        
        Parameters:
        max_attempts (int): Maximum attempts to generate a valid team
        
        Returns:
        list: List of 11 selected players, or None if failed
        """
        for attempt in range(max_attempts):
            selected_players = []
            used_player_codes = set()
            
            # Ensure at least one player from each role
            for role in ['WK', 'Batsman', 'Bowler', 'Allrounder']:
                if role in self.role_players and len(self.role_players[role]) > 0:
                    role_selection = self.weighted_random_selection(
                        self.role_players[role], 
                        1, 
                        used_player_codes
                    )
                    if role_selection:
                        selected_players.extend(role_selection)
                        used_player_codes.update(p['player_code'] for p in role_selection)
            
            # If we couldn't get one from each role, try again
            if len(selected_players) < 4:
                continue
            
            # Fill remaining positions (11 - 4 = 7 players)
            remaining_needed = 11 - len(selected_players)
            
            # Get all available players not already selected
            all_available = []
            for role_players in self.role_players.values():
                for player in role_players:
                    if player['player_code'] not in used_player_codes:
                        all_available.append(player)
            
            if len(all_available) >= remaining_needed:
                additional_players = self.weighted_random_selection(
                    all_available, 
                    remaining_needed, 
                    used_player_codes
                )
                selected_players.extend(additional_players)
                
                if len(selected_players) == 11:
                    # Sort by player_code for consistency in team comparison
                    team = sorted(selected_players, key=lambda x: x['player_code'])
                    return team
        
        return None
    
    def generate_teams(self, num_teams=20000, progress_interval=1000):
        """
        Generate specified number of unique teams
        
        Parameters:
        num_teams (int): Number of teams to generate
        progress_interval (int): Interval for progress updates
        """
        print(f"\n🎯 Generating {num_teams} unique teams...")
        self.teams = []
        self.player_selection_count = defaultdict(int)
        unique_teams = set()
        
        start_time = time.time()
        attempts = 0
        
        while len(self.teams) < num_teams:
            attempts += 1
            team = self.generate_valid_team()
            
            if team is not None:
                # Create a unique identifier for the team
                team_id = tuple(sorted([p['player_code'] for p in team]))
                
                # Check if team is unique
                if team_id not in unique_teams:
                    unique_teams.add(team_id)
                    self.teams.append(team)
                    
                    # Update player selection counts
                    for player in team:
                        self.player_selection_count[player['player_code']] += 1
                    
                    # Progress update
                    if len(self.teams) % progress_interval == 0:
                        elapsed = time.time() - start_time
                        rate = len(self.teams) / elapsed
                        print(f"✅ Generated {len(self.teams)} teams ({rate:.1f} teams/sec)")
            
            # Safety check to avoid infinite loops
            if attempts > num_teams * 10:
                print(f"⚠️ Warning: Stopped after {attempts} attempts. Generated {len(self.teams)} unique teams.")
                break
        
        total_time = time.time() - start_time
        print(f"\n🏆 Successfully generated {len(self.teams)} unique teams in {total_time:.2f} seconds")
        print(f"📊 Total attempts: {attempts}")
        print(f"⚡ Generation rate: {len(self.teams)/total_time:.1f} teams/second")
    
    def analyze_results(self):
        """
        Analyze the results and compare with expected selection probabilities
        """
        print("\n📊 Analyzing Results...")
        
        # Calculate actual selection percentages
        total_teams = len(self.teams)
        results = []
        
        for _, player in self.players_df.iterrows():
            player_code = player['player_code']
            expected_selections = player['perc_selection'] * total_teams
            actual_selections = self.player_selection_count[player_code]
            actual_percentage = actual_selections / total_teams
            
            # Calculate absolute error
            absolute_error = abs(actual_percentage - player['perc_selection'])
            percentage_error = (absolute_error / player['perc_selection']) * 100 if player['perc_selection'] > 0 else 0
            
            # Check if within ±5% threshold
            within_threshold = absolute_error <= 0.05
            
            results.append({
                'player_code': player_code,
                'player_name': player['player_name'],
                'role': player['role'],
                'team': player['team'],
                'expected_perc': player['perc_selection'],
                'actual_perc': actual_percentage,
                'expected_selections': expected_selections,
                'actual_selections': actual_selections,
                'absolute_error': absolute_error,
                'percentage_error': percentage_error,
                'within_threshold': within_threshold
            })
        
        self.results_df = pd.DataFrame(results)
        
        # Calculate summary statistics
        players_within_threshold = self.results_df['within_threshold'].sum()
        total_players = len(self.results_df)
        
        print(f"\n📈 SIMULATION RESULTS SUMMARY")
        print(f"{'='*50}")
        print(f"🎯 Total Teams Generated: {total_teams:,}")
        print(f"👥 Total Players: {total_players}")
        print(f"✅ Players within ±5% error: {players_within_threshold}/{total_players}")
        print(f"🎯 Success Rate: {(players_within_threshold/total_players)*100:.1f}%")
        
        # Check if we meet the requirement (at least 20 out of 22 players)
        if players_within_threshold >= 20:
            print(f"🏆 SUCCESS: Simulation meets the requirement!")
        else:
            print(f"❌ NEEDS IMPROVEMENT: Only {players_within_threshold} players within threshold")
        
        # Show players exceeding threshold
        exceeding_players = self.results_df[~self.results_df['within_threshold']]
        if len(exceeding_players) > 0:
            print(f"\n⚠️ Players exceeding ±5% error threshold:")
            for _, player in exceeding_players.iterrows():
                print(f"  {player['player_name']}: {player['absolute_error']:.4f} error")
        
        return self.results_df
    
    def create_visualizations(self):
        """
        Create visualizations to show the results
        """
        print("\n📊 Creating Visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Fantasy Team Simulation Results', fontsize=16, fontweight='bold')
        
        # 1. Expected vs Actual Selection Comparison
        ax1 = axes[0, 0]
        ax1.scatter(self.results_df['expected_perc'], self.results_df['actual_perc'], 
                   alpha=0.7, s=60, c='blue')
        ax1.plot([0, 1], [0, 1], 'r--', alpha=0.8, label='Perfect Match')
        ax1.set_xlabel('Expected Selection %')
        ax1.set_ylabel('Actual Selection %')
        ax1.set_title('Expected vs Actual Selection Percentages')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Absolute Error Distribution
        ax2 = axes[0, 1]
        ax2.hist(self.results_df['absolute_error'], bins=15, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(x=0.05, color='red', linestyle='--', label='±5% Threshold')
        ax2.set_xlabel('Absolute Error')
        ax2.set_ylabel('Number of Players')
        ax2.set_title('Distribution of Absolute Errors')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Selection Frequency by Role
        ax3 = axes[1, 0]
        role_stats = self.results_df.groupby('role').agg({
            'expected_perc': 'mean',
            'actual_perc': 'mean'
        }).reset_index()
        
        x = np.arange(len(role_stats))
        width = 0.35
        ax3.bar(x - width/2, role_stats['expected_perc'], width, label='Expected', alpha=0.8)
        ax3.bar(x + width/2, role_stats['actual_perc'], width, label='Actual', alpha=0.8)
        ax3.set_xlabel('Role')
        ax3.set_ylabel('Average Selection %')
        ax3.set_title('Average Selection % by Role')
        ax3.set_xticks(x)
        ax3.set_xticklabels(role_stats['role'])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Error Analysis by Player
        ax4 = axes[1, 1]
        colors = ['green' if x else 'red' for x in self.results_df['within_threshold']]
        bars = ax4.bar(range(len(self.results_df)), self.results_df['absolute_error'], 
                      color=colors, alpha=0.7)
        ax4.axhline(y=0.05, color='red', linestyle='--', label='±5% Threshold')
        ax4.set_xlabel('Player Index')
        ax4.set_ylabel('Absolute Error')
        ax4.set_title('Error by Player (Green=Within Threshold)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Create detailed results table
        print("\n📋 DETAILED RESULTS TABLE")
        print("="*100)
        display_df = self.results_df.copy()
        display_df['expected_perc'] = display_df['expected_perc'].round(4)
        display_df['actual_perc'] = display_df['actual_perc'].round(4)
        display_df['absolute_error'] = display_df['absolute_error'].round(4)
        display_df['within_threshold'] = display_df['within_threshold'].map({True: '✅', False: '❌'})
        
        print(display_df[['player_name', 'role', 'expected_perc', 'actual_perc', 
                         'absolute_error', 'within_threshold']].to_string(index=False))
    
    def run_simulation(self, num_teams=20000):
        """
        Run the complete simulation pipeline
        
        Parameters:
        num_teams (int): Number of teams to generate
        """
        print("🚀 Starting Fantasy Team Simulation")
        print("="*50)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Generate teams
        self.generate_teams(num_teams)
        
        # Step 3: Analyze results
        results_df = self.analyze_results()
        
        # Step 4: Create visualizations
        self.create_visualizations()
        
        print("\n🎉 Simulation Complete!")
        return results_df


def main():
    """
    Main function to run the simulation
    """
    csv_file_path = "C:/Desktop/FTS_Proj/player_data_sample.csv"  
    
    
    try:
        simulator = FantasyTeamSimulator(csv_file_path)
        results = simulator.run_simulation(num_teams=20000)
        
        
        results.to_csv("simulation_results.csv", index=False)
        print(f"\n💾 Results saved to 'simulation_results.csv'")
        
    except Exception as e:
        print(f"❌ Error during simulation: {str(e)}")
        print("\n🔧 Troubleshooting Tips:")
        print("1. Ensure 'player_data_sample.csv' is in the same directory")
        print("2. Check that the CSV file has the correct columns")
        print("3. Make sure you have all required libraries installed")

if __name__ == "__main__":
    main()