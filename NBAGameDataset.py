import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class NBAGameDataset(Dataset):
    def __init__(self, training_samples, normalize=True):
        self.samples = training_samples
        self.features = ['PTS', "FG_PCT", 'FG3_PCT', 'FG3M', 'FT_PCT', 'FTM', 'REB', 'AST', 'STL', 'BLK', 'TOV']

        # Normalize the data
        if normalize:
            self.scaler = StandardScaler()
            # Collect all of the features for normalization
            all_sequences = []
            for i, sample in enumerate(training_samples):
                if i % 1000 == 0:  # Print progress every 1000 games
                    print(f"Normalized and added {i} games...")

                # Add team histories
                all_sequences.extend([[game[feature] for feature in self.features] for game in sample['home_team_history']])
                all_sequences.extend([[game[feature] for feature in self.features] for game in sample['away_team_history']])

                # Add previous matchups
                all_sequences.extend([[game['home_stats'][feature] for feature in self.features] for game in sample['previous_matchups']])
                all_sequences.extend([[game['away_stats'][feature] for feature in self.features] for game in sample['previous_matchups']])

                self.scaler.fit(all_sequences)
        else:
            self.scaler = None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Extract home and away features
        home_history = sample['home_team_history']
        home_features = [[game[feature] for feature in self.features] for game in home_history]
        away_history = sample['away_team_history']
        away_features = [[game[feature] for feature in self.features] for game in away_history]

         # Extract matchup features
        matchup_features = []
        for matchup in sample['previous_matchups']:
            home_matchup = [matchup['home_stats'][feature] for feature in self.features]
            away_matchup = [matchup['away_stats'][feature] for feature in self.features]
            matchup_features.extend([home_matchup, away_matchup])

        # Normalize if the scaler exists
        if self.scaler:
            home_features = self.scaler.transform(home_features)
            away_features = self.scaler.transform(away_features)
            matchup_features = self.scaler.transform(matchup_features)

        # Convert to tensors
        home_tensor = torch.FloatTensor(home_features)
        away_tensor = torch.FloatTensor(away_features)
        matchup_tensor = torch.FloatTensor(matchup_features)
        target = torch.FloatTensor([sample['target']['home_score'], sample['target']['away_score']])
        return {
            'home_sequence': home_tensor,
            'away_sequence': away_tensor,
            'matchup_history': matchup_tensor,
            'target': target
        }
