import os
import pickle
import torch
from nba_api.stats.endpoints import leaguegamefinder
import pandas as pd
from collections import defaultdict
from NBAGameDataset import NBAGameDataset
from NBAGamePredictorFCNN import NBAGamePredictorFCNN  # Import the FCNN model
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Build the training data to input into the model 
def build_training_data(games_df, features_to_keep, n_previous_games=10):
    team_histories = defaultdict(list)
    training_samples = []
    game_ids = games_df['GAME_ID'].unique()

    for i, game_id in enumerate(game_ids):
        if i % 1000 == 0:  # Print progress every 1000 games
            print(f"Processed {i} games...")
        game_occurrences = games_df[games_df['GAME_ID'] == game_id]

        home_games = game_occurrences['MATCHUP'].str.contains('vs.')
        away_games = game_occurrences['MATCHUP'].str.contains('@')
                
        if not (home_games.any() and away_games.any()):
            continue

        home_row = game_occurrences[game_occurrences['MATCHUP'].str.contains('vs.')].iloc[0]
        away_row = game_occurrences[game_occurrences['MATCHUP'].str.contains('@')].iloc[0]
        home_team = home_row['TEAM_NAME']
        away_team = away_row['TEAM_NAME']

        if (len(team_histories[home_team]) >= n_previous_games and len(team_histories[away_team]) >= n_previous_games):
            home_history = team_histories[home_team][-n_previous_games:]
            away_history = team_histories[away_team][-n_previous_games:]

            training_samples.append({
                'home_team_history': home_history,
                'away_team_history': away_history,
                'target': {
                    'home_score': home_row['PTS'],
                    'away_score': away_row['PTS']
                },
                'game_date': home_row['GAME_DATE']
            })
        
        home_stats = {feature: home_row[feature] for feature in features_to_keep}
        away_stats = {feature: away_row[feature] for feature in features_to_keep}
        team_histories[home_team].append(home_stats)
        team_histories[away_team].append(away_stats)
    
    print(f"Created {len(training_samples)} training samples")
    return training_samples

def create_data_set(year_start=2000, year_end=2020):
    all_games_list = []
    
    for season in range(year_start, year_end):
        season_id = f"{season}-{str(season+1)[-2:]}"  # Format: "2000-01", "2001-02", etc.
        game_finder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season_id,
            league_id_nullable="00"
        )
        season_games = game_finder.get_data_frames()[0]
        all_games_list.append(season_games)
    
    games_df = pd.concat(all_games_list, ignore_index=True)
    games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])
    games_df = games_df.sort_values('GAME_DATE', ascending=True)

    features_to_keep = [
        'GAME_DATE', 'MATCHUP', 'WL', 'PTS', "FG_PCT", 'FG3_PCT', 'FG3M', 'FT_PCT', 'FTM', 'REB', 'AST', 'STL', 'BLK', 'TOV'
    ]

    training_data = build_training_data(games_df, features_to_keep)

    train_samples, val_samples = train_test_split(
        training_data,
        test_size=0.2,
        random_state=42
    )

    print(f'Training samples: {len(train_samples)}')
    print(f'Validation samples: {len(val_samples)}')

    train_dataset = NBAGameDataset(train_samples)
    val_dataset = NBAGameDataset(val_samples)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader

def save_data_set(train_loader, val_loader, file_path="data_set.pkl"):
    with open(file_path, "wb") as f:
        pickle.dump((train_loader, val_loader), f)
    print(f"Data set saved to {file_path}")

def load_data_set(file_path="data_set.pkl"):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            print(f"Loading data set from {file_path}")
            return pickle.load(f)
    return None, None

# Evaluate the model and calculate accuracy metrics
def evaluate_model(model, val_loader):
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            predictions = model(batch['home_sequence'], batch['away_sequence'])
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(batch['target'].cpu().numpy())

    # Convert lists to numpy arrays
    all_predictions = torch.tensor(all_predictions)
    all_targets = torch.tensor(all_targets)

    # Calculate accuracy metrics
    mae = mean_absolute_error(all_targets, all_predictions)
    mse = mean_squared_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)

    print("Evaluation Results:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R2): {r2:.4f}")

# Main execution
file_name = "nba_train_fcnn"
year_start = 2000
year_end = 2001  # Using just 1 year for now
file_path = f"{file_name}_{year_start}_{year_end}.pkl"
train_loader, val_loader = load_data_set(file_path)

if not train_loader or not val_loader:
    train_loader, val_loader = create_data_set(year_start, year_end)
    save_data_set(train_loader, val_loader, file_path)

# Train or load the FCNN model
model_path = "fcnn_model_v1.pth"
num_epochs = 100
model = NBAGamePredictorFCNN(input_size=11, n_previous_games=10, hidden_size=64)
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("FCNN Model loaded!")
else:
    train_losses, val_losses = model.train_model(num_epochs=num_epochs, train_loader=train_loader, val_loader=val_loader)
    torch.save(model.state_dict(), model_path)
    print("FCNN Model trained and saved!")

# Evaluate the model
evaluate_model(model, val_loader)
