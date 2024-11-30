from nba_api.stats.endpoints import leaguegamefinder
import pandas as pd
from collections import defaultdict
from NBAGameDataset import NBAGameDataset
from NBAGamePredictor import NBAGamePredictor
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Build the training data to input into the model 
def build_training_data(games_df, features_to_keep, n_previous_games=10):
    # initialize dict to store the games
    team_histories = defaultdict(list)
    training_samples = []
    # Get all the unique game IDs so we can sort through them
    game_ids = games_df['GAME_ID'].unique()

    # Go through every id, add game to the respective home and away team, then add a training set expample as well
    for i, game_id in enumerate(game_ids):
        if i % 1000 == 0:  # Print progress every 1000 games
            print(f"Processed {i} games...")
        game_occurrences = games_df[games_df['GAME_ID'] == game_id]

        # Get the home and away if vs is present otherwise discard and continue
        home_games = game_occurrences['MATCHUP'].str.contains('vs.')
        away_games = game_occurrences['MATCHUP'].str.contains('@')
                
        if not (home_games.any() and away_games.any()):
            continue

        print(f'Game occurrences: {game_occurrences}')
        home_row = game_occurrences[game_occurrences['MATCHUP'].str.contains('vs.')].iloc[0]
        away_row = game_occurrences[game_occurrences['MATCHUP'].str.contains('@')].iloc[0]
        home_team = home_row['TEAM_NAME']
        away_team = away_row['TEAM_NAME']

        # Now create training data if both teams have enough history
        if (len(team_histories[home_team]) >= n_previous_games and len(team_histories[away_team]) >= n_previous_games):
            # Get the last ten games
            home_history = team_histories[home_team][-n_previous_games:]
            away_history = team_histories[away_team][-n_previous_games:]

            # Create training set
            training_samples.append({
                'home_team_history': home_history,
                'away_team_history': away_history,
                'target': {
                    'home_score': home_row['PTS'],
                    'away_score': away_row['PTS']
                },
                'game_date': home_row['GAME_DATE']
            })
        
        # Add the game to both teams histories
        home_stats = {feature: home_row[feature] for feature in features_to_keep}
        away_stats = {feature: away_row[feature] for feature in features_to_keep}
        team_histories[home_team].append(home_stats)
        team_histories[away_team].append(away_stats)
    
    print(f"Created {len(training_samples)} training samples")
    return training_samples

# Go through and find all games between 2000 and 2020 and create a training set based off of them
def create_data_set():
    # Create empty list to store all dataframes
    all_games_list = []
    
    # Loop through seasons from 2000 to 2020
    for season in range(2000, 2001):
        season_id = f"{season}-{str(season+1)[-2:]}"  # Format: "2000-01", "2001-02", etc.
        game_finder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season_id,
            league_id_nullable="00"
        )
        season_games = game_finder.get_data_frames()[0]
        all_games_list.append(season_games)
    
    # Combine all seasons
    games_df = pd.concat(all_games_list, ignore_index=True)

    # Convert game date to datetime
    games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])
    
    # Sort games by date
    games_df = games_df.sort_values('GAME_DATE', ascending=True)

    # Features we want to keep
    features_to_keep = [
        'GAME_DATE', 'MATCHUP', 'WL', 'PTS', "FG_PCT", 'FG3_PCT', 'FG3M', 'FT_PCT', 'FTM', 'REB', 'AST', 'STL', 'BLK', 'TOV'
    ]

    training_data = build_training_data(games_df, features_to_keep)

    # Split the data into training and validation samples
    train_samples, val_samples = train_test_split(
        training_data,
        test_size=0.2, # Validation set 20% of all samples
        random_state=42
    )

    print(f'Training samples: {len(train_samples)}')
    print(f'Validation samplesL {len(val_samples)}')

    # Crsate the separate datasets for training and validation and return those
    train_dataset = NBAGameDataset(train_samples)
    val_dataset = NBAGameDataset(val_samples)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader

# Create dataset and data loader and train the model
train_loader, val_loader = create_data_set()
model = NBAGamePredictor()
train_losses, val_losses = model.train_model(num_epochs=10000, train_loader=train_loader, val_loader=val_loader)