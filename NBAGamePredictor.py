import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class NBAGamePredictor(nn.Module):
    # Constructs the model (hidden size is the memory capacity of the RNN, too high will lead to overfitting and too low will be too basic)
    def __init__(self, input_size=11, hidden_size=64):
        super().__init__()

        # Create an embedding layer for sequence type labels, can mess around with the embedding dimensions
        self.sequence_embedding = nn.Embedding(num_embeddings=3, embedding_dim=8)

        # LSTM framework for the input
        self.lstm = nn.LSTM(
            input_size=input_size + 8, # handle the size for both teams' features
            hidden_size=hidden_size,
            num_layers=3,
            batch_first=True,
            dropout=0.2
        )

        # Take in the LSTMs hidden values and predict that as a score
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),  # Make wider
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),  # Add batch normalization
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 2)
        )

        # Debug: Print initial predictor weights
        print("\nInitial predictor weights:")
        for idx, layer in enumerate(self.predictor):
            if isinstance(layer, nn.Linear):
                print(f"Layer {idx} weight norm: {layer.weight.norm().item()}")

    # Defines how data flows through the model, automatically is called when the model is instantiated
    def forward(self, home_seq, away_seq, matchup_seq):
        batch_size = home_seq.shape[0]

        # For the 4 matchup sequences, split into home and away
        matchup_home = matchup_seq[:, 0::2, :] 
        matchup_away = matchup_seq[:, 1::2, :]  

        # Create labels using correct tensor dimensions
        home_labels = torch.zeros((batch_size, home_seq.shape[1]), dtype=torch.long)
        away_labels = torch.ones((batch_size, away_seq.shape[1]), dtype=torch.long)
        matchup_home_labels = torch.full((batch_size, matchup_home.shape[1]), 2, dtype=torch.long)  # 2 sequences
        matchup_away_labels = torch.full((batch_size, matchup_away.shape[1]), 2, dtype=torch.long)  # 2 sequences

        # Get embeddings for each sequence type
        home_embed = self.sequence_embedding(home_labels)
        away_embed = self.sequence_embedding(away_labels)
        matchup_home_embed = self.sequence_embedding(matchup_home_labels)
        matchup_away_embed = self.sequence_embedding(matchup_away_labels)
        
        # Combine features with their embeddings
        home_combined = torch.cat([home_seq, home_embed], dim=2)
        away_combined = torch.cat([away_seq, away_embed], dim=2)
        matchup_home_combined = torch.cat([matchup_home, matchup_home_embed], dim=2)
        matchup_away_combined = torch.cat([matchup_away, matchup_away_embed], dim=2)

        # Combine all sequences
        combined_seq = torch.cat([
            home_combined,          # Regular home games (19 features)
            away_combined,          # Regular away games (19 features)
            matchup_home_combined,  # Home stats from matchups (19 features)
            matchup_away_combined   # Away stats from matchups (19 features)
        ], dim=1)

        _, (hidden, _) = self.lstm(combined_seq)

        scores = self.predictor(hidden[-1])

        return scores

    # A function that trains the model based on passed in iteration amounts as well as a patience counter. Uses validation for early stopping to prevent overfitting
    def train_model(self, num_epochs, train_loader, val_loader, patience=5):
    # At start of training, check a few batches

        # Instantiate mean squared error and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        initial_weights = {name: param.clone().detach() for name, param in self.named_parameters()}

        # Track the losses
        training_losses = []
        validation_losses = []

        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        # The training loop
        for epoch in range(num_epochs):
            self.train()
            total_train_loss = 0

            # Iterate for data in train loader
            for batch_idx, batch in enumerate(train_loader):
                predictions = self(batch['home_sequence'], batch['away_sequence'], batch['matchup_history'])
                loss = criterion(predictions, batch['target'])

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                total_train_loss += loss.item()
            # Print out data for iteration
            avg_train_loss = total_train_loss / len(train_loader)
            training_losses.append(avg_train_loss)

            # Now go through the validation phase
            self.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    predictions = self(batch['home_sequence'], batch['away_sequence'], batch['matchup_history'])
                    val_loss = criterion(predictions, batch['target'])
                    total_val_loss += val_loss
            avg_val_loss = total_val_loss / len(val_loader)
            validation_losses.append(avg_val_loss)

            # Print current epoch data
            print(f'Epoch {epoch+1}/{num_epochs}')
            print(f'Training Loss: {avg_train_loss:.4f}')
            print(f'Validation Loss: {avg_val_loss:.4f}')

            # Check for early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = self.state_dict()
            else:
                patience_counter += 1
                print(f'Early stopping counter: {patience_counter}/{patience}')

            # Quit training if patience has been met
            if patience_counter >= patience:
                print('Early stopping triggered!')
                self.load_state_dict(best_model_state) # Restore the best model state
                break

        # Plot the training and validation losses
        plt.figure(figsize=(10,6))
        plt.plot(training_losses, label='Training Loss', marker='', linestyle='-', linewidth=2, color='blue')
        plt.plot(validation_losses, label='Validation Loss', marker='', linestyle='-', linewidth=2, color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

        return training_losses, validation_losses

    # A function that predicts and evaluates a model
    def evaluate_model(self, test_loader):
        # Initialize all of the stats used to evaluate the model
        self.eval()
        total_games = 0
        total_margin_error = 0
        total_score_error = 0
        correct_winner_predictions = 0

        predictions_list = []

        # Go through each batch and analyze
        with torch.no_grad():
            for batch in test_loader:
                # Get the model predictions and convert to numpy
                predicted_scores = self(batch['home_sequence'], batch['away_sequence'], batch['matchup_history'])
                actual_scores = batch['target']
                pred_scores = predicted_scores.numpy()
                act_scores = actual_scores.numpy()

                batch_size = pred_scores.shape[0]
                total_games += batch_size

                # Now go through every game in the batch and evaluate results
                for i in range(batch_size):
                    pred_home, pred_away = pred_scores[i]
                    actual_home, actual_away = act_scores[i]

                    # Prediction errors
                    pred_margin = pred_home - pred_away
                    actual_margin = actual_home - actual_away
                    margin_error = abs(pred_margin - actual_margin)

                    # Score errors
                    home_score_error = abs(pred_home - actual_home)
                    away_score_error = abs(pred_away - actual_away)
                    avg_score_error = (home_score_error + away_score_error) / 2

                    total_margin_error += margin_error
                    total_score_error += avg_score_error

                    # Check winner prediction
                    pred_winner = 'home' if pred_margin > 0 else 'away'
                    actual_winner = 'home' if actual_margin > 0 else 'away'
                    if pred_winner == actual_winner:
                        correct_winner_predictions += 1

                    # Store prediction details
                    predictions_list.append({
                        'Predicted': {'Home': pred_home, 'Away': pred_away},
                        'Actual': {'Home': actual_home, 'Away': actual_away},
                        'Margin Error': margin_error,
                        'Score Error': avg_score_error,
                        'Correct Winner': pred_winner == actual_winner
                    })

        # Calculate average errors
        avg_margin_error = total_margin_error / total_games
        avg_score_error = total_score_error / total_games
        winner_accuracy = (correct_winner_predictions / total_games) * 100

        print(f"\nModel Evaluation Results:")
        print(f"Total Games Evaluated: {total_games}")
        print(f"Average Margin Error: {avg_margin_error:.2f} points")
        print(f"Average Score Error: {avg_score_error:.2f} points")
        print(f"Winner Prediction Accuracy: {winner_accuracy:.2f}%")

        return predictions_list