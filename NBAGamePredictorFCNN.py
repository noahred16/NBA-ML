import torch
import torch.nn as nn

class NBAGamePredictorFCNN(nn.Module):
    def __init__(self, input_size=11, n_previous_games=10, hidden_size=64):
        super().__init__()
        self.n_previous_games = n_previous_games
        self.input_size = input_size

        # Fully connected network for home team
        self.home_fcnn = nn.Sequential(
            nn.Linear(input_size * n_previous_games, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # Fully connected network for away team
        self.away_fcnn = nn.Sequential(
            nn.Linear(input_size * n_previous_games, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # Predictor network to combine home and away features and predict scores
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 2)  # Predict both team scores
        )

    def forward(self, home_seq, away_seq):
        # Flatten the sequences for each team
        home_flat = home_seq.view(home_seq.size(0), -1)  # Flatten (batch_size, n_previous_games * input_size)
        away_flat = away_seq.view(away_seq.size(0), -1)  # Flatten

        # Pass through the FCNN for each team
        home_features = self.home_fcnn(home_flat)
        away_features = self.away_fcnn(away_flat)

        # Combine features and predict scores
        combined = torch.cat([home_features, away_features], dim=1)
        scores = self.predictor(combined)

        return scores

    def train_model(self, num_epochs, train_loader, val_loader, patience=5):
        # Instantiate mean squared error and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        # Track the losses
        training_losses = []
        validation_losses = []

        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        # Training loop
        for epoch in range(num_epochs):
            self.train()
            total_train_loss = 0

            # Training phase
            for batch in train_loader:
                # Forward pass
                predictions = self(batch['home_sequence'], batch['away_sequence'])

                # Compute loss
                loss = criterion(predictions, batch['target'])

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            # Average training loss
            avg_train_loss = total_train_loss / len(train_loader)
            training_losses.append(avg_train_loss)

            # Validation phase
            self.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    predictions = self(batch['home_sequence'], batch['away_sequence'])
                    val_loss = criterion(predictions, batch['target'])
                    total_val_loss += val_loss.item()
            avg_val_loss = total_val_loss / len(val_loader)
            validation_losses.append(avg_val_loss)

            # Print losses
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Training Loss: {avg_train_loss:.4f}")
            print(f"Validation Loss: {avg_val_loss:.4f}")

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = self.state_dict()
            else:
                patience_counter += 1
                print(f"Early stopping counter: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered!")
                self.load_state_dict(best_model_state)
                break

        return training_losses, validation_losses
