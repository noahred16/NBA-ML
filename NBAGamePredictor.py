import torch
import torch.nn as nn

class NBAGamePredictor(nn.Module):

    # Constructs the model (hidden size is the memory capacity of the RNN, too high will lead to overfitting and too low will be too basic)
    def __init__(self, input_size=11, hidden_size=64):
        super().__init__()

        # LSTM for home team
        self.home_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True
        )

        # LSTM for away team
        self.away_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True
        )

        # Network for combining the features and creating an output
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), # Combines the team features, first layer
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2), # Second layer
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 2) # Predict both teams scores (output layer)
        )

    # Defines how data flows through the model, automatically is called when the model is instantiated
    def forward(self, home_seq, away_seq):
        # Process home and away team sequences
        home_out, (home_hidden, _) = self.home_lstm(home_seq)
        away_out, (away_hidden, _) = self.away_lstm(away_seq)

        # Use the final hidden states
        home_features = home_hidden[-1]
        away_features = away_hidden[-1]

        # Combine the features and predict
        combined = torch.cat([home_features, away_features], dim=1)
        scores = self.predictor(combined)

        return scores

    # A function that trains the model based on passed in iteration amounts as well as a patience counter. Uses validation for early stopping to prevent overfitting
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

        # The training loop
        for epoch in range(num_epochs):
            self.train()
            total_train_loss = 0

            # Iterate for data in train loader
            for batch in train_loader:
                # Feed data and do forward pass
                predictions = self(batch['home_sequence'], batch['away_sequence'])

                # Calculate the loss
                loss = criterion(predictions, batch['target'])

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
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
                    predictions = self(batch['home_sequence'], batch['away_sequence'])
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
        
        return training_losses, validation_losses