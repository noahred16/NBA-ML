# Notes from Meetup at Snell
11/17/2024




# Do a compartive anaylsis to decide if more features are better?
- We can use a basic RNN, one with more features and one with less features. See which one performs better.
- An alternative but acheivable goal would be to compare the RNN performance with a FNN / MLP model that does not use the sequence data.


# Features
- From the game csv, we can use:
- Catgorical: 
    - season_type
    - team_abbreviation_home
    - team_abbreviation_away
- Features: 
    - assists
    - tot rebounds
    - 3pt percentage
    - free throws made
    - points
    - tournovers
 
# Data structure
- Given features that was discussed for the last ten games for each team within the league.
- For example:
    - Boston Celtics
          - Game 1: [12 points, 23 rebounds, 23% 3pt, ...]
          - Game 2: [21 points, 21 rebounds, 23% 3pt, ...]
          - ...
- List of games to train off of from 2000 to 2020

# Action Items
- Agregate some data for this initial POC model.  
- Run a proof of concept given the features above.  
- Consult with the professor to get his thoughts on our choice of model and features.  



