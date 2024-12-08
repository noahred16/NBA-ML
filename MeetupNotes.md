# Notes from Meetup at Snell
11/17/2024




# Do a compartive anaylsis to decide if more features are better?
- We can use a basic RNN, one with more features and one with less features. See which one performs better.
- An alternative but acheivable goal would be to compare the RNN performance with a FNN / MLP model that does not use the sequence data.
- Comparing RNN to normal feedforward network
- RNN structure will be composed of two LSTMs for the home and away team to predict score, where there will be another layer that then takes those predictions as features to predict the final outcome


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
- Make sure that the previous ten games of each team is continuously being updated throughout the season as we train. Can games from prior season as well

# Output
- Model should output final score from home team perspective. For example, [128,120], home team scored 128 and away scored 100.

# Action Items
- Agregate some data for this initial POC model.  
- Run a proof of concept given the features above.  
- Consult with the professor to get his thoughts on our choice of model and features.  




## 12-7-2024 notes
This is a tight matchup so its hard to predict the score. I can try to pull a different example but here are my thoughts. 

We should think from the perspective of either the home team or the away team, not both. Lets think in terms of the home team and predict how many points they will score and how many points they will allow. Then we can use that to predict the final outcome of the game.

For this example (2018-10-29): New York Knicks (NYK) vs Brooklyn Nets (BKN), NYK are home. So we will think in terms of NYK. So they won, 115 to 96. Then we can look at their previous matchups. 

Game To Predict (2018-10-29) 
    Feature:
        - is_home = 1
        - game_type = 'regular_season'
    Result:
        - scored: 115
        - allowed: 96

Prev Game (2018-10-19)
    Feature:
        - is_home = 0
        - game_type = 'regular_season'
    Result:
        - scored: 105
        - allowed: 107

Prev Game (2018-10-03)
    Feature:
        - is_home = 0
        - game_type = 'pre_season'
    Result:
        - scored: 107
        - allowed: 102

So then our input could be simplified to look like this:
{
    "is_home": 1,
    "game_type": "regular_season",
    "prev_matchups": [
        {
            "is_home": 0,
            "game_type": "regular_season",
            "scored": 105,
            "allowed": 107,
            "REB_for": 36,
            "FG3M_against": 55
        },
        {
            "is_home": 0,
            "game_type": "pre_season",
            "scored": 107,
            "allowed": 102,
            "REB_for": 49,
            "FG3M_against": 45
        }
    ]
}




features = ['PTS', "FG_PCT", 'FG3_PCT', 'FG3M', 'FT_PCT', 'FTM', 'REB', 'AST', 'STL', 'BLK', 'TOV']

{
    "is_home": 1,
    "game_type": "regular_season",
    "prev_matchups": [
        {
            "is_home": 0,
            "game_type": "regular_season",
            "PTS_for": 105,
            "PTS_agt": 107,
            "FG_PCT_for": 0.45,
            "FG_PCT_agt": 0.55,
            "FG3_PCT_for": 0.35,
            "FG3_PCT_agt": 0.45,
            "FG3M_for": 10,
            "FG3M_agt": 15,
            "FT_PCT_for": 0.75,
            "FT_PCT_agt": 0.85,
            "FTM_for": 15,
            "FTM_agt": 20,
            "REB_for": 36,
            "REB_agt": 49,
            "AST_for": 20,
            "AST_agt": 25,
            "STL_for": 5,
            "STL_agt": 10,
            "BLK_for": 5,
            "BLK_agt": 10,
            "TOV_for": 10,
            "TOV_agt": 15
        },
        { prev matchup game ...},
        { prev matchup game ...}.
        { prev matchup game ...},
        { prev matchup game ...},
    ]
}



Recording this here so I don't forget and so we can include in the procedures. 



Doing data validation stuff and learning fun facts about the past 20 years of NBA history. 
 
Apparently the "New Orleans Pelicans" were previously known as "New Orleans Hornets". And even crazier is they are listed in a period from 2005 to 2007 under the name "New Orleans/Oklahoma City Hornets" from when they were forced to relocated to Oklahoma City temporarily because of Hurricane Katrine. 
 
And then I was curious to see if that team ever played OKC during that time period bc at that point both teams might have been home and I found out that OKC didn't even exist at that point. There was a team called "Seattle SuperSonics" which relocated to Oklahoma City in 2008 and was renamed OKC. 
 
As part of the data validation, I have filtered out the All-Start games because those games don't have enough context to predict. I also remove duplicate records from the dataset. I did some basic validation to confirm that all rows where the home team had a higher score were marked as 'W' for the home team. I checked fixed some other data where the score was records but the W/L was not. 




Lol another fun fact, apparently in 2007 the Celtics player a game in Worcester, Mass against the nets but the basketball court had been placed over a minor league hockey rink. When the rink ice began melting, condensation built up on the court. The game ended up being canceled but the Celtics were leading, 36-33, when it was called. Probably excluding that one from the training data. Its one of the 2 records where there is no result for who won. 
https://www.netsdaily.com/2007/10/19/1351711/rained-out