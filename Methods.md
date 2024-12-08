# Methods
Methods (2-4 pages)  
(a) Describe your data and how you obtained it.  
(b) Describe your methodology: give flowcharts, diagrams, pseudocode or formulas where appropriate.  
(c) Describe evaluation strategy.  

## Data Source
We looked at a variety of sources for data. In our analysis we looked at data from two different sources. The first source was the NBA API. This was a user friend API that allowed us to pull a wide variety of data from the NBA. The second source was the Kaggle NBA dataset. The benefit of the API was that it was up to date but it was limited in the amount of data that we could pull. The Kaggle dataset was well organized but had a cutoff date of 2023. For early analysis we used the API but for the data that we actually used to train our model we used the Kaggle dataset.  

## Data Limitations
One of the difficulties in trying to predict the outcome of a basketball game is the wide range of variables that are difficult to track through data. For example, individual player performance can have a huge impact on the outcome of a game but it is a difficult variable to track. One could include several players individual stats in the model but a crucial factor often is related to statistics that are not tracked or not easily tracked which would be is a certain player not playing at their best or is a player injured. 

## Data Structure
We used game data from the Kaggle NBA dataset. Here is a table of the different columns used and a brief description of what they represent.

| Column Name | Description |
| --- | --- |
| game_id | Unique identifier for the game |
| game_date | Date of the game |
| season_type | Type of game (Regular Season, Playoffs, Pre Season) |
| team_id_home / team_id_away | Unique identifier for the home and away team |
| team_name_home / team_name_away | Name of the home and away team |
| pts_home / pts_away | Points scored |
| wl_home / wl_away | Win or Loss |
| fg_pct_home / fg_pct_away | Field Goal Percentage |
| fg3_pct_home / fg3_pct_away | Three Point Field Goal Percentage |
| fg3m_home / fg3m_away | Three Point Field Goals Made |
| ft_pct_home / ft_pct_away | Free Throw Percentage |
| ftm_home / ftm_away | Free Throws Made |
| reb_home / reb_away | Rebounds |
| ast_home / ast_away | Assists |
| stl_home / stl_away | Steals |
| blk_home / blk_away | Blocks |
| tov_home / tov_away | Turnovers |

There were other tables in the database such as team details and player details as well as several other columns in the game table but we only used the columns listed above. 

<!-- effect of recent games vs effect of older games. This concept is known as recency bias. -->
Future work, we could look more closely at the recency bias and weight the most recent games more heavily than older games.


## Email, ML NBA Prediction Data Set

Alright, so I was able to get some slightly decent results just using a feed forward nn. Getting about 60% (59.66% is best I've seen so far) accuracy just trying to predict the result of the alone rather than the scores independently. I just submitted a commit with the data that I used as a csv. I tried with different models and features but didn't spend too much time fine tuning. It seemed to perform better with less features so I have most commented out. I've attached an excel file here too which has more data that we can use for the RNN. 

NBA_Games.xlsx Download this. I think its a lot easy to get a feel for whats going on when the data is present in csv format. For the rnn, it should be pretty smoot to generate json or whatever to use the games sheet for the games and then the history tab for the sequential data. 

I went back to using the Kaggle database because it had the same features that we had identified, and it should be the same data either way. Then I basically was able to write 3 sql queries to get me all of the data that we need. I exported the results from these queries and put them in this excel file with the following sheet names:
Matchup History
Team Game History
Opponent Game History
Then I added another sheet which has the full list of games that I plan for us to use to train and test. This full list of games has 28,926 results. There was a total of 29,079 games played in the Kaggle data set between the dates '2000-01-01' and '2023-06-12'. I had to remove 153, a small fraction (less than 1%), of the games because they did not have enough matchup game history between the two teams playing. 

# 11 features used from previous games. 
features = ['PTS', 'FG_PCT', 'FG3_PCT', 'FG3M', 'FT_PCT', 'FTM', 'REB', 'AST', 'STL', 'BLK', 'TOV']

My plan for this data is that we can use one version of it for the standard feedforward neural network and another more dynamic set of the data that takes into account the sequential nature of the history data that I've collected. 

Slight tangent, but an interesting thing that I've realized while doing this is that categorical data must be converted to numerical representation that the model can process. For each of the features that I thought we might use as categories, I've done some preprocessing to convert them into meaningful numerical data. For the season type which would be either 'Regular Season', 'Playoffs', or 'Pre Season', I've created 3 boolean features called is_regular_season, is_playoffs, and is_pre_season. I thought about doing something similiar for the 30 NBA teams to use as categories but decided against it. I'm not sure it would add much value. Then for the dates of the historical matches, I've converted those to be 'days_ago' which marks the record in relation to the game that we are looking to predict. I would expect the more recent data is more relevant, which is why I think this was a good idea. 

Feedforward neural network - I'm thinking for this model, instead of trying to flatten the sequential data that we just aggregate it using averages. For each of the meaningful stats that the RNN would pass sequentially into the modal, we can just take the average. For matchup history data, we would take the average of the 5 previous matchups. For the team game history and opponent game history we could take the average of up to 10 previous matches. 

Recurrent Neural Network (RNN) - For this one, just like before we now have sequential data for 3 types of series of data. Except it has been massaged a bit so that hopefully it is easier for the model to process. 

For both models, we're using the same features as before. The features are organized in a way so that we no longer have to think in terms of home and away where sometimes the home is home but in the historical data sometimes the team is away. No. Now the data is organized so that all of that stats that are good are in one set of features with the feature label such as "pts_for". And the stats that are represent the opposing time are all in another set of features with the feature label such as "pts_against". And we also have a flag that describes if the team we are predicting for was the home team for that game. 

Results and findings. I'm not sure exactly how we want to do it but I have a few ideas for results. I'm open to suggestions. At the moment I'm think we do a setup like this
Training data: 2001–2020
Validation data: 2021
Test data: 2022–2023
I'm not familiar with using a validation set but read some interesting things. Also sorry I accidentally did not include the year 2000 in our data. I hope thats not a big deal. 

An interesting metric that we can compare our accuracy against is to compare it against Prediction Using (Average of prev 10 games W % greater than 0.50). I ran that for the data and it was much worse than I had expected! 

Prediction Using (Average of prev 10 games W % greater than 0.50)
Actual v Predicted
Win (Pred)
Loss (Pred)
Accuracy
0.553792
Win
8119
8942
Correct
16019
Loss
3965
7900
Total
28926
Prediction Using (Guess Home Team Every Time)
Actual v Predicted
Win (Pred)
Loss (Pred)
Accuracy
0.589815
Win
17061
0
Correct
17061
Loss
11865
0
Total
28926


If we were to just choose whatever team has won on average more over the last 10 games we'd only be right 55% of the time. Predicting the home team to be the winner every time actually has a higher accuracy of about 59%! That was surprising to me. So that is an indicator that other factors are key. I read that momentum and home-court advantage are two largely influential factors. Thats why I worked hard to include the Team Game History and Opponent Game History as well. These should indicate the teams' momentum. I've only pulled the Team Game History so far. I still need to pull the Opp Game History. Hopefully this is enough to start with. 

Thanks,
Noah