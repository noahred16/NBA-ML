# Predicting NBA Game Outcomes Using Team and Player - Statistics with RNNs
Ali Daeihagh, Dhruv Belai, Noah Smith  
Khoury College of Computer Sciences  
Northeastern University  
Boston, MA  
daeihagh.a, belai.d, smith.noah @northeastern.edu  
October 29, 2024  

## 1. Objectives and Significance
In recent years, sports betting has rapidly grown into one of America’s largest industries,
allowing individuals to place bets on a variety of sports. The introduction of predictive models has
proven highly beneficial in enhancing decision-making for betting on game outcomes, as these
models provide valuable insights into game dynamics, enabling bettors to make more informed
wagers. Beyond betting, predictive models are also highly useful for team front offices, which can
use them to identify key factors that contribute to their team’s performance. By analyzing these
insights, teams can pinpoint areas for improvement in practice, focusing on the elements that most
significantly impact game outcomes.  

Predictive modeling also influences sports journalism, shaping discussions, debates, and
narratives. Talk shows and media increasingly rely on these predictions to create engaging content
and capture audience interest. With these developments, there is a clear demand for accurate
models to predict sports outcomes. While past models have shown success, there remains
significant room for improvement. Efforts to enhance predictive accuracy may involve refining
the set of key features considered and exploring various model architectures to achieve better
precision.  

## 2. Background
Applying machine learning to determine the outcomes of sports games is not necessarily a new
endeavor. Decades of research have introduced diverse methodologies to forecast various aspects
of sports. For example, within the NBA alone, machine learning has been used to predict playoff
contenders, All-Star selections, game outcomes, as well as to determine the viability of the “Hot
Streak Fallacy” [7]. While the specific approach varies based on the prediction goal, it is crucial
to recognize the range of methodologies available to address different aspects of sports prediction.  

Traditionally, sports prediction has been approached as a classification problem, though numeric
methods have also been employed [2, 1]. For instance, Delen et al. (2012) used regression-based
models to predict NCAA bowl game outcomes, employing neural networks, decision trees, and
support vector machines. However, when comparing these models with classification-based
approaches, they found that classification models provided more accurate game outcome
predictions. This preference for classification is widely supported across research groups, and there
is a general consensus that sports predictions are best suited to classification approaches [1].  

In classification-based sports prediction, a variety of models have been explored. These range from
K-Nearest Neighbor clustering, support vector machines, and naive Bayes to neural networks and
decision trees [7, 1, 5]. Each model has its own strengths and weaknesses, yet neural networks are
among the most commonly used and accurate methods in sports prediction [1]. Neural networks
have proven effective at creating precise classification models across other fields that rely on
extensive input data [3].  

NBA game prediction models have employed a wide range of input features to improve predictive
accuracy. Team statistics have been traditionally used, with Thabta et al. (2019) achieving 83%
accuracy in game predictions based on 20 team-based statistics per matchup. Among the possible
team features to include when predicting, defensive rebounds, three point percentages, free throws
made, and total rebounds are the most influential for improving model accuracy [6]. It is worth
noting that player based statistics have not always been utilized when developing these models.
This could be detrimental as it overlooks the impact that individual players may have on the game.
There are many instances in NBA games where an individual player can single handedly take over
and dominate a game, which makes the inclusion of player based statistics all the more important.
In addition, many studies miss out on the temporal aspect of team and player statistics when
developing and predicting their models. This could be particularly important because teams often
hit a period of momentum or a winning streak within their season in which they may be playing
better than they were before. Considering this could offer marked improvements in predictive
accuracy [5].  

Khanmohammadi et al. (2024) have built on these areas for improvement by introducing a
recurrent neural network that integrates both player and team statistics to predict NBA game
outcomes. This network's recurrent structure allows it to capture the temporal dynamics of team
and player performance, while the addition of player-based statistics addresses previous models'
shortcomings. Their model achieved an 82% accuracy in predicting NBA game outcomes [5].
Drawing on prior research, we aim to further investigate the interplay between player and team
statistics and explore how combining both within a neural network might enhance prediction
accuracy in comparison to the results achieved by Khanmohammadi et al.  

## 3. Proposed Approach
Our project centers around the extensive NBA Database [4] available on Kaggle, a resource that
spans over 65,000 games, capturing every NBA matchup since the league’s very first season in
1946-47. This dataset not only covers comprehensive information on games, teams, players, and
referees but also includes an extensive play-by-play record with over 13 million rows, providing a
detailed breakdown of in-game events. This wealth of data offers a rich foundation for analyzing
trends, extracting insights, and making data-driven predictions for current games.  

With over 15 interconnected tables, each offering unique perspectives on NBA games, we will
selectively utilize key features across these tables to build a predictive model with greater accuracy
and depth than previous attempts. By focusing on nuanced, relevant aspects of gameplay and
player performance, this approach aims to push the boundaries of game outcome forecasting,
leveraging both historical patterns and real-time analysis for an edge in predictive power.  

Our project aims to predict the outcomes of NBA games for the 2024-2025 season—specifically,
predicting game results in terms of win or loss. With access to a robust dataset covering historical
team, player, and play-by-play data, we will leverage this resource to build a model that accurately
reflects the dynamics of live NBA games. Our approach combines aggregated team and player
statistics with sequential play-by-play information, enabling us to capture both static trends and
real-time in-game shifts.  

### Data Aggregation and Preprocessing
For team statistics, we will compile metrics like points per game, offensive and defensive ratings,
assists, rebounds, and turnovers for each team. Given the scale of our data, our preprocessing
strategy includes aggregating these metrics across multiple seasons, with normalization to account
for evolving play styles and rule changes. These metrics will give a high-level view of team
performance and tendencies, providing reliable features that will complement real-time game data.  

For player statistics, we’ll extract individual contributions such as shooting percentage, assists,
blocks, and fouls. These will then be aggregated to create team-level impact features that reflect
the influence of each player’s contributions on game outcomes. To capture player availability
realistically, we will weight player contributions based on average playing time and adjust for any
injuries or absences, thus preserving the consistency and accuracy of our team-level features. This
aggregation will allow us to quantify each team’s current roster strength, factoring in active versus
injured players for a more accurate prediction of game-day performance.  

### Feature Engineering for Play-by-Play Data
The play-by-play data is structured sequentially, capturing each action in chronological order. We
will utilize this format to build a comprehensive view of game flow, extracting features such as
possessions, scoring runs, turnovers, and fouls, which are key to understanding momentum shifts.
For efficient modeling, we will employ feature engineering to standardize this data across games
by padding or truncating sequences to a fixed length, allowing the model to process and interpret
each game sequence effectively. This standardized approach ensures that the RNN processes
sequences of comparable structure, making training more stable and generalizable across games.  

### Exploratory Data Analysis (EDA)
To uncover underlying patterns, we plan to perform exploratory data analysis (EDA) on our
aggregated features. Key visualizations will include correlation matrices, which will help us
pinpoint relationships between features and game outcomes, as well as trend analyses of play-by-play
sequences to identify impactful in-game events. This analysis will guide our feature selection,
helping us identify and prioritize features with strong predictive value. Additionally, EDA will
provide a solid foundation for hypothesis testing and further validation of our model design.

### Model Architecture
Our model architecture will integrate both static and dynamic data using a hybrid approach:

1. RNN for Sequential Play-by-Play Data:
- To capture the temporal dependencies in play-by-play sequences, we will implement a
Long Short-Term Memory (LSTM) network. LSTMs are particularly suited for
handling the sequence of events in a game, as they can learn dependencies between plays
that span over different timescales.
- Each action within the sequence will be encoded into a vector representing event types
(e.g., scoring play, turnover, foul), and the LSTM will output a vector summarizing the
impact of these events on the overall game momentum and likely outcome.

2. Tree-Based Model for Team and Player Statistics:
- For processing team and player statistics, we will employ a tree-based model such as
XGBoost or Random Forests. These models handle structured, tabular data effectively
and are well-suited to learning from aggregated statistics, capturing non-linear
relationships and interactions among features.
- This tree-based model will allow us to incorporate static data reliably, providing a
prediction layer that represents pre-game factors such as team strength and player
availability.

3. Combining RNN and Static Data Outputs:
- After training both the LSTM and the tree-based model, we will combine their outputs
by concatenating the feature vectors from each model. This hybrid vector will then pass
through a dense layer that learns how to integrate the insights from both models,
capturing the combined influence of pre-game statistics and in-game events on the final
outcome.
- The dense layer will enable the model to weigh the relative importance of static and
dynamic data, providing a holistic prediction that considers both game context and team
dynamics.  

### Model Evaluation
Predicting game outcomes on a week-by-week basis could be interesting, but separating training
and test data by a historical cut-off date might yield a more consistent evaluation. By using this
approach, we can start evaluating our model’s accuracy from a fixed point in time and measure its
performance on a more substantial dataset.  

To enhance our evaluation, we can incorporate the final game scores to adjust the results. Including
the score in our prediction accuracy calculation would allow us to account for discrepancies
between predicted and actual outcomes more effectively. For instance, if the model predicts a team
will win but they lose, the evaluation should penalize the model more heavily if the loss was by a
large margin than if it was by a close score.  

Additionally, we’ll compare results from different models and parameter combinations to identify
the features that play a significant role in predicting game outcomes.
In the NBA, performance can be influenced by recent results due to factors like fatigue, team
morale, and momentum from winning or losing streaks. We will assess our model’s ability to
capture these temporal patterns by applying techniques such as sliding window averages. We’ll
also analyze specific game samples from unique periods like pre-season, tournament play, and
playoff games to gauge the model’s adaptability across different contexts.  

### Expected Outcome
We expect our model to predict game results with a high degree of accuracy, although we
recognize that some outliers may arise due to factors not captured in our dataset. There is a level
of stochasticity which is one of the things that makes the NBA so captivating to watch.
Nonetheless, we anticipate that the model will accurately favor higher-ranked teams in many cases,
reflecting the tendency for such teams to win.  

Our goal is to identify a set of key features that consistently contribute to accurate predictions. We
expect that more complex models, which incorporate both static and dynamic information, will
outperform simpler models with a limited feature set.  

While we aim to include a wide range of features, we may ultimately select only a few high-impact
features. Ideally, we would include data on individual player performance, but due to the data’s
complexity and potential inconsistencies, we may need to rely on standard features that are more
reliably available.  

## 3. Individual Tasks
After some discussion, we have devised the following list of tasks to be completed: data
aggregation and preprocessing, exploratory data analysis, model development, hyperparameter
tuning and cross-validation, testing and evaluation, and documenting results and interpreting
findings. We have decided to divide the tasks among team members, with group collaboration as
needed. Dhruv will work on data aggregation and preprocessing, while Noah will take on
exploratory data analysis, and Ali will focus on model development and hyperparameter tuning.  

From there, we all plan to work together on testing, evaluation, and contributing to the results and
interpretation of findings. Each individual's task is an essential part of the planned pipeline for this
project, making each group member’s contribution both important and necessary. These tasks are
tentative, and we anticipate some level of shared contribution within each task.  

## References
[1] Bunker, Rory P., and Fadi Thabtah. “A Machine Learning Framework for Sport Result
Prediction.” Applied Computing and Informatics, vol. 15, no. 1, Jan. 2019, pp. 27–33,
www.sciencedirect.com/science/article/pii/S2210832717301485,
https://doi.org/10.1016/j.aci.2017.09.005.  
[2] Delen, Dursun, et al. “A Comparative Analysis of Data Mining Methods in Predicting
NCAA Bowl Outcomes.” International Journal of Forecasting, vol. 28, no. 2, Apr. 2012,
pp. 543–552, https://doi.org/10.1016/j.ijforecast.2011.05.002.  
[3] Mohammad, Rami M., et al. “Predicting Phishing Websites Based on Self-Structuring
Neural Network.” Neural Computing and Applications, vol. 25, no. 2, 21 Nov. 2013, pp.
443–458, https://doi.org/10.1007/s00521-013-1490-z.  
[4] “NBA Database.” Www.kaggle.com, www.kaggle.com/datasets/wyattowalsh/basketball.  
[5] Reza Khanmohammadi, et al. “MambaNet: A Hybrid Neural Network for Predicting the
NBA Playoffs.” SN Computer Science, vol. 5, no. 5, 10 June 2024,
https://doi.org/10.1007/s42979-024-02977-0. Accessed 17 Sept. 2024.  
[6] Thabtah, Fadi, et al. “NBA Game Result Prediction Using Feature Analysis and Machine
Learning.” Annals of Data Science, vol. 6, no. 1, 3 Jan. 2019, pp. 103–116,
https://doi.org/10.1007/s40745-018-00189-x.  
[7] Wang, Jingru, and Qishi Fan. “Application of Machine Learning on NBA Data Sets.”
Journal of Physics: Conference Series, vol. 1802, no. 3, 1 Mar. 2021, p. 032036,
https://doi.org/10.1088/1742-6596/1802/3/032036.  
