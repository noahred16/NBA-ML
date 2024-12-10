WITH team_recent_games AS (
	SELECT
		-- Label Data
		G.game_id, 							-- ids are just for labeling
		G.team_id_home, G.team_id_away, 	-- ids are just for labeling
		DATE(G.game_date) AS 'game_date', 	-- the date can be considered relative and not a feature
		G.team_name_home, G.team_name_away, -- if we wanted to consider the name as category we could do 30 booleans. Not sure it would add value.  
		G.season_type, -- converted to 3 booleans
		-- Categorical Data
		CASE WHEN G.season_type = 'Regular Season' THEN 1 ELSE 0 END AS 'is_regular_season',
		CASE WHEN G.season_type = 'Playoffs' THEN 1 ELSE 0 END AS 'is_playoffs',
		CASE WHEN G.season_type = 'Pre Season' THEN 1 ELSE 0 END AS 'is_pre_season',
		-- Target Data
		G.pts_home, -- target 1
		G.pts_away, -- target 2
		G.wl_home, -- target 3
	-- HistoryStats (Sequential Data)
		-- Sequential Label Data
		M.game_id AS 'team_recent_game_id',
		DATE(M.game_date) AS 'team_recent_game_date', -- label data as we can use days ago as a more valuable feature
		M.season_type AS 'team_recent_game_season_type',
		CASE WHEN M.team_id_home = G.team_id_home THEN M.team_id_home ELSE M.team_id_away END AS 'team_id',
		CASE WHEN M.team_id_home = G.team_id_home THEN M.team_id_away ELSE M.team_id_home END AS 'opponent_team_id',
		CASE WHEN M.team_id_home = G.team_id_home THEN M.team_name_home ELSE M.team_name_away END AS 'team_name',
		CASE WHEN M.team_id_home = G.team_id_home THEN M.team_name_away ELSE M.team_name_home END AS 'opponent_team_name',
		-- Sequential Categorical/Meta Data
		JULIANDAY(G.game_date) - JULIANDAY(M.game_date) AS 'days_ago', -- number of days old this data is
		ROW_NUMBER() OVER (PARTITION BY G.game_id ORDER BY M.game_date DESC) AS 'games_ago', -- how many games ago they played
		CASE WHEN M.team_id_home = G.team_id_home THEN 1 ELSE 0 END AS 'is_home_game_team_recent',
		CASE WHEN M.season_type = 'Regular Season' THEN 1 ELSE 0 END AS 'is_regular_season_team_recent',
		CASE WHEN M.season_type = 'Playoffs' THEN 1 ELSE 0 END AS 'is_playoffs_team_recent',
		CASE WHEN M.season_type = 'Pre Season' THEN 1 ELSE 0 END AS 'is_pre_season_team_recent',
		-- Sequential Target Data
		CASE WHEN M.team_id_home = G.team_id_home THEN M.pts_home ELSE M.pts_away END AS 'pts',
		CASE WHEN M.team_id_home = G.team_id_home THEN M.pts_away ELSE M.pts_home END AS 'opponent_pts',
		CASE WHEN M.team_id_home = G.team_id_home THEN M.wl_home ELSE M.wl_away END AS 'wl',
		-- features for
		-- home_team_features = ['pts_home', 'fg_pct_home', 'fg3_pct_home', 'fg3m_home', 'ft_pct_home', 'ftm_home', 'reb_home', 'ast_home', 'stl_home', 'blk_home', 'tov_home']
		CASE WHEN M.team_id_home = G.team_id_home THEN M.pts_home ELSE M.pts_away END AS 'pts_for',
		CASE WHEN M.team_id_home = G.team_id_home THEN M.fg_pct_home ELSE M.fg_pct_away END AS 'fg_pct_for',
		CASE WHEN M.team_id_home = G.team_id_home THEN M.fg3_pct_home ELSE M.fg3_pct_away END AS 'fg3_pct_for',
		CASE WHEN M.team_id_home = G.team_id_home THEN M.fg3m_home ELSE M.fg3m_away END AS 'fg3m_for',
		CASE WHEN M.team_id_home = G.team_id_home THEN M.ft_pct_home ELSE M.ft_pct_away END AS 'ft_pct_for',
		CASE WHEN M.team_id_home = G.team_id_home THEN M.ftm_home ELSE M.ftm_away END AS 'ftm_for',
		CASE WHEN M.team_id_home = G.team_id_home THEN M.reb_home ELSE M.reb_away END AS 'reb_for',
		CASE WHEN M.team_id_home = G.team_id_home THEN M.ast_home ELSE M.ast_away END AS 'ast_for',
		CASE WHEN M.team_id_home = G.team_id_home THEN M.stl_home ELSE M.stl_away END AS 'stl_for',
		CASE WHEN M.team_id_home = G.team_id_home THEN M.blk_home ELSE M.blk_away END AS 'blk_for',
		CASE WHEN M.team_id_home = G.team_id_home THEN M.tov_home ELSE M.tov_away END AS 'tov_for',
		-- features against
		-- away_team_features = ['pts_away', 'fg_pct_away', 'fg3_pct_away', 'fg3m_away', 'ft_pct_away', 'ftm_away', 'reb_away', 'ast_away', 'stl_away', 'blk_away', 'tov_away'] 
		CASE WHEN M.team_id_home = G.team_id_home THEN M.pts_away ELSE M.pts_home END AS 'pts_against',
		CASE WHEN M.team_id_home = G.team_id_home THEN M.fg_pct_away ELSE M.fg_pct_home END AS 'fg_pct_against',
		CASE WHEN M.team_id_home = G.team_id_home THEN M.fg3_pct_away ELSE M.fg3_pct_home END AS 'fg3_pct_against',
		CASE WHEN M.team_id_home = G.team_id_home THEN M.fg3m_away ELSE M.fg3m_home END AS 'fg3m_against',
		CASE WHEN M.team_id_home = G.team_id_home THEN M.ft_pct_away ELSE M.ft_pct_home END AS 'ft_pct_against',
		CASE WHEN M.team_id_home = G.team_id_home THEN M.ftm_away ELSE M.ftm_home END AS 'ftm_against',
		CASE WHEN M.team_id_home = G.team_id_home THEN M.reb_away ELSE M.reb_home END AS 'reb_against',
		CASE WHEN M.team_id_home = G.team_id_home THEN M.ast_away ELSE M.ast_home END AS 'ast_against',
		CASE WHEN M.team_id_home = G.team_id_home THEN M.stl_away ELSE M.stl_home END AS 'stl_against',
		CASE WHEN M.team_id_home = G.team_id_home THEN M.blk_away ELSE M.blk_home END AS 'blk_against',
		CASE WHEN M.team_id_home = G.team_id_home THEN M.tov_away ELSE M.tov_home END AS 'tov_against'
	FROM
		game G
	JOIN game M ON -- grab recent game history (M) for each game (G) search for prev games for home team
		M.game_date > DATE(G.game_date, '-365 days') -- and within 365 days. this speeds up query and is a safe assumption
		AND M.game_date < G.game_date -- only grab games that happened before the game in question
		AND (M.team_id_home = G.team_id_home OR M.team_id_away = G.team_id_home) -- find games where the home team played
		AND M.season_type IN ('Regular Season', 'Playoffs', 'Pre Season')
	WHERE CAST(strftime('%Y', G.game_date) AS INTEGER) > 2000 -- Only care about games after 2000
		AND G.season_type IN ('Regular Season', 'Playoffs', 'Pre Season') -- exlude All-Star games
		AND G.game_id IN (SELECT game_id FROM game LIMIT (10000)) -- limit for faster query, assume each teams plays once a year
--		AND G.game_id = '0020100001'
)
SELECT *
FROM team_recent_games 
WHERE games_ago <= 10 -- Limit for sequential data
--	 AND game_id IN (SELECT DISTINCT game_id FROM team_recent_games WHERE games_ago = 10) -- filter out games with less than 10 games of history, should be none filtered out. 
--ORDER BY game_date, game_id, team_recent_game_date DESC

