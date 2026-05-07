CREATE TABLE IF NOT EXISTS Arenas (
    arena_name TEXT PRIMARY KEY,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS Teams (
    team_acronym TEXT PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS Players (
    player_id TEXT PRIMARY KEY,
    player_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS Games (
    game_id TEXT PRIMARY KEY,
    game_date TEXT NOT NULL,
    home_team TEXT REFERENCES Teams(team_acronym),
    visitor_team TEXT REFERENCES Teams(team_acronym),
    arena_name TEXT REFERENCES Arenas(arena_name)
);

CREATE TABLE IF NOT EXISTS Performances (
    performance_id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT REFERENCES Games(game_id),
    player_id TEXT REFERENCES Players(player_id),
    player_team TEXT REFERENCES Teams(team_acronym),
    is_home INTEGER NOT NULL,
    
    -- Fatigue & Context Features
    miles_traveled REAL DEFAULT 0.0,
    days_rest INTEGER DEFAULT 10,
    is_back_to_back INTEGER DEFAULT 0,
    altitude_impact INTEGER DEFAULT 0,
    
    -- Basic Stats
    mp REAL NOT NULL,
    fg INTEGER DEFAULT 0,
    fga INTEGER DEFAULT 0,
    fg_pct REAL DEFAULT 0.0,
    fg3 INTEGER DEFAULT 0,
    fg3a INTEGER DEFAULT 0,
    fg3_pct REAL DEFAULT 0.0,
    ft INTEGER DEFAULT 0,
    fta INTEGER DEFAULT 0,
    ft_pct REAL DEFAULT 0.0,
    orb INTEGER DEFAULT 0,
    drb INTEGER DEFAULT 0,
    trb INTEGER DEFAULT 0,
    ast INTEGER DEFAULT 0,
    stl INTEGER DEFAULT 0,
    blk INTEGER DEFAULT 0,
    tov INTEGER DEFAULT 0,
    pf INTEGER DEFAULT 0,
    pts INTEGER DEFAULT 0,
    gmsc REAL DEFAULT 0.0,
    plus_minus INTEGER DEFAULT 0,
    
    -- Advanced Stats
    adv_ts_pct REAL DEFAULT 0.0,
    adv_efg_pct REAL DEFAULT 0.0,
    adv_3par REAL DEFAULT 0.0,
    adv_ftr REAL DEFAULT 0.0,
    adv_orb_pct REAL DEFAULT 0.0,
    adv_drb_pct REAL DEFAULT 0.0,
    adv_trb_pct REAL DEFAULT 0.0,
    adv_ast_pct REAL DEFAULT 0.0,
    adv_stl_pct REAL DEFAULT 0.0,
    adv_blk_pct REAL DEFAULT 0.0,
    adv_tov_pct REAL DEFAULT 0.0,
    adv_usg_pct REAL DEFAULT 0.0,
    adv_ortg REAL DEFAULT 0.0,
    adv_drtg REAL DEFAULT 0.0,
    adv_bpm REAL DEFAULT 0.0,

    UNIQUE(game_id, player_id)
);
