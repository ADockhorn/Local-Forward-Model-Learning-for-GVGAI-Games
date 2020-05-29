LEARNING_TRACK_GAMES = ["golddigger", "treasurekeeper", "waterpuzzle"]

DETERMINISTIC_TEST_GAMES = ["waterpuzzle", "labyrinth", "labyrinthdual", "bait"]
NON_DETERMINISTIC_TEST_GAMES = ["golddigger", "treasurekeeper", "sokoban", "boulderdash", "zelda"]
ALL_GAMES = DETERMINISTIC_TEST_GAMES + NON_DETERMINISTIC_TEST_GAMES

THRESHOLDS = {
    "waterpuzzle": 0.85,
    "labyrinth": 0.85,
    "labyrinthdual": 0.85,
    "bait": 0.86,
    "golddigger": 0.9,
    "treasurekeeper": 0.8,
    "sokoban": 0.3,
    "boulderdash": 0.8,
    "zelda": 0.8,
}

SPANS = {
    "waterpuzzle": 1,
    "labyrinth": 1,
    "labyrinthdual": 1,
    "bait": 2,
    "golddigger": 1,
    "treasurekeeper": 2,
    "sokoban": 2,
    "boulderdash": 1,
    "zelda": 2
}
