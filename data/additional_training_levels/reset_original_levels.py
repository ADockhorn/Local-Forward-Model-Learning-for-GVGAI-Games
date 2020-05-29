from data.additional_training_levels.create_additional_training_levels import load_file, write_file
import os


def reset_levels_for_game(game, sourcefolder):
    levels = ["lvl0", "lvl1", "lvl2", "lvl3", "lvl4"]
    game_name = game.split("-")[1]
    for level in levels:
        source_file = f"{sourcefolder}{level}_backup.txt"
        target_file = f"D:\\Git Folders\\General Game AI\\GVGAI_GYM_Optimized\\games\\{game_name}_v0\\{game_name}_{level}.txt"
        #target_file = f"D:\\GVGAI_GYM\\gym_gvgai\\envs\\games\\{game_name}_v0\\{game_name}_{level}.txt"
        if os.path.exists(source_file):
            print("reset ", level, " of ", game)
            a = load_file(source_file)
            write_file(target_file, a)


if __name__ == "__main__":
    #from Constants import GAMES
    #games = GAMES
    for game in ["bait"]:
        reset_levels_for_game("gvgai-"+game, f"gvgai-{game}\\")


