from Constants import ALL_GAMES
from agents.training.deterministic_active_state_selection import extended_training_level_generator
import matplotlib.pyplot as plt
import os


if __name__ == "__main__":
    for game in ALL_GAMES:

        level_generator = extended_training_level_generator(
            "gvgai-" + game,
            f"..\\..\\data\\additional_training_levels\\gvgai-{game}\\",
            f"D:\\Git Folders\\General Game AI\\GVGAI_GYM_Optimized\\games\\{game}_v0\\{game}_lvl2.txt",
            1, ["lvl0", "lvl1", "lvl2", "lvl3", "lvl4"], plot_levels=False)

        if not os.path.exists(f"gvgai-{game}\\"):
            os.mkdir(f"gvgai-{game}\\")

        for training_run, (env, game, level, initial_frame) in enumerate(level_generator):
            plt.imshow(initial_frame)
            plt.savefig(f"{game}\\{level[:-4]}.png")
            plt.close()
