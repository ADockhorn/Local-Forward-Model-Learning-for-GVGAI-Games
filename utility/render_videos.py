from agents.models.tile_map import TileSet
from replay.sparsereplay import SparseReplay
import os

"""
This script renders replays in the way the agent sees them, meaning it does not need to represent the true game in case
the tile_set does not perfectly match the game's tile set.
"""
if __name__ == "__main__":
    modelfolder = f"..\\data\\paper_results\\bfs_agent_fully_trained\\{game}\\"
    replayfolder = modelfolder + "replay_data\\"
    videofolder = modelfolder + "video_data\\"
    if not os.path.exists(videofolder):
        os.makedirs(videofolder)

    tileset = TileSet.load_from_file(modelfolder + "tile_set.bin")
    tileset.plot_tile_dict()

    for replayfile in os.listdir(replayfolder)[60:65]:
        replay = SparseReplay.load_from_file(replayfolder + replayfile)
        replay.create_animation(tileset, videofolder+replayfile[:-8]+".mp4")
