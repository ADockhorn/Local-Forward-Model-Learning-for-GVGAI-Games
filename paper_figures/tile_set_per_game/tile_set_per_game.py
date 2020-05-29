from agents.models.tile_map import TileSet

if __name__ == "__main__":
    waterpuzzle = TileSet.load_from_file("tile_set_waterpuzzle.bin")
    waterpuzzle.plot_tile_dict()

    golddigger = TileSet.load_from_file("tile_set_golddigger.bin")
    golddigger.plot_tile_dict()

    treasurekeeper = TileSet.load_from_file("tile_set_treasurekeeper.bin")
    treasurekeeper.plot_tile_dict()
