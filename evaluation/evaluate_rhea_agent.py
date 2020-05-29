from Constants import NON_DETERMINISTIC_TEST_GAMES, LEARNING_TRACK_GAMES, THRESHOLDS
from agents.models.probabilistic_local_forward_model import ProbabilisticLearningLocalForwardModelData
from agents.random_agent import RandomAgent
from evaluation.test_agent import evaluate_agent, evaluate_lfm_agent
import os
from agents.models.local_forward_model_data import LocalForwardModelData
from agents.models.score_model_data import ScoreModelData
from agents.models.local_forward_model import LocalForwardModel
from agents.models.score_model import ScoreModel
from agents.models.tile_map import TileSet
from sklearn.tree import DecisionTreeClassifier
from data.additional_training_levels.reset_original_levels import reset_levels_for_game
from agents.rolling_horizon_ea import SimpleRHEA


if __name__ == "__main__":

    for game in NON_DETERMINISTIC_TEST_GAMES:
        if game != "golddigger":
            continue

        if not os.path.exists(f"..\\data\\paper_results\\rhea_agent_symmetry_trained\\{game}"):
            os.makedirs(f"..\\data\\paper_results\\rhea_agent_symmetry_trained\\{game}")
        else:
            continue

        if game in LEARNING_TRACK_GAMES:
            evaluation_levels = ['lvl0-v0', 'lvl1-v0']
        else:
            evaluation_levels = ['lvl0-v0', 'lvl1-v0', 'lvl2-v0', 'lvl3-v0', 'lvl4-v0']

        reset_levels_for_game("gvgai-" + game, f"..\\data\\additional_training_levels\\gvgai-{game}\\")

        success = False

        # load symmetric data
        modelfolder = f"..\\data\\paper_training\\{game}\\symmetric_active_learning_optimized\\"
        tile_set = TileSet.load_from_file(modelfolder + "tile_set.bin")
        tile_set.threshold = THRESHOLDS[game]
        lfm_data = ProbabilisticLearningLocalForwardModelData.load_from_file(modelfolder + "lfm_data.bin")
        sm_data = ScoreModelData.load_from_file(modelfolder + "sm_data.bin", tile_set)

        # train Models
        lfm = LocalForwardModel(DecisionTreeClassifier(), lfm_data, True)
        lfm.train()
        sm = ScoreModel(DecisionTreeClassifier(), sm_data, True)
        sm.train()

        agent = SimpleRHEA(lfm, sm, 5, 0.7, 20, use_shift_buffer=True, flip_at_least_one=True, discount=0.99)

        results = evaluate_lfm_agent(agent, game, evaluation_levels, tile_set=tile_set, repetitions=20,
                                     result_folder=f"..\\data\\paper_results\\rhea_agent_symmetry_trained\\{game}")
