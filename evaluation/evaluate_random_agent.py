from Constants import PAPER_GAMES, GAMES
from agents.random_agent import RandomAgent
from evaluation.test_agent import evaluate_agent
import os

if __name__ == "__main__":

    evaluation_levels = ['lvl0-v0', 'lvl1-v0', 'lvl2-v0', 'lvl3-v0', 'lvl4-v0']

    for game in GAMES:
        if not os.path.exists(f"..\\data\\paper_results\\random_agent\\{game}"):
            os.makedirs(f"..\\data\\paper_results\\random_agent\\{game}")
        else:
            continue

        success = False

        while not success:
            try:
                agent = RandomAgent()
                results = evaluate_agent(agent, game, evaluation_levels, repetitions=20,
                                         result_folder=f"..\\data\\paper_results\\random_agent\\{game}")
                success = True
            except Exception:
                pass
