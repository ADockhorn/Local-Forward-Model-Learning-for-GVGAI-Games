import numpy as np

from agents.models.score_model_data import ScoreModelData


class ScoreModel:
    """
    Predicts the score change of a state-transition based on previously observed state-transitions in score_model_data.
    """
    def __init__(self, score_model, sm_data: ScoreModelData, remember_predictions=True):
        self.score_model = score_model
        self.sm_data = sm_data
        self.remember_predictions = remember_predictions
        if self.remember_predictions:
            self.known_predictions = dict()
            self.sm_dict = None
        pass

    def train(self):
        """
            Trains the score_model classifier or regressor
        """
        data, reward = self.sm_data.get_data_set()
        self.score_model.fit(data, reward)
        if self.sm_dict is None:
            self.sm_dict = dict()
            for pattern, score in zip(data, reward):
                self.sm_dict[tuple(pattern)] = score
        return

    def predict(self, prev_lfm_state, lfm_state):
        """
        Predict the value of a given state transition. If self.remember_predictions is True predictions for each
        pattern will be remembered and applied during upcoming prediction tasks. This results in a speed-up in case
        of otherwise complex models.

        :param prev_lfm_state: state before the action has been applied
        :param lfm_state: state after the action has been applied
        :return: predicted reward
        """
        if self.remember_predictions:
            pattern = self.sm_data.get_pattern(prev_lfm_state, lfm_state)
            el = tuple(pattern.tolist())
            if el in self.sm_dict:
                return self.sm_dict[el]
            if el in self.known_predictions:
                return self.known_predictions[el]
            else:
                pred = self.score_model.predict(pattern[:self.score_model.n_features_].reshape(1, -1))
                self.known_predictions[el] = pred[0]
                return pred[0]
        else:
            pattern = self.sm_data.get_pattern(prev_lfm_state, lfm_state)
            pattern = pattern[:self.score_model.n_features_].reshape(1, -1)
            return self.score_model.predict(pattern)[0]

    def predict_and_get_pattern(self, prev_lfm_state, lfm_state):
        """
        Predict the value of a given state transition and also return the observed pattern. If self.remember_predictions
        is True predictions for each pattern will be remembered and applied during upcoming prediction tasks. This
        results in a speed-up in case of otherwise complex models.

        :param prev_lfm_state: state before the action has been applied
        :param lfm_state: state after the action has been applied
        :return: predicted reward, score model pattern
        """
        if self.remember_predictions:
            pattern = self.sm_data.get_pattern(prev_lfm_state, lfm_state)
            el = tuple(pattern.tolist())
            if el in self.sm_dict:
                return self.sm_dict[el], pattern
            if el in self.known_predictions:
                return self.known_predictions[el], pattern
            else:
                pred = self.score_model.predict(pattern[:self.score_model.n_features_].reshape(1, -1))
                self.known_predictions[el] = pred[0]
                return pred[0], pattern
        else:
            pattern = self.sm_data.get_pattern(prev_lfm_state, lfm_state)
            pattern = pattern[:self.score_model.n_features_].reshape(1, -1)
            return self.score_model.predict(pattern)[0], pattern

    def predict_pattern(self, pattern):
        if pattern is np.ndarray:
            el = tuple(pattern.tolist())
        else:
            el = tuple(pattern)
        if el in self.sm_dict:
            return self.sm_dict[el]
        if el in self.known_predictions:
            return self.known_predictions[el]
        else:
            pred = self.score_model.predict(pattern[:self.score_model.n_features_].reshape(1, -1))
            self.known_predictions[el] = pred[0]
            return pred[0]

    def correct_mistake(self, prev_lfm_state, lfm_state, score):
        """
        Adds a wrongly predicted pattern to the known_prediction dictionary to avoid the same mistake in the future.
        :param prev_lfm_state: state before the action has been applied
        :param lfm_state: state after the action has been applied
        :param score: observed score
        """
        pattern = tuple(self.sm_data.add_observation(prev_lfm_state, lfm_state, score))
        self.known_predictions.pop(pattern, None)
        self.sm_dict[pattern] = score
        self.train()
        return pattern

    def add_pattern(self, prev_lfm_state, lfm_state, score):
        """
        adds an observed score and returns the score model pattern
        :param prev_lfm_state: state before the action has been applied
        :param lfm_state: state after the action has been applied
        :param score: observed score
        :return: score model pattern
        """
        pattern = tuple(self.sm_data.add_observation(prev_lfm_state, lfm_state, score))
        self.sm_dict[pattern] = score
        return pattern
