from agents.models.local_forward_model_data import LocalForwardModelData
import numpy as np


class LocalForwardModel:
    def __init__(self, state_model, lfm_data: LocalForwardModelData, remember_predictions=True):
        self.state_model = state_model
        self.lfm_data = lfm_data
        self.remember_predictions = remember_predictions
        if self.remember_predictions:
            self.known_predictions = dict()
        pass

    def initialize(self, observation):
        self.lfm_data.initialize(observation)

    def train(self):
        data = self.lfm_data.get_data_set()
        self.state_model.fit(data[:, :-1], data[:, -1])
        self.known_predictions = dict()
        return

    def extract_unknown_patterns(self, lfm_state, action):
        prediction_mask = np.zeros(lfm_state.shape, dtype=np.bool)
        result = np.zeros(lfm_state.shape, dtype=np.int)
        data_set = []

        patterns = self.lfm_data.get_patterns(lfm_state, action)

        i = 0
        for x in range(lfm_state.shape[0]):
            for y in range(lfm_state.shape[1]):
                el = tuple(patterns[i].tolist())
                if el in self.known_predictions:
                    result[x, y] = self.known_predictions[el]
                else:
                    prediction_mask[x, y] = 1
                    data_set.append(el)
                i += 1

        return data_set, prediction_mask, result

    def predict(self, frame, action):
        if self.remember_predictions:
            data, prediction_mask, result = self.extract_unknown_patterns(frame, action)

            if len(data) > 0:
                prediction = self.state_model.predict(data)
                result[prediction_mask] = prediction
                for el, pred in zip(data, prediction):
                    self.known_predictions[el] = pred
            return result
        else:
            data = self.lfm_data.get_patterns(frame, action)
            return self.state_model.predict(data).reshape(frame.shape)

    def predict_and_get_unknown_patterns(self, frame, action):
        if self.remember_predictions:
            data, prediction_mask, result = self.extract_unknown_patterns(frame, action)

            if len(data) > 0:
                prediction = self.state_model.predict(data)
                result[prediction_mask] = prediction
                for el, pred in zip(data, prediction):
                    self.known_predictions[el] = pred
            return result, np.sum(prediction_mask)
        else:
            data = self.lfm_data.get_patterns(frame, action)
            return self.state_model.predict(data).reshape(frame.shape)


