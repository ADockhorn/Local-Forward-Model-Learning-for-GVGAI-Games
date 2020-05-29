from abc import ABC, abstractmethod


class AbstractForwardModelAgent(ABC):

    def __init__(self, forward_model, score_model):
        super().__init__()
        self._forward_model = forward_model
        self._score_model = score_model

    def re_initialize(self, initial_game_state, actions):
        raise NotImplementedError("re_initialize is not implemented yet")

    def set_forward_model(self, forward_model):
        self._forward_model = forward_model

    def set_score_model(self, score_model):
        self._score_model = score_model

    @abstractmethod
    def get_next_action(self, state, actions, prev_score):
        raise NotImplemented("get_next_action is not implemented yet")

    @abstractmethod
    def get_agent_name(self) -> str:
        raise NotImplemented("get_agent_name is not implemented yet")
