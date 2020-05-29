import numpy as np
import pickle
import os

from tqdm import tqdm
from replay.replay import Replay


class SparseReplay(Replay):

    def __init__(self, game, level, initial_frame):
        super(SparseReplay, self).__init__(game, level, initial_frame)
        self._initial_frame = initial_frame.copy()
        self._previous_frame = initial_frame.copy()
        self._changed_positions = []
        self._changed_values = []

    def add_frame(self, action, resulting_frame, reward=0, result=0):
        self.length += 1
        self._actions.append(action)
        self._rewards.append(reward)
        self._result = result

        changes = self._previous_frame != resulting_frame
        changed_positions = np.where(changes)
        changed_values = resulting_frame[changed_positions]

        self._previous_frame[changed_positions] = changed_values

        if self._tiles is not None:
            if np.sum(self._previous_frame == resulting_frame) != self._width*self._height*self._tiles:
                raise Exception("errors in the reconstructions")
        else:
            if np.sum(self._previous_frame == resulting_frame) != self._width*self._height:
                raise Exception("errors in the reconstructions")

        self._previous_frame = resulting_frame
        self._changed_positions.append(changed_positions)
        self._changed_values.append(changed_values)

    def get_transition(self, tick):
        raise NotImplementedError

    def get_transitions(self):
        frames = [self._initial_frame]
        for changed_positions, changed_values in zip(self._changed_positions, self._changed_values):
            next_frame = frames[-1].copy()
            next_frame[changed_positions] = changed_values
            frames.append(next_frame)
        return frames

    def get_frames(self):
        next_frame = self._initial_frame
        yield next_frame
        for changed_positions, changed_values in zip(self._changed_positions, self._changed_values):
            next_frame = next_frame.copy()
            next_frame[changed_positions] = changed_values
            yield next_frame

    def create_animation(self, tileset, filename):
        import matplotlib.pyplot as plt
        from matplotlib import animation

        fig, axis = plt.subplots(1, 1)
        plt.axis("off")
        ims = []

        total_reward = 0
        for tick, frame in enumerate(tqdm(self.get_frames(), desc="create video", ncols=100)):
            ttl = axis.text(0.5, 1.01, f"{self._game} | total score = {total_reward} | tick = {tick}",
                            horizontalalignment='center',
                            verticalalignment='bottom', transform=axis.transAxes)
            ims.append([plt.imshow(tileset.map_lfm_state_to_frame(frame)), ttl])
            if tick > 0:
                total_reward += self._rewards[tick-1]

        anim = animation.ArtistAnimation(fig, ims, interval=100, blit=False, repeat=False)
        anim.save(filename)
        plt.close()

    @classmethod
    def load_from_file(cls, file):
        with open(file, "rb") as f:
            game = pickle.load(f)
            level = pickle.load(f)
            result = pickle.load(f)

            length = pickle.load(f)
            initial_frame = pickle.load(f)
            changed_values = pickle.load(f)
            changed_positions = pickle.load(f)
            actions = pickle.load(f)
            rewards = pickle.load(f)

        obj = cls(game, level, initial_frame)
        obj.length = length
        obj._changed_values = changed_values
        obj._changed_positions = changed_positions
        obj._actions = actions
        obj._rewards = rewards
        obj._result = result
        return obj

    def write_to_file(self, file):
        if not os.path.exists(os.path.dirname(file)):
            os.makedirs(os.path.dirname(file))

        with open(file, "wb") as f:
            pickle.dump(self._game, f)
            pickle.dump(self._level, f)
            pickle.dump(self._result, f)

            pickle.dump(self.length, f)
            pickle.dump(self._initial_frame, f)
            pickle.dump(self._changed_values, f)
            pickle.dump(self._changed_positions, f)
            pickle.dump(self._actions, f)
            pickle.dump(self._rewards, f)
