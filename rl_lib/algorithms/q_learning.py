import numpy as np


class QLearning:

    def __init__(self, shape, learning_rate, discount_factor):
        assert 0 < learning_rate < 1, "Learning rate must be between 0 and 1."
        assert 0 < discount_factor < 1, "Discount factor must be between 0 and 1."

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = np.zeros(shape=shape)

    def update(self, s_t, a_t, reward, s_next):
        self.q_table[s_t][a_t] += self.learning_rate * self._get_temporal_difference(s_t, a_t, reward, s_next)

    def _get_temporal_difference(self, s_t, a_t, reward, s_next):
        return self._get_temporal_difference_target(reward, s_next) - self.q_table[s_t][a_t]

    def _get_temporal_difference_target(self, reward, s_next):
        return reward + self.discount_factor * np.max(self.q_table[s_next])

    def get_best_action(self, s_t):
        return np.argmax(self.q_table[s_t])
