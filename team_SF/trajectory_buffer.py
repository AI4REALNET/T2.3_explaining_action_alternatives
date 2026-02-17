from collections import namedtuple
import numpy as np
from copy import deepcopy
        
class MCReplayBuffer:
    TrajectoryData = namedtuple('TrajectoryData', ['state', 'belief', 'action', 'explanation', 'user_action', 'reward_return', 'feature_return'])

    def __init__(self, discounting = 0.99):
        self.buffer = []

        self.discounting = discounting

    def reset(self):
        self.buffer = []
        self.traj_lengths = []

    def append(self, s_vec, b_vec, a_vec, e_vec, ua_vec, r_vec, fsa_vec):
        self._append(s_vec.copy(), b_vec.copy(), a_vec.copy(), e_vec.copy(), ua_vec.copy(), r_vec.copy(), fsa_vec.copy())

    def _append(self, s_vec, b_vec, a_vec, e_vec, ua_vec, r_vec, fsa_vec):
        """
        done means that the last state in s_vec is a terminal state.
        """
        R = 0
        feature_return = np.zeros(fsa_vec.shape[1:])
        for i in range(a_vec.shape[0]-1, -1, -1):
            R *= self.discounting
            feature_return *= self.discounting

            R += r_vec[i]
            feature_return += fsa_vec[i,...]
            self.buffer.append(deepcopy(MCReplayBuffer.TrajectoryData(s_vec[i,...], b_vec[i,...], a_vec[i,...], e_vec[i,...], ua_vec[i,...], R, feature_return)))
    
    def size(self):
        return len(self.buffer)
    
    def sample_batch_MC(self, batch_size):
        s_batch = []
        b_batch = []
        a_batch = []
        expl_batch = []
        ua_batch = []
        r_return_batch = []
        fsa_return_batch = []

        for idx in np.random.randint(0, self.size(), (batch_size,)):
            s, b, a, e, ua, r_ret, fsa_ret = self.buffer[idx]
            s_batch.append(s)
            b_batch.append(b)
            a_batch.append(a)
            expl_batch.append(e)
            ua_batch.append(ua)
            r_return_batch.append(r_ret)
            fsa_return_batch.append(fsa_ret)

        return np.array(s_batch, dtype=np.float32), np.array(b_batch, dtype=np.float32), np.array(a_batch, dtype=np.float32), np.array(expl_batch, dtype=np.float32), np.array(ua_batch, dtype=np.float32), np.array(r_return_batch, dtype=np.float32), np.array(fsa_return_batch, dtype=np.float32)