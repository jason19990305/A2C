import numpy as np 
import torch

class ReplayBuffer:
    def __init__(self, args):
        self.clear_batch()
        self.episode_count = 0
        self.episode_batch = []
        
    def clear_batch(self):
        self.s = []
        self.a = []
        self.r = []
        self.s_ = []
        self.done = []
        self.count = 0
        
    def store(self, s, a , r, s_, done):
        self.s.append(s)
        self.a.append(a)
        self.r.append(r)
        self.s_.append(s_)
        self.done.append(done)
        
    def to_episode_batch(self):
        s = torch.tensor(np.array(self.s), dtype=torch.float)
        a = torch.tensor(np.array(self.a), dtype=torch.int64)
        r = torch.tensor(np.array(self.r), dtype=torch.float)
        s_ = torch.tensor(np.array(self.s_), dtype=torch.float)
        done = torch.tensor(np.array(self.done), dtype=torch.float)
        self.episode_batch.append((s, a, r, s_, done))
        self.episode_count += 1
        self.clear_batch()
        
    def clear_episode_batch(self):
        self.episode_batch = []
        self.episode_count = 0
