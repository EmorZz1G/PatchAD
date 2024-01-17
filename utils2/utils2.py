import torch
import time
import numpy as np
import os
import torch as t
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import random
import torch.nn as nn


def softmax(f):
    # instead: first shift the values of f so that the highest number is 0:
    f -= np.max(f)  # f becomes [-666, -333, 0]
    return np.exp(f) / np.sum(np.exp(f))  # safe to do, gives the correct answer


def mk_dir(pth):
    import os
    if not os.path.exists(pth):
        os.mkdir(pth)
        print(f"{pth} create")
    else:
        print(f'{pth} exist')
        pass


def dist(a, b):
    # n, L, d
    # n l 1,d
    # n l d,1
    a = a.squeeze(-1)
    b = a.squeeze(-2)
    dis = a - b
    # n l d,d
    dis = dis ** 2
    return dis


class MyTri(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(MyTri, self).__init__()
        self.margin = margin

    def forward(self, x, pos, neg):
        p = dist(x, pos)
        p = torch.mean(p)
        n = dist(x, neg)
        n = torch.mean(n)
        return p - n + self.margin


def seed_all(seed=2022):
    random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


class TimeRecorder:
    def __init__(self, record_name, save_pth, logger=None):
        self.name = record_name
        self.pth = save_pth
        self.times_tab = {}
        self.last_event = None
        self.logger = logger
        self._context = "0_"

    @property
    def context(self):
        return self._context

    @context.setter
    def context(self, value):
        self._context = value + "_"

    def start(self, event):
        event = self.context + event
        self.times_tab[event] = time.time()
        self.last_event = event

    def stop(self, event=None):
        if event is None:
            event = self.last_event
        else:
            event = self.context + event
        if event is None:
            print('no event exists')
            return
        elif event not in self.times_tab:
            print('no event started')
            return

        self.times_tab[event] = time.time() - self.times_tab[event]

    def save(self):
        pth = self.pth
        f = open(os.path.join(pth, 'rec_' + self.name + '.txt'), 'w')
        for k, tm in self.times_tab.items():
            s = k + '#' + str(tm) + 's'
            if self.logger: self.logger.info(s)
            f.write(s + '\n')
        f.close()
