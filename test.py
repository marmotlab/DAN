import torch
import os
import ray
from time import time
from runner_for_test import TestRunner
from config import config
from env import Env

cfg = config()
model_path = cfg.model_path
device = cfg.device
decode_type = 'greedy'
test_size = 500


def test():
    average_max_length = 0
    average_mean_length = 0
    average_time = 0
    sum_time = 0

    runner = TestRunner(metaAgentID=0, cfg=cfg, decode_type=decode_type)

    checkpoint = torch.load(model_path + '/model_states.pth')
    runner.model.load_state_dict(checkpoint['model'])
    for i in range(test_size):
        print(i)
        env = Env(cfg, seed=i)
        t1 = time()
        with torch.no_grad():
            max_length = runner.sample(env)
        t2 = time()

        max_length = max_length.item()
        # mean_length = mean_length.item()
        t = t2 - t1

        average_max_length = (max_length + average_max_length * i) / (i + 1)
        #average_mean_length = (mean_length + average_mean_length * i) / (i + 1)
        average_time = (t + average_time * i) / (i + 1)
        sum_time += t
        print('average_max_length', average_max_length)
        print('average_time', average_time)
    print('average_max_length', average_max_length)
    #print('average_mean_length', average_mean_length)
    print('average_time', average_time)
    print('sum_time', sum_time)

if __name__ == '__main__':
    test()
