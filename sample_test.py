import ray
import torch
from time import time
import os
from env import Env

from model import Model
from config import config
from runner_for_test import TestRunner,RayTestRunner

ray.init()
cfg = config()
#cfg.model_path = 'model_am'

test_size = 500
sample_size = 64


def main(cfg):
    device = cfg.device
    global_model = Model(cfg)
    global_model.to(device)

    meta_agent_list = [RayTestRunner.remote(metaAgentID=i, cfg=cfg) for i in range(cfg.meta_agent_amount)]

    checkpoint = torch.load(cfg.model_path + '/model_states.pth')
    global_step = checkpoint['step']
    global_model.load_state_dict(checkpoint['model'])
    print("load model at", global_step)

    # get global network weights
    global_weights = global_model.state_dict()
    # update local network
    update_local_network_job_list = []
    for i, meta_agent in enumerate(meta_agent_list):
        update_local_network_job_list.append(meta_agent.set_weights.remote(global_weights))

    average_max_length = 0
    average_time = 0
    sum_time = 0

    with torch.no_grad():
        for i in range(test_size):
            print(i)
            env = Env(cfg, i)
            env_id = ray.put(env)  # initialize a new env for meta agents
            min_max_length = 100
            t1 = time()
            for j in range(sample_size // cfg.meta_agent_amount):
                sample_job_list = []
                for _, meta_agent in enumerate(meta_agent_list):
                    sample_job_list.append(meta_agent.sample.remote(env_id))
                sample_done_id, _ = ray.wait(sample_job_list, num_returns=cfg.meta_agent_amount)
                returns = ray.get(sample_done_id)

                for result in returns:
                    max_length = result.item()
                    if min_max_length > max_length:
                        min_max_length = max_length
            t2 = time()
            t = t2 - t1

            average_time = (t + average_time * i) / (i +1)
            sum_time += t
            average_max_length = (min_max_length + average_max_length * i) / (i + 1)
            print(average_max_length)
            print('average_time', average_time)
    print('average_max_length', average_max_length)
    print('average_time', average_time)
    print('sum_time', sum_time)


if __name__ == '__main__':
    main(cfg)
