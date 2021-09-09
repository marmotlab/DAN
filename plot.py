import torch
import matplotlib.pyplot as plt
import os
import imageio
import numpy as np
from runner_for_test import TestRunner
from config import config
from env import Env

cfg = config()

model_path = cfg.model_path
device = cfg.device
decode_type = cfg.strategy

def show(target, routes, max_length):
    target = target.squeeze(0).cpu()
    print('max_length:{:.3f}'.format(max_length.item()))
    print(target.size())
    plt.figure()
    plt.plot(target[:, 0], target[:, 1], 'ro', markersize=4)
    for route in routes:
        print(route)
        plt.plot(target[:, 0], target[:, 1], 'ro', markersize=4)
        depot = torch.tensor([0]).cuda()
        route = torch.cat([route, depot])
        np_tour = route[:].cpu().detach()
        plt.plot(target[np_tour, 0], target[np_tour, 1], linewidth=1)
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.savefig(cfg.gifs_path + '/{:d}_seed_{:d}_length_{:.2f}_{}.jpg'.format(cfg.target_size, cfg.seed, max_length,
                                                                              decode_type))


if __name__ == '__main__':
    env = Env(cfg, cfg.seed)
    runner = TestRunner(metaAgentID=0, cfg=cfg, decode_type=decode_type)

    checkpoint = torch.load(model_path + '/model_states.pth')
    runner.model.load_state_dict(checkpoint['model'])

    with torch.no_grad():
        max_length, route_list = runner.sample(env)

    min_max_length=max_length
    min_route_list=route_list
    if decode_type=='sampling':
        for i in range(128):
            env.global_mask=env.generate_mask()
            with torch.no_grad():
                max_length, route_list= runner.sample(env)
            print(max_length)
            print(min_max_length>max_length)
            if min_max_length>max_length:
                min_max_length=max_length
                min_route_list=route_list
    target_inputs = env.target_inputs
    target_set = torch.cat((target_inputs[0], target_inputs[1]), dim=1)

    show(target_set, min_route_list, min_max_length)
