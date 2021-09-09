import torch
import numpy as np
from config import config
from agentEncoder import AgentEncoder
from targetEncoder import TargetEncoder
from decoder import Decoder


class Env():
    def __init__(self, cfg, seed=None):
        self.agent_amount = cfg.agent_amount
        self.target_size = cfg.target_size
        self.device = cfg.device
        self.seed = seed
        self.target_inputs = self.generate_target_inputs()
        self.global_mask = self.generate_mask()

    def generate_target_inputs(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        depot_position = np.random.rand(1, 1, 2)
        target_position = np.random.rand(1, self.target_size - 1, 2)
        target_inputs = [torch.FloatTensor(depot_position).cuda(), torch.FloatTensor(target_position).cuda()]
        return target_inputs

    def generate_mask(self):
        mask = torch.zeros((1, self.target_size), device=self.device, dtype=torch.int64)
        return mask

    def update_mask(self, target_index):
        self.global_mask = self.global_mask.scatter_(dim=1, index=target_index.unsqueeze(1), value=1)

    def get_agent_inputs(self, workersList,agent_id):
        agent_position = []
        agent_next_action_gap = []
        agent_partial_route = []
        for i in range(self.agent_amount):
            agent_position.append(workersList[i].agent_position)
            agent_next_action_gap.append(workersList[i].next_select_gap)
            agent_partial_route.append(workersList[i].point_list)
        agent_position = torch.stack(agent_position, dim=2).squeeze(0)
        agent_next_action_gap = torch.Tensor(agent_next_action_gap).unsqueeze(0).unsqueeze(-1).cuda()
        agent_inputs = torch.cat((agent_position, agent_next_action_gap), -1)
        return agent_inputs

