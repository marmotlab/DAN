import torch
import torch.nn as nn
from config import config
from worker import Worker
from agentEncoder import AgentEncoder
from targetEncoder import TargetEncoder
from decoder import Decoder
from env import Env


class Model(nn.Module):
    def __init__(self, cfg, decode_type='sampling', training=True):
        super(Model, self).__init__()
        self.local_agent_encoder = AgentEncoder(cfg)

        self.local_target_encoder = TargetEncoder(cfg)
        self.local_decoder = Decoder(cfg)
        self.training = training
        self.decode_type = decode_type

    def forward(self, env, agent_amount,routes=None):
        return self.single_thread_job(env, agent_amount,routes)

    def single_thread_job(self, env, agent_amount,routes=None):
        if routes is not None:
            for i in range(agent_amount):
                routes[i] = routes[i][1:-1]
        workerList = [Worker(agentID=i, local_agent_encoder=self.local_agent_encoder,
                             local_decoder=self.local_decoder, target_inputs=env.target_inputs,
                             local_target_encoder=self.local_target_encoder,
                             decode_type=self.decode_type) for i in
                      range(agent_amount)]
        global_time=0
        while True:
            all_finished = True
            global_time += 0.1
            for i in range(agent_amount):
                if workerList[i].next_select_gap <= 0:
                    agent_inputs = env.get_agent_inputs(workerList,i)
                    if routes is None:
                        next_target_index, _ = workerList[i].work(env, agent_inputs)
                        if next_target_index == 0:
                            workerList[i].sum_distance += 0.1
                    else:
                        next_target = routes[i][0]
                        next_target_index, _ = workerList[i].work(env, agent_inputs, next_target)
                        routes[i] = routes[i][1:]
                    if next_target_index is not None:
                        if next_target_index.item() != 0:
                            env.global_mask = env.global_mask.scatter_(dim=1, index=next_target_index.unsqueeze(1), value=1)
                
                if self.training:
                    workerList[i].next_select_gap += -0.1
                else:
                    workerList[i].next_select_gap += -0.01
                
                if workerList[i].next_select_gap < 0:
                    workerList[i].next_select_gap = 0
                all_finished = all_finished and workerList[i].finish
            if all_finished:
                for i in range(agent_amount):
                    workerList[i].sum_distance += workerList[i].get_sum_distance()
                break
            if self.training and len(workerList[i].action_list)>=150:
                for i in range(agent_amount):
                    workerList[i].sum_distance += workerList[i].get_sum_distance()+10
                break
        cost_list = []
        route_list = []
        reward_list = []
        
        episode_buffer = []
        for i in range(5):
            episode_buffer.append([])
        for i in range(agent_amount):
            if self.training: 
                if len(workerList[i].action_list)<=150:
                    cost_list.append(workerList[i].get_sum_distance())
                    reward_list.append(workerList[i].sum_distance)
                    route_list.append(workerList[i].point_list)
                    episode_buffer[0] += workerList[i].observation_agent
                    episode_buffer[1] += workerList[i].observation_depot
                    episode_buffer[2] += workerList[i].observation_city
                    episode_buffer[3] += workerList[i].observation_mask
                    episode_buffer[4] += workerList[i].action_list
            else:
                cost_list.append(workerList[i].get_sum_distance())
                reward_list.append(workerList[i].sum_distance)
                route_list.append(workerList[i].point_list)
                episode_buffer[0] += workerList[i].observation_agent
                episode_buffer[1] += workerList[i].observation_depot
                episode_buffer[2] += workerList[i].observation_city
                episode_buffer[3] += workerList[i].observation_mask
                episode_buffer[4] += workerList[i].action_list

        cost_set = torch.stack(cost_list)  # [agent_amount,1]
        reward_set = torch.stack(reward_list)
        route_set = route_list

        per_agent_reward = -torch.max(reward_set).unsqueeze(0).repeat(len(episode_buffer[4]))
        average_reward = -torch.max(cost_set)
        max_length = torch.max(cost_set)
        return route_set, per_agent_reward, average_reward, max_length, episode_buffer

    def get_log_p(self, _log_p, pi):
        """	args:
            _log_p: (batch, city_t, city_t)
            pi: (batch, city_t), predicted tour
            return: (batch) sum of the log probability of the chosen targets
        """
        log_p = torch.sum(torch.gather(input=_log_p, dim=2, index=pi[:, 1:, None]), dim=1)
        return log_p
