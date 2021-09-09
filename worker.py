import torch
import numpy as np
import copy
from agentEncoder import AgentEncoder
from targetEncoder import TargetEncoder
from decoder import Decoder
from config import config
from env import Env


class Worker():
    def __init__(self, agentID, local_agent_encoder, local_decoder, local_target_encoder, target_inputs,
                 decode_type='sampling'):
        self.ID = agentID
        self.target_encoder = local_target_encoder
        self.agent_encoder = local_agent_encoder
        self.decoder = local_decoder
        self.target_encoder.share_memory()
        self.agent_encoder.share_memory()
        self.decoder.share_memory()
        
        self.target_inputs = target_inputs  # it is a list(depot,target)
        self.target_set = torch.cat((target_inputs[0], target_inputs[1]), dim=1)  # it is the depot+target input tensor
        self.agent_position = target_inputs[0]  # initial agent position at the depot
        
        self.point_list = torch.tensor([0]).cuda()  # list to store route, start at depot
        self.action_list = []
        self.observation_agent = []
        self.observation_depot = []
        self.observation_city = []
        self.observation_mask = []

        self.next_select_gap = 0  # time to select next target
        self.sum_distance = 0
        self.finish = False  # finish flag
        self.decode_type = decode_type

    def calculate_encoded_agent(self, agent_inputs):
        agent_inputs = agent_inputs - torch.cat((self.agent_position,torch.FloatTensor([[[0]]]).cuda()),dim=-1)
        agent_feature = self.agent_encoder(agent_inputs)
        return agent_feature, agent_inputs

    def calculate_encoded_target(self):
        target_inputs = copy.deepcopy(self.target_inputs)
        depot_inputs = target_inputs[0] - self.agent_position
        city_inputs = target_inputs[1] - self.agent_position
        target_feature = self.target_encoder(depot_inputs,city_inputs)
        return target_feature,depot_inputs,city_inputs

    def select_next_target(self, env, agent_inputs, next_target=None):
        if 0 in env.global_mask[0,1:]:
            
            agent_feature, agent_input = self.calculate_encoded_agent(agent_inputs=agent_inputs)
            target_feature, depot_input,city_input = self.calculate_encoded_target()
            
            self.observation_agent.append(agent_input)
            self.observation_depot.append(depot_input)
            self.observation_city.append(city_input)
            mask=copy.deepcopy(env.global_mask)
            self.observation_mask.append(mask)

            next_target_index, log_prob = self.decoder(target_feature=target_feature,
                                                       current_state=torch.mean(target_feature,dim=1).unsqueeze(1),
                                                       agent_feature=agent_feature,
                                                       mask=env.global_mask,
                                                       decode_type=self.decode_type,
                                                       next_target=next_target)
            self.action_list.append(next_target_index)
            self.point_list = torch.cat((self.point_list, next_target_index))
        else:
            self.finish = True
            next_target_index = None
        return next_target_index, self.finish

    def update_next_action_gap(self):
        index1 = self.point_list[-1].item()
        index2 = self.point_list[-2].item()
        current_position = self.target_set[:, index2]
        target_position = self.target_set[:, index1]
        self.next_select_gap = (current_position - target_position).norm(p=2, dim=1)
        self.agent_position = target_position.unsqueeze(0)

    def add_final_distance(self):
        index1 = self.point_list[-1].item()
        index2 = self.point_list[0].item()
        current_position = self.target_set[:, index2]
        depot_position = self.target_set[:, index1]
        final_distance = (current_position - depot_position).norm(p=2, dim=1)
        return final_distance

    def get_sum_distance(self):
        route = self.point_list
        d = torch.gather(input=self.target_set, dim=1, index=route[None, :, None].repeat(1, 1, 2))
        return (torch.sum((d[:, 1:] - d[:, :-1]).norm(p=2, dim=2), dim=1)
                + (d[:, 0] - d[:, -1]).norm(p=2, dim=1))  # distance from last node to first selected node)

    def work(self, env, agent_inputs,next_target=None):
        next_target_index, finish = self.select_next_target(env, agent_inputs,next_target)
        if finish is not True:
            self.update_next_action_gap()  # use add_final_distance to add 'return to depot' distance
        else:
            self.next_select_gap = 0
        return next_target_index, finish
