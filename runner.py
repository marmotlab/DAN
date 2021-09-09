import torch
import torch.optim as optim
import numpy as np
import ray
from config import config
from model import Model
from env import Env
from ortools_routes import ortools_routes

cfg = config()


@ray.remote(num_gpus= 4/cfg.meta_agent_amount,num_cpus=1)
class Runner(object):
    def __init__(self, metaAgentID, cfg, imitation=False, decode_type='sampling'):
        self.ID = metaAgentID
        self.model = Model(cfg)
        self.baseline_model = Model(cfg,decode_type='greedy')
        self.model.to(cfg.device)
        self.baseline_model.to(cfg.device)
        self.local_model_gradient = []
        
        self.reward_buffer = []
        self.max_length_buffer = []
        self.baseline_buffer = []
        self.episode_buffer = []
        for i in range(5):
            self.episode_buffer.append([])

        self.decode_type = decode_type
        self.imitation = imitation
        self.cfg = cfg

    def run_model(self, env, routes=None):
        return self.model(env, self.cfg.agent_amount, routes)

    def run_baseline(self, env):
        return self.baseline_model(env,self.cfg.agent_amount)

    def get_logp(self):
        agent_inputs = torch.cat(self.episode_buffer[0]).squeeze(0).cuda()
        depot_inputs = torch.cat(self.episode_buffer[1]).squeeze(0).cuda()
        city_inputs = torch.cat(self.episode_buffer[2]).squeeze(0).cuda()
        mask = torch.cat(self.episode_buffer[3]).squeeze(0).cuda()
        agent_feature = self.model.local_agent_encoder(agent_inputs)
        target_feature = self.model.local_target_encoder(depot_inputs, city_inputs)
        _, log_prob = self.model.local_decoder(target_feature=target_feature,
                                                   current_state=torch.mean(target_feature,dim=1).unsqueeze(1),
                                                   agent_feature=agent_feature,
                                                   mask=mask,
                                                   decode_type=self.decode_type)
        action_list=torch.cat(self.episode_buffer[4]).squeeze(0).cuda()
        logp=torch.gather(log_prob,1,action_list.unsqueeze(1))
        entropy=(log_prob*log_prob.exp()).sum(dim=-1).mean()
        return logp, entropy

    def get_advantage(self, reward_buffer, baseline):
        advantage = (reward_buffer - baseline)
        return advantage

    def get_loss(self, advantage, log_p_buffer, entropy_buffer):
        policy_loss = -log_p_buffer.squeeze(1) * advantage.detach()
        loss = policy_loss.sum()/(self.cfg.batch_size*self.cfg.agent_amount)
        return loss

    def get_gradient(self, loss):
        self.model.zero_grad()
        loss.backward()
        g = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1000, norm_type=2)
        self.local_model_gradient = []
        for local_param in self.model.parameters():
            self.local_model_gradient.append(local_param.grad)
        return g

    def set_model_weights(self, global_weights):
        self.model.load_state_dict(global_weights)

    def set_baseline_model_weights(self, baseline_weights):
        self.baseline_model.load_state_dict(baseline_weights)

    def sample(self):
        env = Env(self.cfg)
        with torch.no_grad():
            route_set,reward, reward_scalar, max_length, episode_buffer = self.run_model(env)
        env.global_mask = env.generate_mask()
        with torch.no_grad():
            _, _, baseline, _ , _ = self.run_baseline(env)
        self.reward_buffer += reward
        self.max_length_buffer.append(max_length)
        self.baseline_buffer += baseline.expand_as(reward)
        for i in range(5):
            self.episode_buffer[i] += episode_buffer[i]

    def return_gradient(self):
        reward_buffer = torch.stack(self.reward_buffer)
        log_p_buffer, entropy_loss = self.get_logp()
        baseline_buffer = torch.stack(self.baseline_buffer)
        advantage = self.get_advantage(reward_buffer=reward_buffer, baseline=baseline_buffer)
        loss = self.get_loss(advantage, log_p_buffer, entropy_loss)
        grad_norm = self.get_gradient(loss)
        max_length = torch.stack(self.max_length_buffer).squeeze(0).mean()
        self.reward_buffer = []
        self.episode_buffer = []
        for i in range(5):
            self.episode_buffer.append([])
        self.max_length_buffer = []
        self.baseline_buffer = []
        
        # if you want to random the number of cities and agents
        # self.cfg.agent_amount = np.random.randint(5,10)
        # self.cfg.target_size = np.random.randint(50,200)
        return self.local_model_gradient, loss.mean().item(), grad_norm, advantage.mean().item(), max_length.item(), entropy_loss.mean().item(),-max_length.item()


if __name__ == '__main__':
    cfg = config()
    cfg.agent_amount = 5
    cfg.target_size = 20
    env = Env(cfg)
    runner = Runner(1)
    for i in range(16):
        runner.sample()
    for i in range(5):
        print(torch.cat(runner.episode_buffer[i]).squeeze(0).size())
    runner.return_gradient()
    # print(reward)
