import torch
import ray
import numpy as np
from config import config
from model import Model
from env import Env

cfg = config()


#@ray.remote(num_gpus= 4 / cfg.meta_agent_amount, num_cpus=1) # use this for training and sampling test
class TestRunner(object):
    def __init__(self, metaAgentID, cfg, decode_type='sampling'):
        self.ID = metaAgentID

        self.decode_type = decode_type
        self.model = Model(cfg,self.decode_type, training=False)
        self.model.to(cfg.device)
        self.local_decoder_gradient = []
        self.local_agent_encoder_gradient = []
        self.local_target_encoder_gradient = []
        self.agent_amount = cfg.agent_amount

    def run(self, env):
        return self.model(env,self.agent_amount)

    def set_weights(self, global_weights):
        self.model.load_state_dict(global_weights)

    def sample(self, env):
        with torch.no_grad():
            route_set, _, _, max_length, _ = self.run(env)
        #return max_length,route_set # use this code for plot
        return max_length


if __name__ == '__main__':
    cfg = config()
    env = Env(cfg)
    env1 = Env(cfg)
    # device = 'cuda:0'
    # agent_encoder = AgentEncoder(cfg)
    # agent_encoder.to(device)
    # target_encoder = TargetEncoder(cfg)
    # target_encoder.to(device)
    # decoder = Decoder(cfg)
    # decoder.to(device)
    # workerList = [Worker(agentID=i, local_agent_encoder=agent_encoder, local_target_encoder=target_encoder,
    #                      local_decoder=decoder, target_inputs=env.target_inputs) for i in range(cfg.agent_amount)]
    # agent_inputs = env.get_agent_inputs(workerList)
    runner = TestRunner(1, cfg)
    # cost_set, route_set, log_p_set, reward_set,reward = runner.single_thread_job(cfg=cfg, env=env)
    # baseline = torch.Tensor([0]).cuda()
    # advantage = runner.get_advantage(reward.expand_as(cost_set), baseline)
    # loss = runner.get_loss(advantage, log_p_set)
    # loss.backward()
    reward1 = runner.sample(env)
    reward2 = runner.sample(env1)
    baseline = torch.stack([(reward1 - 1).unsqueeze(0).unsqueeze(0).repeat(5, 1),
                            (reward2 - 1).unsqueeze(0).unsqueeze(0).repeat(5, 1)])
    # baseline size should be [buffer_size,agent_size,1]
    g = runner.return_gradient(baseline)
