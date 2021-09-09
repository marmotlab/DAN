import ray
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import copy
import numpy as np
import time
from scipy.stats import ttest_rel

from env import Env
from model import Model
from config import config
from runner import Runner
from runner_for_test import TestRunner
from pytorch_lamb import Lamb

ray.init()
cfg = config()
writer = SummaryWriter(cfg.train_path)
if not os.path.exists(cfg.model_path):
    os.makedirs(cfg.model_path)
if not os.path.exists(cfg.gifs_path):
    os.makedirs(cfg.gifs_path)


def main(cfg):
    device = cfg.device
    global_model = Model(cfg)
    global_model.share_memory()
    global_model.to(device)

    optimizer = optim.AdamW(global_model.parameters(), lr=cfg.lr)

    lr_decay = optim.lr_scheduler.StepLR(optimizer, step_size=256, gamma=0.96)

    meta_agent_list_rl = [Runner.remote(metaAgentID=i, cfg=cfg) for i in range(cfg.meta_agent_amount)]
    meta_agent_list_il = [Runner.remote(metaAgentID=i, cfg=cfg,imitation=True) for i in range(0)]
    meta_agent_list = meta_agent_list_rl + meta_agent_list_il

    # info for tensorboard
    average_loss = 0
    average_advantage = 0
    average_grad_norm = 0
    average_rewards = 0
    average_max_length = 0
    average_entropy = 0

    global_step = 0

    if cfg.load_model:
        checkpoint = torch.load(cfg.model_path + '/model_states.pth')
        global_step = checkpoint['step']
        global_model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_decay.load_state_dict(checkpoint['lr_decay'])

        print("load model at", global_step)
        print(optimizer.state_dict()['param_groups'][0]['lr'])

    # get global network weights
    global_weights = global_model.state_dict()

    # update local network
    update_local_network_job_list = []
    for i, meta_agent in enumerate(meta_agent_list):
        update_local_network_job_list.append(meta_agent.set_model_weights.remote(global_weights))
    baseline_weights = copy.deepcopy(global_weights)
    update_baseline_network_job_list = []
    for i, meta_agent in enumerate(meta_agent_list):
        update_baseline_network_job_list.append(meta_agent.set_baseline_model_weights.remote(baseline_weights))

    baseline_value = None
    test_set = np.random.randint(low=0, high=1e8, size=[2048 // cfg.meta_agent_amount, cfg.meta_agent_amount])

    try:
        while True:
            global_step += 1
            #print(global_step)
            sample_job_list = []
            for i, meta_agent in enumerate(meta_agent_list):
                sample_job_list.append(meta_agent.sample.remote())

            if global_step % cfg.batch_size == 0:

                # get gradient and loss from runner
                get_gradient_job_list = []
                for i, meta_agent in enumerate(meta_agent_list):
                    get_gradient_job_list.append(meta_agent.return_gradient.remote())
                gradient_set_id, _ = ray.wait(get_gradient_job_list, num_returns=cfg.meta_agent_amount)
                gradient_loss_set = ray.get(gradient_set_id)

                for gradients, loss, grad_norm, advantage, max_length,entropy,reward in gradient_loss_set:
                    average_max_length += max_length
                    average_loss += loss
                    average_advantage += advantage
                    average_grad_norm += grad_norm
                    average_entropy += entropy
                    average_rewards += reward

                    optimizer.zero_grad()
                    for g, global_param in zip(gradients, global_model.parameters()):
                        global_param._grad = g

                    # update networks
                    optimizer.step()
                lr_decay.step()

                update_local_network_job_list = []
                for i, meta_agent in enumerate(meta_agent_list):
                    update_local_network_job_list.append(meta_agent.set_model_weights.remote(global_weights))

                # tensorboard update
                if global_step % cfg.tensorboard_batch == 0:
                    writer.add_scalar('loss/loss',
                                      average_loss / (cfg.meta_agent_amount * cfg.tensorboard_batch / cfg.batch_size),
                                      global_step)
                    average_loss = 0
                    writer.add_scalar('loss/entropy',
                                      average_entropy / (cfg.meta_agent_amount * cfg.tensorboard_batch / cfg.batch_size),
                                      global_step)
                    average_entropy = 0
                    writer.add_scalar('loss/advantage',
                                      average_advantage / (
                                              cfg.meta_agent_amount * cfg.tensorboard_batch / cfg.batch_size),
                                      global_step)
                    average_advantage = 0
                    writer.add_scalar('grad/grad_norm',
                                      average_grad_norm / (
                                              cfg.meta_agent_amount * cfg.tensorboard_batch / cfg.batch_size),
                                      global_step)
                    average_grad_norm = 0
                    writer.add_scalar('perf/reward', average_rewards / (cfg.meta_agent_amount * cfg.tensorboard_batch / cfg.batch_size), global_step)
                    average_rewards = 0
                    writer.add_scalar('perf/max_length', average_max_length / (
                            cfg.meta_agent_amount * cfg.tensorboard_batch / cfg.batch_size), global_step)
                    average_max_length = 0

            # save model
            if global_step % cfg.log_size == 0:
                model_states = {"model": global_model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "lr_decay": lr_decay.state_dict(),
                                "step": global_step}
                torch.save(obj=model_states, f=cfg.model_path + '/model_states.pth')

            # update baseline model every 1024 steps
            if global_step % (2048) == 0:
                # stop the training
                ray.wait(update_local_network_job_list, num_returns=cfg.meta_agent_amount)
                for a in meta_agent_list:
                    ray.kill(a)
                torch.cuda.empty_cache()
                time.sleep(5)
                print('evaluate baseline model at ', global_step)

                # test the baseline model on the new test set
                if baseline_value is None:
                    test_agent_list = [TestRunner.remote(metaAgentID=i, cfg=cfg, decode_type='greedy') for i in
                                       range(cfg.meta_agent_amount)]
                    update_local_network_job_list = []
                    for _, test_agent in enumerate(test_agent_list):
                        update_local_network_job_list.append(test_agent.set_weights.remote(baseline_weights))

                    max_length_list = []
                    for i in range(2048 // cfg.meta_agent_amount):
                        sample_job_list = []
                        for j, test_agent in enumerate(test_agent_list):
                            env = Env(cfg, test_set[i][j])
                            sample_job_list.append(test_agent.sample.remote(env))
                        sample_done_id, _ = ray.wait(sample_job_list, num_returns=cfg.meta_agent_amount)
                        rewards = ray.get(sample_done_id)
                        max_length_list = max_length_list + rewards
                    baseline_value = torch.stack(max_length_list).squeeze(0).cpu().numpy()
                    for a in test_agent_list:
                        ray.kill(a)

                # test the current model's performance
                test_agent_list = [TestRunner.remote(metaAgentID=i, cfg=cfg, decode_type='greedy') for i in
                                   range(cfg.meta_agent_amount)]
                update_local_network_job_list = []
                for _, test_agent in enumerate(test_agent_list):
                    update_local_network_job_list.append(test_agent.set_weights.remote(global_weights))

                max_length_list = []
                for i in range(2048 // cfg.meta_agent_amount):
                    sample_job_list = []
                    for j, test_agent in enumerate(test_agent_list):
                        env = Env(cfg, test_set[i][j])
                        sample_job_list.append(test_agent.sample.remote(env))
                    sample_done_id, _ = ray.wait(sample_job_list, num_returns=cfg.meta_agent_amount)
                    rewards = ray.get(sample_done_id)
                    max_length_list = max_length_list + rewards
                test_value = torch.stack(max_length_list).squeeze(0).cpu().numpy()

                # restart training
                print('lr', optimizer.state_dict()['param_groups'][0]['lr'])
                for a in test_agent_list:
                    ray.kill(a)
                               
                time.sleep(5)
                meta_agent_list_rl = [Runner.remote(metaAgentID=i, cfg=cfg) for i in range(cfg.meta_agent_amount)]
                meta_agent_list_il = [Runner.remote(metaAgentID=i, cfg=cfg,imitation=True) for i in range(0)]
                meta_agent_list = meta_agent_list_rl + meta_agent_list_il

                for i, meta_agent in enumerate(meta_agent_list):
                    update_local_network_job_list.append(meta_agent.set_model_weights.remote(global_weights))
                update_baseline_network_job_list = []       
                for i, meta_agent in enumerate(meta_agent_list):        
                    update_baseline_network_job_list.append(meta_agent.set_baseline_model_weights.remote(baseline_weights))

                # update baseline if the model improved more than 5%
                print('test value', test_value.mean())
                print('baseline value', baseline_value.mean())
                if test_value.mean() < baseline_value.mean():
                    _, p = ttest_rel(test_value, baseline_value)
                    print('p value', p)
                    if p < 0.05:
                        print('update baseline model at ', global_step)
                        global_weights = global_model.state_dict()

                        baseline_weights = copy.deepcopy(global_weights)
                        update_baseline_network_job_list = []
                        for i, meta_agent in enumerate(meta_agent_list):
                            update_baseline_network_job_list.append(meta_agent.set_baseline_model_weights.remote(baseline_weights))

                        test_set = np.random.randint(low=0, high=1e8,
                                                     size=[2048 // cfg.meta_agent_amount, cfg.meta_agent_amount])
                        print('update test set')
                        baseline_value = None

    except KeyboardInterrupt:
        print("CTRL-C pressed. killing remote workers")
        for a in meta_agent_list:
            ray.kill(a)


if __name__ == '__main__':
    main(cfg)
