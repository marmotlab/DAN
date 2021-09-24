import os
import argparse
import torch


def config(args=None):
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--target_size', type=int, default=20, help="The targets amount")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")

    # model
    parser.add_argument('--embedding_dim', type=int, default=128, help="input embedding dimension")
    parser.add_argument('--tanh_clipping', type=float, default=10, help="avoid the model to be overconfident")

    # training
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.96, help='learning rate decay per epoch')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="clip the gradient l2 norm")
    parser.add_argument('--agent_amount',type=int,default=5,help="agent amount")
    parser.add_argument('--meta_agent_amount', type=int, default=8, help="meta agent amount")
    parser.add_argument('--number_of_gpu', default=4, help="number of GPUs")

    # other
    parser.add_argument('--log_size',default=512,help='every 256 step save the model')
    parser.add_argument('--tensorboard_batch', type=int, default=256, help="tensorboard batch size")
    parser.add_argument('--train_path', default='train_am', help="tensorboard path")
    parser.add_argument('--model_path', default='model_am', help="model path")
    parser.add_argument('--gifs_path', default='gifs_am', help="gifs path")
    parser.add_argument('--device',default='cuda',help="run on which GPU")
    parser.add_argument('--load_model',default=False,help='whether load model')
    parser.add_argument('--strategy', default='greedy', help='whether greedy or sampling')
    parser.add_argument('--seed', default=1023, help='seed for test')

    cfg = parser.parse_args(args=[])

    return cfg


if __name__=='__main__':
    cfg=config()
    print(cfg)
