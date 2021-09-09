# DAN
Public version of the decentralized, attention-based mTSP code

## Setting up Code
- Python == 3.8
- Pytorch == 1.8
- Ray == 1.2

## Running Code
- Train the code by running the following command, set the number of cities and the number of agents as you want. To train on the random scale mTSP, use Line 110-111 in runner.py. 
```bash
python driver.py --target_size 50 --agent_amount 5
```
- Test the trained model using either greedy strategy or sampling strategysh.
```bash
python test.py --target_size 50 --agent_amount 5
``` 
```bash
python sample_test.py --target_size 50 --agent_amount 5
```
- Specify the test strategy to plot the instances.
```bash
python plot.py --target_size 50 --agent_amount 5 --strategy 'sampling'
```
## Key Files
- driver.py  - Driver of program. Holds global network.
- runner.py  - Compute node for training. Maintains a single meta agent.
- worker.py  - A single agent in a mTSP instance.
- model.py   - Use DAN to solve the mTSP instance cooperatively.
- config.py  - Parameters for training and test.
- env.py - Define the environment class.

## Authors
[Yuhong Cao](caoyuhong@u.nus.edu)

[Zhanhong Sun](dexter.s@u.nus.edu)

[Guillaume Sartoretti](guillaume.sartoretti@gmail.com)
