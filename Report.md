# Report
---

## Learning algorithm

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, agent must get an average score of +13 over 100 consecutive episodes. 
Training algorithm is `In [6]: dqn` inside [Navigation.ipynb](https://github.com/AInitikesh/DRLND-DQN-Banana-Navigation/blob/main/Navigation.ipynb). This function iterates over `n_episodes=2000` to train the dqn agen model. Max lenght of episode can be `max_t=10000`. Epsilon-greedy action selection starts at `eps_start=0.5` and it decreases over the training episodes by `eps_decay=0.98` and can go minimum upto `eps_end=0.01`. The epsilon-greedy values were choosen by trial and error method. Maximum timesteps value should be equal to Agent replay buffer. After 2k episodes model was not learning much and average score was constant so it doesn't makes sense to train the Agent after 2K steps. 

### DQN Agent Hyper Parameters

- BUFFER_SIZE (int): replay buffer size
- BATCH_SIZ (int): mini batch size
- GAMMA (float): discount factor
- TAU (float): for soft update of target parameters
- LR (float): learning rate for optimizer
- UPDATE_EVERY (int): how often to update the network

Where 
`BUFFER_SIZE = int(1e6)`, `BATCH_SIZE = 128`, `GAMMA = 0.99`, `TAU = 1e-3`, `LR = 0.0001` and `UPDATE_EVERY = 2`  

Along with this Agent was implemented in DQN mode and Double DQN mode setting `double_dqn` will activate the double dqn. 

### Neural Network
Two neural network Architectures were implemented for evaluation purpose.
1) [QNetwork model](https://github.com/AInitikesh/DRLND-DQN-Banana-Navigation/blob/50216e48211c18ea37ec732a0ff4e2265f713c06/model.py#L5) - which consist of an input layer of state size(37), two fully connected hidden layes of size 64 having relu activation and output fully connected layer size of action_size(4)

2) [Dueling QNetwork model](https://github.com/AInitikesh/DRLND-DQN-Banana-Navigation/blob/50216e48211c18ea37ec732a0ff4e2265f713c06/model.py#L29) - which consist of an input layer of state size(37), one fully connected hidden layes of size 64 having relu activation. This layer splits then into two streams. First Stream with fully connected hidden layer of size 4 with relu activation and outputs state value V(s) of size 1. Second stream with fully connected hidden layer of size 32 with relu activation and outputs advantage values A(s, a) of size action_size(4). Finally these layers merge into one as Q(s, a) = V(s) + Normalised A(s, a)

We can choose between QNetwork or Dueling QNetwork by setting `duel_q` to true while intialising the agent. 

## Plot of Rewards

![Reward Plot QNetwork](https://github.com/AInitikesh/DRLND-DQN-Banana-Navigation/blob/main/plots/vanila-dqn.png)

```
Episode 100	Average Score: 5.53
Episode 200	Average Score: 12.09
Episode 236	Average Score: 13.01
Environment solved in 136 episodes!	Average Score: 13.01
```

![Reward Plot Dueling QNetwork](https://github.com/AInitikesh/DRLND-DQN-Banana-Navigation/blob/main/plots/dueling-dqn.png)

```
Episode 100	Average Score: 4.10
Episode 200	Average Score: 11.13
Episode 263	Average Score: 13.04
Environment solved in 163 episodes!	Average Score: 13.04
```

## Ideas for Future Work

Dueling network and Double Q learning methods should idealy out perform the scores of vanila DQN. As a future task I might retune the hyperparamets to suite Dueling network and Double Q learning methods. Further improvements could be implementing [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952), [multi-step bootstrap targets](https://arxiv.org/abs/1602.01783), [Distributional DQN](https://arxiv.org/abs/1707.06887), [Noisy DQN](https://arxiv.org/abs/1706.10295). Also insted of capturing state space of 37 dimensions we could directly use pixel values of the environemt image and train a network with CNN layers at the begining of Q network.