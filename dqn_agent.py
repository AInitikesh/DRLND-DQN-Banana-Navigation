import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim
from segment_tree import *

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 1e-3               # learning rate 
UPDATE_EVERY = 2        # how often to update the network
PER_E = 0.01
PER_A = 0.6
PER_B = 0.4
PER_B_INC = 0.001

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, double_dqn=False):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
        self.double_dqn = double_dqn
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        idx, weights, states, actions, rewards, next_states, dones = experiences
        

        ## TODO: compute and minimize the loss
        "*** YOUR CODE HERE ***"
        Q_expected = self.qnetwork_local(states).gather(1, actions)
               
        if self.double_dqn:
            next_actions = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(1)
            Q_targets_next = self.qnetwork_target(next_states).gather(1, next_actions).detach()
        else:
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        Q_targets = rewards + gamma * ((Q_targets_next) * (1 - dones))    
        
        delta = abs(Q_targets - Q_expected.detach()).numpy().tolist()
 
        
        td = (Q_targets - Q_expected)
        loss =  td ** 2
        loss *= weights
        loss = loss.mean()
        
        self.memory.update_priorities(idx, delta)

#         loss = (weights * F.mse_loss(Q_expected, Q_targets)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a PrioratisedReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.per_a = PER_A
        self.per_e = PER_E
        self.per_b = PER_B
        self.per_b_inc = PER_B_INC
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.sum_priorities = 0
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "priority"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done, self.per_e)
        self.sum_priorities += self.per_e ** self.per_a
        self.memory.append(e)
    
    def sample(self):
        """Priority sample a batch of experiences from memory."""
        sampling_probabililites =  [((i.priority ** self.per_a) / self.sum_priorities) for i in self.memory]
        idx = random.choices(population=range(len(self.memory)), weights=sampling_probabililites, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([self.memory[i].state for i in idx if i is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([self.memory[i].action for i in idx if i is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([self.memory[i].reward for i in idx if i is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([self.memory[i].next_state for i in idx if i is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([self.memory[i].done for i in idx if i is not None]).astype(np.uint8)).float().to(device)
        
        self.per_b = np.min([1., self.per_b + self.per_b_inc])
        weights = np.vstack([((len(self.memory) * sampling_probabililites[i]) ** (-self.per_b)) for i in idx if i is not None])
        weights = torch.from_numpy(weights / weights.max()).float().to(device)
        
        return (idx, weights, states, actions, rewards, next_states, dones)

    def update_priorities(self, idx, delta):
        for i, d in zip(idx, delta):
            e = self.memory[i]
            new_priority = d[0] + self.per_e
            self.sum_priorities = self.sum_priorities - (e.priority ** self.per_a) + (new_priority ** self.per_a)
            self.memory[i] = self.experience(e.state, e.action, e.reward, e.next_state, e.done, new_priority)
        
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)