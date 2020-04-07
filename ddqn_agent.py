import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

###################### DEAFULTS ######################
# BUFFER_SIZE = int(1e5)  # replay buffer size
# BATCH_SIZE = 128        # minibatch size
# GAMMA = 0.99            # discount factor
# TAU = 1e-3              # for soft update of target parameters
# LR_ACTOR = 1e-4         # learning rate of the actor 
# LR_CRITIC = 1e-3        # learning rate of the critic
# WEIGHT_DECAY = 0        # L2 weight decay
###################### DEAFULTS ######################

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 40         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
UPDATE_EVERY = 20
UPDATE_COUNT = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MultiAgent():
    def __init__(self, state_size, action_size, random_seed, num_agents, share_critic=True):
        """Initialize a MultiAgent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            num_agents (int): number of agents in play
            share_critic (boolean): share critic with all agents or to create their own.
        """

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.num_agents = num_agents
        self.share_critic = share_critic

        self.agents = []
        for i in range(num_agents):
            self.agents.append(Agent(state_size, action_size, random_seed, with_memory=False, with_critic=not share_critic))


        if share_critic:
            raise Exception("make sure we are not here")
        #     self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        #     self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        #     self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.t_step = 0

    def act(self, states, add_noise=True):
        # return [agent.act(state, add_noise) for agent, state in zip(self.agents, states)]
        return [agent.act(state, add_noise) for agent, state in zip(self.agents, states)]

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % UPDATE_EVERY

        if len(self.memory) > BATCH_SIZE and self.t_step == 0:
            for i in range(UPDATE_COUNT):
                for agent in self.agents:
                    experiences = self.memory.sample()
                    self.learn(agent, experiences, GAMMA)

    def learn(self, agent, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """

        if not self.share_critic:
            agent.learn(experiences, gamma)
        else:
            raise Exception("make sure we are not here")
        #     states, actions, rewards, next_states, dones = experiences

        #     # ---------------------------- update critic ---------------------------- #
        #     # Get predicted next-state actions and Q values from target models
        #     actions_next = agent.actor_target(next_states)
        #     Q_targets_next = self.critic_target(next_states, actions_next)
        #     # Compute Q targets for current states (y_i)
        #     # Q_targets = rewards.view(-1, 1) + torch.mm((gamma * Q_targets_next).view(-1, 1), (1 - dones).view(-1, 1).t())
        #     Q_targets_next = gamma * Q_targets_next * (1 - dones).unsqueeze(-1)
        #     Q_targets = rewards.unsqueeze(-1) + Q_targets_next
        #     # Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        #     # Compute critic loss
        #     Q_expected = self.critic_local(states, actions)
        #     # critic_loss = F.mse_loss(Q_expected.view(-1, 1), Q_targets)
        #     critic_loss = F.mse_loss(Q_expected, Q_targets)
        #     # Minimize the loss
        #     self.critic_optimizer.zero_grad()
        #     critic_loss.backward()
        #     self.critic_optimizer.step()

        #     # ---------------------------- update actor ---------------------------- #
        #     # Compute actor loss
        #     actions_pred = agent.actor_local(states)
        #     actor_loss = -self.critic_local(states, actions_pred).mean()
        #     # Minimize the loss
        #     agent.actor_optimizer.zero_grad()
        #     actor_loss.backward()
        #     agent.actor_optimizer.step()

        #     # ----------------------- update target networks ----------------------- #
        #     self.soft_update(self.critic_local, self.critic_target, TAU)
        #     self.soft_update(agent.actor_local, agent.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save(self, actor_save_name, critic_save_name):
        if self.share_critic:
            for agent, i in zip(self.agents, range(self.num_agents)):
                torch.save(agent.actor_local.state_dict(), '{}_{}.pth'.format(actor_save_name, i))
            torch.save(self.critic_local.state_dict(), critic_save_name + '.pth')
        else:
            for agent, i in zip(self.agents, range(self.num_agents)):
                torch.save(agent.actor_local.state_dict(), '{}_{}.pth'.format(actor_save_name, i))
                torch.save(agent.critic_local.state_dict(), '{}_{}.pth'.format(critic_save_name, i))

    def load(self, actor_save_name, map_location='cpu'):
        # If all the agents have been similarly trained can we just load a single checkpoint?
        for agent, i in zip(self.agents, range(self.num_agents)):
            agent.actor_local.load_state_dict(torch.load('{}_{}.pth'.format(actor_save_name, i), map_location=map_location))


class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed, with_memory = True, with_critic = True):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            with_memory (boolean): shared memory in another agent
            with_critic (boolean): initialize the critic
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        if with_critic:
            # Critic Network (w/ Target Network)
            self.critic_local = Critic(state_size, action_size, random_seed).to(device)
            self.critic_target = Critic(state_size, action_size, random_seed).to(device)
            self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        if with_memory:
            raise Exception("no memory buffer, it is handled in the multiagent class")
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        # print(action)
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        # Q_targets = rewards.view(-1, 1) + torch.mm((gamma * Q_targets_next).view(-1, 1), (1 - dones).view(-1, 1).t())
        Q_targets_next = gamma * Q_targets_next * (1 - dones).unsqueeze(-1)
        Q_targets = rewards.unsqueeze(-1) + Q_targets_next
        # Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        # critic_loss = F.mse_loss(Q_expected.view(-1, 1), Q_targets)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.1):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()
        self.scale = 1

        self.size = size

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    # def sample(self):
    #     """Update internal state and return it as a noise sample."""
    #     x = self.state
    #     dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
    #     self.state = x + dx
    #     return torch.tensor(self.state * self.scale).float()

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["states", "actions", "rewards", "next_states", "dones"])
        self.seed = random.seed(seed)
    
    def add(self, states, actions, rewards, next_states, dones):
        """Add a new experience to memory."""
        # for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
        e = self.experience(states, actions, rewards, next_states, dones)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.states for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.actions for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.rewards for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_states for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.dones for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)