import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic

# Determine whether GPU is available
sep = "\n   *****************\n"
if torch.cuda.is_available() :
    device = torch.device("cuda:0")
    print(sep + "   *** Using GPU ***" + sep)
else : 
    device = torch.device("cpu")
    print(sep + "   *** Using CPU ***" + sep)


class MADDPG_Agent():
    """ 
    The agent uses experiences from a single or multiple agents to train the agents
    using the Deep Deterministic Policy Gradient (DDPG) algorithm.

    Code is taken from the 'ddpg-pendulum' example provided by Udacity
    and modified to learn from the shared experiences of multiple agents.
    """
    
    def __init__(self, state_size, action_size, num_agents, config):
        """Initialize a DDPG_Agent object.
        
        Arguments
            state_size (int) : dimension of each state
            action_size (int): dimension of each action
            num_agents (int) : number of agents in the environment
            config (Config)) : hyperparameters
        """

        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.eps = config.EPS_INITIAL
        self.eps_decay = 1 / (config.EPS_DECAY * config.LEARN_NUM)
        self.eps_min = config.EPS_TERMINAL
        self.learn_every = config.LEARN_EVERY
        self.learn_num = config.LEARN_NUM
        self.batch_size = config.BATCH_SIZE
        self.gamma = config.GAMMA
        self.tau = config.TAU

        self.seed = random.seed(config.SEED)
        self.steps = 0  # to track number of steps

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, config.FC1_UNITS_ACTOR, config.FC2_UNITS_ACTOR, config.SEED).to(device)
        self.actor_target = Actor(state_size, action_size, config.FC1_UNITS_ACTOR, config.FC2_UNITS_ACTOR, config.SEED).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=config.LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, config.FCS1_UNITS_CRITIC, config.FC2_UNITS_CRITIC, config.SEED).to(device)
        self.critic_target = Critic(state_size, action_size, config.FCS1_UNITS_CRITIC, config.FC2_UNITS_CRITIC, config.SEED).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=config.LR_CRITIC)

        # Noise process
        self.noise = OUNoise((num_agents, action_size), config.SEED)
        #self.noise = OUNoise(action_size, config.SEED)
        

        # Replay memory
        self.memory = ReplayBuffer(action_size, config.BUFFER_SIZE, config.BATCH_SIZE, config.SEED)
    
    def step(self, states, actions, rewards, next_states, dones, agent_number):
        """Save experiences of all agents in replay memory, and
           use batch of random samples from memory to perform training step."""
        
        # Increment step count
        self.steps += 1

        # Save experience to the replay buffer
        self.memory.add(states, actions, rewards, next_states, dones)
        #self.memory.add(state, action, reward, next_state, done)
        #for i in range(self.num_agents):
        #    self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

        # Learn if enough samples are available in the replay buffer
        if (self.steps % self.learn_every == 0) and (len(self.memory) > self.batch_size):
            for _ in range(self.learn_num):
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma, agent_number)
        
    def act(self, states, add_noise):
        """Return an action taken by each agent using current policy
            given the state of each agent's environment.
            
            Returns numpy array with shape [num_agents, action_size]
            
            Arguments:
                states: numpy array of shape [num_agents, state_size]
                add_noise: boolean, True if noise should be added to actions
            """

        states = torch.from_numpy(states).float().to(device)
        actions = np.zeros((self.num_agents, self.action_size))
        self.actor_local.eval()
        with torch.no_grad():
            # get action for each agent and concatenate them
            for i, state in enumerate(states):
                actions[i, :] = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        
        # add noise to actions
        if add_noise:
            actions += self.eps * self.noise.sample()
        actions = np.clip(actions, -1, 1)
        return actions

    def play(self, state):
        """Return an action taken by each agent using current policy
            given the state of the agent's environment.
            
            Arguments:
                states: numpy array of shape [num_agents, state_size]
                add_noise: boolean, True if noise should be added to actions
        """

        state = state.reshape(1,48)
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        return action

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma, agent_number):
        """Update policy and value parameters using given batch of experience tuples.
        
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Arguments:
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        
        # Get predicted next-state action
        actions_next = self.actor_target(next_states)

        # Construct next actions vector relative to the agent
        if agent_number == 0:
            actions_next = torch.cat((actions_next, actions[:,2:]), dim=1)
        else:
            actions_next = torch.cat((actions[:,:2], actions_next), dim=1)
        
        # get Q values from target models
        Q_targets_next = self.critic_target(next_states, actions_next)
        
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        
        # Get predicted action
        actions_pred = self.actor_local(states)

        # Construct action prediction vector relative to the agent
        if agent_number == 0:
            actions_pred = torch.cat((actions_pred, actions[:,2:]), dim=1)
        else:
            actions_pred = torch.cat((actions[:,:2], actions_pred), dim=1)
        
        # Compute actor loss
        actor_loss = -self.critic_local(states, actions_pred).mean()
        
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

        # --------------------------- update epsilon --------------------------- #
        self.eps -= self.eps_decay
        self.eps = max(self.eps, self.eps_min)
        self.noise.reset()


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Arguments:
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.size = size
        self.seed = random.seed(seed)
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        #dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
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
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)