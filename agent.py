import numpy as np
import random

from rl_utilities import ExperienceReplay
from models import NNModel


class QAgent():
    def __init__(self, n_actions, state_size, architecture, epsilon_exp_decay=0.999, epsilon_final=0.05, 
                 gamma=0.99, exp_replay_size = int(1e5), batch_size=128, initial_exploration_steps = 1e4,
                 tau = 1e-3, learning_rate=1e-3, update_every=1):
        self.epsilon = 1.0
        self.epsilon_final = epsilon_final
        self.eps_decay = epsilon_exp_decay
        self.n_actions = n_actions
        self.state_size = state_size
        self.gamma = gamma
        self.exp_replay = ExperienceReplay(size=exp_replay_size)
        self.batch_size = batch_size
        self.neural_net = NNModel(arch=architecture, batch_size=self.batch_size, n_outputs=n_actions, state_shape=state_size, learning_rate=learning_rate)
        self.target_net = NNModel(arch=architecture, batch_size=self.batch_size, n_outputs=n_actions, state_shape=state_size, learning_rate=learning_rate)
        self.initial_exploration_steps = initial_exploration_steps
        self.tau = tau
        self.update_every = update_every
        self.c = 0  
    
    def choose_action(self, state, greedy=False):
        q_values = self._get_q_values(state, use_target_net=False)
        best_action = np.argmax(q_values)
        if greedy:
            # Choose the greedy action
            action = best_action
        else:
            # Perform epsilon-greedy and update epsilon
            if random.random() > self.epsilon:
                action = best_action
            else:
                action = random.choice(range(self.n_actions))
        return action
    
    def _update_epsilon(self):
        # Epsilon exponential decay
        self.epsilon =  self.eps_decay * self.epsilon + (1-self.eps_decay)*self.epsilon_final
        
    def _get_q_values(self, state, use_target_net=True):
        if len(state.shape)<=len(self.state_size):
            state = np.expand_dims(state, 0)
            
        if use_target_net:
            q_values = self.target_net.predict(state)
        else:
            q_values = self.neural_net.predict(state)
        assert q_values.shape[1] == self.n_actions
        assert len(q_values.shape) == 2
        return q_values
        
    def step(self, state, action, reward, next_state):
        self.c+=1
        # 1. Add observation to the deque
        observation = (np.squeeze(state), action, reward, np.squeeze(next_state))
        self.exp_replay.append(observation)
        
        # 2. Update the neural net
        if (self.exp_replay.length > self.initial_exploration_steps): 
            self._update_epsilon()
            if ((self.c % self.update_every)==0):
                self.c += 1
                sample = self.exp_replay.draw_sample(sample_size=self.batch_size)
                states, actions, rewards, next_states = sample
                # Train
                batch_x = states
                batch_y = rewards + self.gamma*np.max(self._get_q_values(next_states, use_target_net=True), axis=1, keepdims=True) # TD_target
                self.neural_net.train(batch_x, batch_y, actions)
                # update target net
                self.target_net.copy_weights_from(self.neural_net.net, tau=self.tau)

        