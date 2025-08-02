from A2C.ReplayBuffer import ReplayBuffer
from A2C.ActorCritic import Actor,Critic
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

class Agent():
    def __init__(self , args , env , hidden_layer_list=[64,64]):
        # Hyperparameter
        self.max_train_steps = args.max_train_steps
        self.evaluate_freq_steps = args.evaluate_freq_steps
        self.num_actions = args.num_actions
        self.num_states = args.num_states
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.lr = args.lr     
        
        # Variable
        self.episode_count = 0
        self.total_steps = 0
        self.evaluate_count = 0

                
        # other
        self.env = env
        self.env_eval = copy.deepcopy(env)
        self.replay_buffer = ReplayBuffer(args)
        
        
        # The model interacts with the environment and gets updated continuously
        self.actor = Actor(args , hidden_layer_list.copy())
        self.critic = Critic(args , hidden_layer_list.copy())
        print(self.actor)
        print(self.critic)

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=1e-5)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr, eps=1e-5)
        
        
    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float)

        with torch.no_grad():
            s = torch.unsqueeze(state, dim=0)
            action_probability = self.actor(s).numpy().flatten()
            action = np.random.choice(self.num_actions, p=action_probability)
        return action

    def evaluate_action(self, state):
        state = torch.tensor(state, dtype=torch.float)
        with torch.no_grad():
            s = torch.unsqueeze(state, dim=0)
            action_probability = self.actor(s)
            action =  torch.argmax(action_probability).item()
        return action
        
    def train(self):
        time_start = time.time()
        epoch_reward_list = []
        epoch_count_list = []
        epoch_count = 0
        # Training loop
        while self.total_steps < self.max_train_steps:
            # reset environment
            s, info = self.env.reset()

            while True:                
                a = self.choose_action(s)
                # interact with environment
                s_ , r , done, truncated, _ = self.env.step(a)   
                done = done or truncated
                # stoare transition in replay buffer
                self.replay_buffer.store(s, a, [r], s_, [done])
                # update state
                s = s_
                
                if self.replay_buffer.count >= self.batch_size:
                    self.update()
                    epoch_count += 1
            
                if self.total_steps % self.evaluate_freq_steps == 0:
                    self.evaluate_count += 1
                    evaluate_reward = self.evaluate(self.env_eval)
                    epoch_reward_list.append(evaluate_reward)
                    epoch_count_list.append(epoch_count)
                    time_end = time.time()
                    h = int((time_end - time_start) // 3600)
                    m = int(((time_end - time_start) % 3600) // 60)
                    second = int((time_end - time_start) % 60)
                    print("---------")
                    print("Time : %02d:%02d:%02d"%(h,m,second))
                    print("Training epoch : %d\tStep : %d / %d"%(epoch_count,self.total_steps,self.max_train_steps))
                    print("Evaluate count : %d\tEvaluate reward : %0.2f"%(self.evaluate_count,evaluate_reward))
                    
                self.total_steps += 1
                if done or truncated :
                    break
            epoch_count += 1

        # Plot the training curve
        plt.plot(epoch_count_list, epoch_reward_list)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Training Curve")
        plt.show()
        
    def update(self):
        s, a, r, s_, done = self.replay_buffer.numpy_to_tensor()
        
        a = a.view(-1, 1)  # Reshape action from (N) -> (N, 1) for gathering
        
        # get target value and advantage
        value = self.critic(s)
        with torch.no_grad():            
            next_value = self.critic(s_)
            target_value = r + self.gamma * next_value * (1 - done)
            advantage = target_value - value # baseline advantage
            
        # Update critic
        critic_loss = F.mse_loss(value, target_value)  # Mean Squared Error loss
        self.optimizer_critic.zero_grad()
        critic_loss.backward()  # Backpropagation
        self.optimizer_critic.step()
        
        # Update actor
        prob = self.actor(s).gather(dim=1, index=a)  # Get action probability from the model
        log_prob = torch.log(prob + 1e-10)  # Add small value to avoid log(0)
        actor_loss = (-advantage * log_prob).mean()  # Calculate loss
        self.optimizer_actor.zero_grad()
        actor_loss.backward()  # Backpropagation
        self.optimizer_actor.step()
        
        
        
        
    def evaluate(self , env):
        times = 10
        evaluate_reward = 0
        
        for i in range(times):
            s , info = env.reset()
            episode_reward = 0
            while True:
                a = self.evaluate_action(s)  # We use the deterministic policy during the evaluating
                s_, r, done, truncted, _ = env.step(a)

               
                episode_reward += r
                s = s_
                
                if truncted or done:
                    break
            evaluate_reward += episode_reward

        return evaluate_reward / times
    