from A2C.ReplayBuffer import ReplayBuffer
from A2C.ActorCritic import Network
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Agent():
    def __init__(self , args , env , hidden_layer_list=[64,64]):
        # Hyperparameter
        self.training_episodes = args.training_episodes
        self.advantage = args.advantage
        self.num_states = args.num_states
        self.num_actions = args.num_actions
        self.epochs = args.epochs
        self.lr = args.lr     
        
        # Variable
        self.episode_count = 0
        
                
        # other
        self.env = env
        self.replay_buffer = ReplayBuffer(args)
        # The model interacts with the environment and gets updated continuously
        self.actor = Network(args , hidden_layer_list.copy())
        print(self.actor)

        self.optimizer = torch.optim.Adam(self.actor.parameters() , lr = self.lr , eps=1e-5)
        
        
    def choose_action(self, state):
        with torch.no_grad():
            state = torch.unsqueeze(torch.tensor(state), dim=0)
            action_probability = self.actor(state).numpy().flatten()
            action = np.random.choice(self.num_actions, p=action_probability)
        return action

    def evaluate_action(self, state):
        with torch.no_grad():
            # choose the action that have max q value by current state
            state = torch.unsqueeze(torch.tensor(state), dim=0)
            action_probability = self.actor(state)
            action =  torch.argmax(action_probability).item()
        return action
        
    def train(self):
        episode_reward_list = []
        episode_count_list = []
        episode_count = 0
        # Training loop
        for epoch in range(self.epochs):
            # reset environment
            state, info = self.env.reset()
            done = False
            while not done:
                
                action = self.choose_action(state)

                # interact with environment
                next_state , reward , terminated, truncated, _ = self.env.step(action)   
                done = terminated or truncated
                self.replay_buffer.store(state, action, [reward], next_state, [done])

                state = next_state
            self.replay_buffer.to_episode_batch()  # Convert to episode batch
            if (epoch + 1)% self.training_episodes == 0 and epoch != 0:
                # Update the model
                self.update()
                self.replay_buffer.clear_episode_batch()  # Clear the episode batch after updating
            if epoch % 10 == 0:
                evaluate_reward = self.evaluate(self.env)
                print("Epoch : %d / %d\t Reward : %0.2f"%(epoch,self.epochs , evaluate_reward))
                episode_reward_list.append(evaluate_reward)
                episode_count_list.append(episode_count)

            episode_count += 1

        # Plot the training curve
        plt.plot(episode_count_list, episode_reward_list)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Training Curve")
        plt.show()
        
    def update(self):
        loss = 0
        base_line = 0
        for batch in self.replay_buffer.episode_batch:
            s, a, r, s_, done = batch
            base_line += self.TotalReward(r) 
        base_line /= self.replay_buffer.episode_count  # Normalize baseline by number of episodes
        for batch in self.replay_buffer.episode_batch:
            s, a, r, s_, done = batch
            a = a.view(-1, 1)  # Reshape action from (N) -> (N, 1) for gathering            
            if self.advantage == 0:
                adv = self.TotalReward(r)
            elif self.advantage == 1:
                adv = self.RewardFollowing(r)
            elif self.advantage == 2:
                adv = self.RewardFollowing(r)
                adv = adv - base_line
            #print(adv)
            prob = self.actor(s).gather(dim=1, index=a)  # Get action probability from the model
            log_prob = torch.log(prob + 1e-10)  # Add small value to avoid log(0)
            loss += (adv * log_prob).sum()  
        loss = - loss / self.replay_buffer.episode_count  # Normalize loss by number of episodes
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        
        
        
    def evaluate(self, env):
        render_env = env

        reward_list = []
        for i in range(5):
            # reset environment
            state, info = render_env.reset()
            done = False
            episode_reward = 0

            while not done:                
                action = self.evaluate_action(state)
                
                # interact with environment
                next_state , reward , terminated, truncated, _ = render_env.step(action)
                    
                done = terminated or truncated
                state = next_state
                episode_reward += reward
            
            reward_list.append(episode_reward)
        reward_list = np.array(reward_list)
        return reward_list.mean()
    
    def TotalReward(self, reward):
        return reward.sum()
    
    def RewardFollowing(self , reward):
        flip_reward = reward.flip(dims=[0])
        reward_following = flip_reward.cumsum(dim=0)  # Cumulative sum in reverse order
        adv = reward_following.flip(dims=[0])
        return adv