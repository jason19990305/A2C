import gymnasium as gym
import numpy as np
import argparse
import time

from A2C.Agent import Agent

class main():
    def __init__(self , args):
        
        env_name = 'CartPole-v1'
        env = gym.make(env_name)
        
        args.num_states = env.observation_space.shape[0]
        args.num_actions = env.action_space.n
        
        # Pring hyperparameters 
        print("---------------")
        for arg in vars(args):
            print(arg,"=",getattr(args, arg))
        print("---------------")
        
        
        
        self.agent = Agent(args, env , [128,128]) # hidden layer size   
        
        self.agent.train()       
        render_env = gym.make(env_name, render_mode="human")  
        for i in range(1000):
            self.agent.evaluate(render_env)
        render_env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for A2C")
    parser.add_argument("--evaluate_freq_steps", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--max_train_steps", type=int, default=2e5, help="Set the number of steps used for training the agent")
    parser.add_argument("--batch_size", type=int, default=3, help="Batch size")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for optimizer")
    

    args = parser.parse_args()
    
    main(args)
