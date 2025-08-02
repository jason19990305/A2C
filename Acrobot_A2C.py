import gymnasium as gym
import numpy as np
import argparse
import time

from A2C.Agent import Agent

class main():
    def __init__(self , args):
        
        env_name = 'Acrobot-v1'
        env = gym.make(env_name)
        args.num_states = 6
        args.num_actions = 3
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
    parser = argparse.ArgumentParser("Hyperparameters Setting for Policy Gradient")
    parser.add_argument("--training_episodes", type=int, default=3, help="Set the number of episodes used for training the agent")
    parser.add_argument("--advantage", type=int, default=1, help="Set advantage function, 0 for Total reward, 1 for reward following the policy")
    parser.add_argument("--epochs", type=int, default=5000, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for optimizer")
    

    args = parser.parse_args()
    
    main(args)
