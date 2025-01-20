from environment import FirePropagationEnv
from stable_baselines3 import PPO
import numpy as np
from utils import squareAnimationPlot

# Testing environment with render 
class TestForestFireEnv(FirePropagationEnv):
    def __init__(self,neighbours,neighboursBoolTensor, grid_size, threshold, max_steps, extinguish_limit):
        # Initialize parent class
        super().__init__(neighbours,neighboursBoolTensor, grid_size, threshold, max_steps, extinguish_limit)
        self.historicalSteps = []
    
    # Save historical animation data to render
    def episode(self):
        
        # Load the model
        model = PPO.load("ppo_fire_agent")
        observation,_ = self.reset()
        done = False
        
        while not done:
            # The agent evaluatea the environment and decides an action
            actions,_ = model.predict(observation, deterministic = True)
            print(actions)
            
            # The Agent acts and the anvironment envolves one time step
            observation, reward, done, truncated, info = self.step(actions=actions)
            
            # Save the step 
            self.historicalSteps.append(np.copy(observation))
            #print(observation)
            
        # render animation
        self.render()
        
    def render(self, fileName='animation', interval=100):
        print('Starting simulation, wait a sec...')
        squareAnimationPlot(fileName,
                            self.historicalSteps,
                            interval,
                            p_bond=self.threshold,
                            p_site=1)
        print('Done.')
            
if __name__ == "__main__":
    shape = 50
    parameters = {
    'neighbours': [(-1,0),(1,0),(0,1),(0,-1)],
    'neighboursBoolTensor': np.ones((4,shape,shape), dtype=bool),
    'grid_size':shape,
    'threshold':0.55,
    'max_steps': 70,
    'extinguish_limit':1
    }
    TestFire = TestForestFireEnv(**parameters)
    TestFire.episode()