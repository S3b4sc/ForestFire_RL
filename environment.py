import gymnasium as gym
from gymnasium import spaces
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from utils import parameters

class FirePropagationEnv(gym.Env):
    """
    
    """
    def __init__(self,neighbours,neighboursBoolTensor, grid_size, threshold, max_steps, extinguish_limit):
        super(FirePropagationEnv, self).__init__()
        
        # Neighbours list
        self.neighbours = neighbours
        # Boolean neighbour tensor for regular tesellation
        self.neighboursBoolTensor = neighboursBoolTensor
        # Size of the forest
        self.grid_size = grid_size
        # Burning threshold
        self.threshold = threshold
        # Max time steps
        self.max_steps = max_steps
        self.extinguish_limit = extinguish_limit

        # Action space: extinguish cells (limited per step)
        self.action_space = spaces.Discrete(self.grid_size*self.grid_size)
        #self.action_space = spaces.MultiDiscrete([self.grid_size,self.grid_size] * self.extinguish_limit)
        #print(self.action_space.sample())
        # Historical actions record
        self.historical_actions = []
        #print(self.action_space.sample())
        # Observation space: forest of cell states (0=empty, 1=healthy, 2=burning, 3=burned)
        self.observation_space = spaces.Box(low=0, high=3, shape=(grid_size, grid_size), dtype=np.int32)
        # Initialize environment
        self.reset()

    def reset(self, seed:int = None, options:dict = None):
        '''
        Reset the environment to its initial state. 
        Args:
            seed (int, optional): A seed value for reproducibility. Defaults to None.
            **kwargs: Other keyword arguments passed by the environment wrapper.   
        Returns:
            observation (np.ndarray): The initial observation of the environment.
        '''
        super().reset(seed=seed)
        # Create initial forest: majority healthy, one burning
        self.forest = np.ones((self.grid_size, self.grid_size), dtype=np.int32)
        #self.forest[np.random.randint(self.grid_size), np.random.randint(self.grid_size)] = 2
        # Start initial fire in the center of the lattice
        self.forest[self.grid_size//2,  self.grid_size//2] = 2
        self.steps = 0
        self.historical_actions = []
        return self.forest, {}

    def step(self, actions):
        
        #if self.steps > 3:
        
        # Extract tuples from action (tuples are hashable into sets)
        #agent_actions = [(actions[i], actions[i+1]) for i in range(0,len(actions),2) ]
        # Ensure uniqueness and within bounds
        #cells = set(agent_actions) # Use a set to ensure uniqueness
        #cells = {c for c in cells if (0 <= c[0] < self.grid_size) and (0 <= c[1] < self.grid_size)}  # Filter invalid cells
        # Turn off the fire or create vacancies in the valid cells 
        #for x, y in cells:
        #    self.forest[x, y] = 0 # Extinguish cells
        #    self.historical_actions.append((x,y))   # Save the record
        x,y = divmod(actions,self.grid_size)
        #print(actions)
        self.forest[x,y] = 0
        
        # Run one time step after agent's action (Update environment)
        

        neighboursTensor = self.createNeighbourTensor()
        couldPropagate = (neighboursTensor == 2)
        amountOfBurningNeighbours = np.sum(couldPropagate, axis=0)
        cellsToEvaluate = amountOfBurningNeighbours > 0
        burningTrees = (self.forest == 2)
        probabilityMatrixForest = np.random.rand(*self.forest.shape)
        probabilityMatrixForest[cellsToEvaluate] = 1. - (1. - probabilityMatrixForest[cellsToEvaluate]) ** (1/amountOfBurningNeighbours[cellsToEvaluate])
        
        couldBurn = (probabilityMatrixForest <= self.threshold)
        # Find new burning trees
        newBurningTrees = (self.forest == 1) & couldBurn & np.logical_or.reduce(couldPropagate,axis=0)
        
        old_burning = np.sum(self.forest == 2)
        # Update forest matrix for the next step
        self.forest[burningTrees] = 3
        self.forest[newBurningTrees] = 2
        
        new_bunrning = np.sum(self.forest == 2)

        # Stablish reward criteria
        # Penalize repeated actions
        #repeated = np.sum([True for pair in cells if (pair in self.historical_actions)])
        # Penalize burned trees
        burned = np.sum(self.forest == 3)
        # Penalize propagation speed
        if new_bunrning == 0:
            propagation_speed = self.grid_size * self.grid_size
        else:
            propagation_speed = old_burning - new_bunrning            
            
        # Calculate reward  
        #print(propagation_speed)
        

        # Check if terminated
        self.steps += 1
        #print(self.steps)
        #terminated = ( (self.steps >= self.max_steps) or (np.sum(self.forest == 2) == 0) )
        terminated = (np.sum(self.forest == 2) == 0)
        truncated = False
        #reward = (-burned + 10*propagation_speed) #if terminated else 0
        reward = np.sum(self.forest == 0) - 0.2*np.sum(self.forest == 3)
    
        return self.forest, reward, terminated, truncated, {}

# Some auxiliar functions

# Method necessary for step method propagating time
    def createNeighbourTensor(self):
        neighborhoodSize = len(self.neighbours)
        tensor = np.zeros((neighborhoodSize, *self.forest.shape))
        for i, neigh in enumerate(self.neighbours):
            x,y = neigh
            tensor[i] = np.roll(self.forest, (-x,y), axis=(1,0))
            if x:
                if x == 1.:
                    tensor[i, : , -1 ] = 0
                elif x == -1.:
                    tensor[i, : , 0 ] = 0
                else:
                    continue # Maybe Another condition and method for second neighbours and more
            if y:
                if y == 1.:
                    tensor[i, 0 , : ] = 0
                elif y == -1.:
                    tensor[i, -1 , : ] = 0
                else:
                    continue # Maybe Another condition and method for second neighbours and more
            else:
                continue
        tensor = tensor * self.neighboursBoolTensor
        return tensor

# -------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Set Up the PPO Agent
    
    # Create the environment

    # Wrap the environment for vectorized training
    def make_env():
        '''
        Create a new instance of the environment for each of the 4 parallel environments
        '''
        return FirePropagationEnv(**parameters)

    vec_env = make_vec_env(make_env, n_envs=4)

    # Create the PPO agent
    model = PPO("MlpPolicy", vec_env, verbose=1, ent_coef=0.05, learning_rate=3e-4, clip_range=0.2)

    # Train the agent
    model.learn(total_timesteps=200000,log_interval=1)

    # Save the trained model
    model.save("./ppo_fire_agent")
