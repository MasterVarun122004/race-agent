import gymnasium
from gymnasium import spaces
import numpy as np


class RacingEnv(gymnasium.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=np.array([0,-1,0,0,0]),high=np.array([1,1,1,1,1]),shape=(5,),dtype=np.float32)
        self.action_space = spaces.Discrete(5)
        
    def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.speed = 0
            self.heading_error = 0
            self.lb = 0.5
            self.rb = 0.5
            self.progress = 0
            observation = np.array([self.speed,self.heading_error,self.lb,self.rb,self.progress],dtype=np.float32)
            info = {}
            return observation,info
        
    def step(self,action):
        reward = 0
        info ={}
        terminated = False
        truncated = False
        if(action==1):
            self.speed = min(1,self.speed+0.1)
            
        if(action ==2):
            self.speed = max(0,self.speed-0.1)
            
        if(action == 3):
            self.heading_error = max(-1,self.heading_error-0.1)
            
            
        if(action == 4):
            self.heading_error = min(1,self.heading_error+0.1)
           
        lateral_shift = self.speed * self.heading_error * 0.05
        
        self.lb +=lateral_shift
        self.rb-= lateral_shift
          
        if(self.lb <=1e-6 or self.rb<=1e-6):
            reward -= 10
            terminated = True
            info["mistake"]="Track limit exceeded"
        else:
            reward +=0.1*self.speed
            
        self.progress += self.speed*0.02
        
        if(min(self.progress,1) == 1):
            self.progress = 1
            info["completed"] = "Lap completed"
            terminated = True
            reward+=10
        
            
        
        observation = np.array([self.speed,self.heading_error,self.lb,self.rb,self.progress],dtype=np.float32)
        
        return observation, reward, terminated, truncated, info
    
        