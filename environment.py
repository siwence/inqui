import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import os
import torch
import time
import torch.nn as nn
from cube_controller import DQN, start_model
from training_file import train

os.environ["NUMEXPR_MAX_THREADS"] = "1"

neg_inf = float("-inf")
pos_inf = float("inf")

class Environment(gym.Env):
    
    def __init__(self):
        super(Environment, self).__init__()

        # action and observation spaces
        self.action_space = gym.spaces.Discrete(5)  # 5 for amount of movement directions
        self.observation_space = gym.spaces.Box(low=neg_inf, high=pos_inf, shape=(3,), dtype=np.float32)

        # loading pybullet
        p.connect(p.GUI)
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # loading plane and cube (ai)
        self.plane_id = p.loadURDF("plane.urdf") 
        self.cube = p.loadURDF("cube.urdf", basePosition=[0, 0, 0.05])

        # pressure plate location
        self.plate_pos = [-2, 0, 0.1]
        self.plate = p.loadURDF("cube.urdf", basePosition=self.plate_pos, globalScaling=0.5, useFixedBase = True) # makes it so that pressure plate cannot move

        self.start_time = None # start time

        self.done = False 
        
    # reseting the simulation
    def reset(self):
        p.resetSimulation()

        # readding everything each reset
        p.setGravity(0, 0, -9.81)
        self.plane_id = p.loadURDF("plane.urdf") 
        self.cube = p.loadURDF("cube.urdf", basePosition=[0, 0, 0.5], globalScaling=0.5)
        self.plate_pos = [-2, 0, 0.1]
        self.plate = p.loadURDF("cube.urdf", basePosition=self.plate_pos, globalScaling=0.5, useFixedBase = True)

        self.start_time = time.time() # timer starts

        self.done = False

        return self.get_observation()
    
    def step(self, action): # xyz grid
        if action == 0: # left
            p.resetBaseVelocity(self.cube, linearVelocity=[-1, 0, 0])

        elif action == 1: # right
            p.resetBaseVelocity(self.cube, linearVelocity=[1, 0, 0]) 
        
        elif action == 2: # forward
            p.resetBaseVelocity(self.cube, linearVelocity = [0, 1, 0])

        elif action == 3: # backward
            p.resetBaseVelocity(self.cube, linearVelocity = [0, -1, 0])

        elif action == 4: # jump - applies force upwards (z axis) causing a jump
            p.applyExternalForce(self.cube, -1, [0, 0, 10], self.get_observation(), p.WORLD_FRAME)

        # updates the simulation based on what action was chosen above (adding / subtracting that amount to the corresponding value)
        p.stepSimulation() 

        cube_pos = self.get_observation() # returns xyz coords of position
        plate_pos = np.array(self.plate_pos)

        distance = np.linalg.norm(cube_pos - plate_pos) # distance formula

        reward = -distance

        if self.on_pressure_plate(np.array(cube_pos), plate_pos):
            reward = 1000 # positive reinforcement
            self.done = True # done with the test

        elapsed_time = time.time() - self.start_time

        if elapsed_time > 5:
            reward = -10  # penalty for failing
            self.done = True  # End the episode

        return cube_pos, reward, self.done
    
    
    # distance formula to check if it's on the pressure plate
    def on_pressure_plate(self, cube_pos, plate_pos, threshold=0.25):
        distance_squared = (cube_pos[0] - plate_pos[0]) ** 2 + (cube_pos[1] - plate_pos[1]) ** 2
        touch = true
        return distance_squared < threshold ** 2
    
    def get_observation(self):
        # Get the position of the cube
        cube_pos = p.getBasePositionAndOrientation(self.cube)[0]  # Get position
        return np.array(cube_pos, dtype=np.float32)
    
    def end(self):
        p.disconnect()

if __name__ == "__main__":
    env = Environment()
    model, optimizer = start_model() # starts the model and adam's optimizer
    criterion = nn.MSELoss()  # Loss function for training
    observation = env.reset()

    train(env, model, optimizer, criterion)
    env.end()  # Ensure the environment is properly closed
