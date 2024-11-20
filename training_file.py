import torch
import random
from cube_controller import DQN, ReplayBuffer 

def select_action(state, model, epsilon):
    if random.random() < epsilon:  # exploring
        return random.randint(0, 4)  
    else:  # "best" action
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = model(state_tensor)
            return torch.argmax(q_values).item()

def train(env, model, optimizer, criterion, episodes=1000, batch_size=32):
    replay_buffer = ReplayBuffer(max_size=10000) # replay system with a max size of 10k
    epsilon = 1.0  # explores the environment fully through random actions
    epsilon_decay = 0.995 # reduces likelihood of taking random actions
    min_epsilon = 0.01 # ensuring that 1% of the time it will still be exploring
    gamma = 0.99  # values future rewards just as much as immediate rewards 

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = select_action(state, model, epsilon)  # calling the select_action function
            next_state, reward, done = env.step(action) # receives the new state, the reward (how good or bad), and if it's done
            total_reward += reward

            replay_buffer.add((state, action, reward, next_state, done)) # adds state of the env, actions it took, the reward after the action, the state of env after actions, and if done all to replay buffer
            state = next_state # moves to the next state

            # training the model if there are enough samples
            if len(replay_buffer) > batch_size:
                experiences = replay_buffer.sample(batch_size) # takes batch_size amount of experiences randomly from replay_buffer to free up space
                states, actions, rewards, next_states, dones = zip(*experiences) # unpack sampled experiences

                # converting the zipped experiences into tensors for neural network
                states_tensor = torch.FloatTensor(states)
                actions_tensor = torch.LongTensor(actions)
                rewards_tensor = torch.FloatTensor(rewards)
                next_states_tensor = torch.FloatTensor(next_states)
                dones_tensor = torch.FloatTensor(dones)

                # calculate q-values for actions taken in current states 
                q_values = model(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
                # model(states_tensor) sends the current states to neural network, and outputs a set of q-values for each possible action
                # select the q-values corresponding to the actions that the agent actually took
        
                # calculation q-values for the next state
                next_q_values = model(next_states_tensor).max(1)[0]
                # model(next_states_tensor) read above comments. same thing but with next_states_tensor this time. 
                # .max(1)[0] finds the max q-values in each state

                # calculating the target q-values
                target_q_values = rewards_tensor + (gamma * next_q_values * (1 - dones_tensor))

                # calculating the loss between predicted and target q-values
                loss = criterion(q_values, target_q_values.detach())

                optimizer.zero_grad() # reset gradients 
                loss.backward() # backpropagation math thingy idk and don't understand it really    
                optimizer.step() # updates parameters based on gradients

        # decay epsilon
        if epsilon > min_epsilon:
            epsilon *= epsilon_decay
