#Demendencies
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

#Start the environment
env = gym.make('CartPole-v1')


epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
#Policy
def play_action(model, state):
    #If random number is less than our epsilon parameter play random action
    if np.random.rand() <= epsilon:
        return random.randrange(number_of_actions)
    else:
        #Get action values from the model
        action_values = model.predict(state)
        #Return the action with the highest value for the current state
        return np.argmax(action_values[0])


gamma = 0.95
#Train the model
def fit(model, memory, epsilon):
    #Sample from the memory
    batch = random.sample(memory, batch_size)
    
    #Iterate through the batch from the memory
    for state, action, reward, next_state, done in batch:
        target = reward
        if not done:
            #Apply reward discount
            target = reward + gamma * np.amax(model.predict(next_state)[0])
        #Get the target for the current state
        state_target = model.predict(state)
        #Set the target for the current action to be discounted reward
        state_target[0][action] = target
        #Train the model for the current state and target
        model.fit(state, state_target, epochs=1, verbose=0)
    
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    return model, epsilon

#Get environment data:
state_size = env.observation_space.shape[0]
number_of_actions = env.action_space.n

#Build the model
model = Sequential()
model.add(Dense(24, input_dim=state_size, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(number_of_actions, activation='linear'))

#Compile the model
model.compile(loss='mse', optimizer='adam')

#Experience replay
memory = deque(maxlen=2000)
number_of_episodes = 1000
batch_size = 32

for e in range(number_of_episodes):
    state = env.reset()
    #Reshaping the state because of the Keras model
    state = np.reshape(state, [1, state_size]) 

    #Run the game for 500 steps <- this can be even smaller
    for time_step in range(500):
        #env.render will visualize the gameplay process
        env.render()
        #Choose an action from policy
        action = play_action(model, state)

        #Play the action in the gym and get next_state, reward, and is it done or not boolean
        next_state, reward, done, _ = env.step(action)

        if done:
            #If done penalize the model
            reward = -15 
        else:
            reward = reward

        next_state = np.reshape(next_state, [1, state_size])
        #Adding knowledge of the environment to the memory - experience replay
        memory.append((state, action, reward, next_state, done))

        state = next_state

        if done:
            print("Number of points: ", time_step)
            break
        if len(memory) > batch_size:
            model, epsilon = fit(model, memory, epsilon)