import gym
import numpy as np

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.losses import CategoricalCrossentropy

#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

tf.get_logger().setLevel('ERROR')

class REINFORCE:
  def __init__(self, env, path=None):
    self.env=env 
    self.state_shape=env.observation_space.shape # the state space
    self.action_shape=env.action_space.n # the action space
    self.gamma=0.99 # decay rate of past observations
    self.alpha=1e-4 # learning rate of gradient
    self.learning_rate=0.01 # learning of deep learning model
    
    if not path:
      self.model=self.build_policy_network() #build model
    else:
      self.model=self.load_model(path) #import model

    # record observations
    self.states=[]
    self.gradients=[] 
    self.rewards=[]
    self.probs=[]
    self.discounted_rewards=[]
    self.total_rewards=[]
    
    

  def build_policy_network(self):
    model=Sequential()
    model.add(Dense(24, input_shape=self.state_shape, activation="relu"))
    model.add(Dense(12, activation="relu"))
    model.add(Dense(self.action_shape, activation="softmax")) 
    model.compile(loss="categorical_crossentropy",
            optimizer=Adam(lr=self.learning_rate))
        
    return model

  def hot_encode_action(self, action):

    action_encoded=np.zeros(self.action_shape)
    action_encoded[action]=1

    return action_encoded
  
  def remember(self, state, action, action_prob, reward):
    encoded_action=self.hot_encode_action(action)
    self.gradients.append(encoded_action-action_prob)
    self.states.append(state)
    self.rewards.append(reward)
    self.probs.append(action_prob)


  def compute_action(self, state):

    # transform state
    state=state.reshape([1, state.shape[0]])
    # get action probably
    action_probability_distribution=self.model.predict(state).flatten()
    # norm action probability distribution
    action_probability_distribution/=np.sum(action_probability_distribution)
    
    # sample action
    action=np.random.choice(self.action_shape,1,
                            p=action_probability_distribution)[0]

    return action, action_probability_distribution


  def get_discounted_rewards(self, rewards): 
   
    discounted_rewards=[]
    cumulative_total_return=0
    # iterate the rewards backwards and calc the total return 
    for reward in rewards[::-1]:      
      cumulative_total_return=(cumulative_total_return*self.gamma)+reward
      discounted_rewards.insert(0, cumulative_total_return)

    # normalize discounted rewards
    mean_rewards=np.mean(discounted_rewards)
    std_rewards=np.std(discounted_rewards)
    norm_discounted_rewards=(discounted_rewards-
                          mean_rewards)/(std_rewards+1e-7) # avoiding zero div
    
    return norm_discounted_rewards

  def train_policy_network(self):
       
    # get X_train
    states=np.vstack(self.states)

    # get y_train
    gradients=np.vstack(self.gradients)
    rewards=np.vstack(self.rewards)
    discounted_rewards=self.get_discounted_rewards(rewards)
    gradients*=discounted_rewards
    y_train = self.alpha*np.vstack([gradients])+self.probs
    #y_train = gradients
    history=self.model.train_on_batch(states, y_train)
    
    self.states, self.probs, self.gradients, self.rewards=[], [], [], []

    return history


  def train(self, episodes):
     
    env=self.env
    total_rewards=np.zeros(episodes)

    for episode in range(episodes):
      # each episode is a new game env
      state=env.reset()
      done=False          
      episode_reward=0 #record episode reward
      
      while not done:
        # play an action and record the game state & reward per episode
        action, prob=self.compute_action(state)
        next_state, reward, done, _=env.step(action)
        self.remember(state, action, prob, reward)
        state=next_state
        episode_reward+=reward

        #if episode%render_n==0: ## render env to visualize.
        env.render()
        if done:
          # update policy 
            history=self.train_policy_network()

      total_rewards[episode]=episode_reward
      print('\n episode = ',episode, 'reward = ',episode_reward)
      
    self.total_rewards=total_rewards

  def hot_encode_action(self, action):

    action_encoded=np.zeros(self.action_shape)
    action_encoded[action]=1

    return action_encoded
  
  def remember(self, state, action, action_prob, reward):

    encoded_action=self.hot_encode_action(action)
    self.gradients.append(encoded_action-action_prob)
    self.states.append(state)
    self.rewards.append(reward)
    self.probs.append(action_prob)


ENV="CartPole-v1"

N_EPISODES=500



# set the env
env=gym.make(ENV) # env to import
env.reset() # reset to env 

Agent = REINFORCE(env)

Agent.train(N_EPISODES)


