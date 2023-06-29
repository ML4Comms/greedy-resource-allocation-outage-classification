#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import json
from data_generator import OutageData
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
#from outage_loss import InfiniteOutageCoefficientLoss
#from outage_loss import FiniteOutageCoefficientLoss



class DQNLSTM:
    def __init__(self, qth:float,loss_name=None, epochs=100,data_config= None,batch_size=32,force_retrain: bool= False):
        self.input_shape = (data_config["resources"], data_config["input_size"], 1)
        self.output_shape = (data_config["output_size"], 1)
        self.memory = []
        self.gamma = 0.9  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.loss_name = loss_name
        self.epochs = epochs
        self.data_config = data_config
        self.qth = qth
        self.batch_size = batch_size
        self.force_retrain = force_retrain
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(LSTM(32, input_shape=(self.input_shape[1], self.input_shape[2]),return_sequences= False))
        model.add(Dense(10, activation='PReLU'))
        model.add(Dense(2,activation='sigmoid'))  # Use the first element of output_shape as the number of units
        path = f"models/{self.loss_name}"
        if self.force_retrain or not os.path.exists(path):
            if self.loss_name == 'mse':
                model.compile(loss=MeanSquaredError(), optimizer='adam',metrics=[tf.keras.metrics.MeanSquaredError()])
            elif self.loss_name == 'bce':
                model.compile(loss=BinaryCrossentropy(from_logits=False), optimizer='adam',metrics=[tf.keras.metrics.Recall(), 
                                    tf.keras.metrics.Precision(),
                                    tf.keras.metrics.BinaryAccuracy()])
            elif self.loss_name == 'mae':
                model.compile(loss=MeanAbsoluteError(), optimizer='adam', metrics=[tf.keras.metrics.MeanAbsoluteError()])
            elif self.loss_name == "fin_coef_loss":
                model.compile(loss=FiniteOutageCoefficientLoss(S=self.data_config["resources"], qth=self.qth), optimizer='adam',metrics=[tf.keras.metrics.Recall(), 
                                    tf.keras.metrics.Precision(),
                                    tf.keras.metrics.BinaryAccuracy(),
                                    tf.keras.metrics.Accuracy()])
            else:
                raise ValueError(f"Invalid loss name: {self.loss_name}")
            print(f"Training model: {self.loss_name}")
            training_generator = OutageData(**self.data_config)
            callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=self.epochs, restore_best_weights=True)
            history = model.fit(training_generator, epochs=self.epochs, callbacks=[callback])
            return model
        else:
            print(f"Loading model: {self.loss_name}")
            model = tf.keras.models.load_model(path, compile = False)
            return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.uniform()  # Choose a random action
        next_state_pred = self.model.predict(state)  # Predict the value of the next state
        return next_state_pred


    def replay(self, batch_size):
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        for index in batch:
            state, action, reward, next_state, done = self.memory[index]
            target = reward if done else reward + self.gamma * np.amax(self.model.predict(next_state))
            target_full = self.model.predict(state)
            target_full[0] = target
            self.model.fit(state, target_full, epochs=1, verbose=1)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, X, y_label, num_episodes=1000, batch_size=32):
        rewards = []
        if self.force_retrain:
            self.model = self.build_model()  # Rebuild the model from scratch
        for episode in range(num_episodes):
            for i in range(len(X) - self.data_config["input_size"] - self.data_config["output_size"]):  # Iterate over the range taking into account the state and next state sizes
                state = X[i:i+self.data_config["input_size"]]  # Get a sequence of 100 data points as the state
                next_state = y_label[i+self.data_config["input_size"]:i+self.data_config["input_size"]+self.data_config["output_size"]]  # Get the next 10 data points as the next state
                total_reward = 0
                done = False

                while not done:
                    next_state_pred = self.act(state)
                    reward = 1 if next_state_pred == np.argmax(next_state) else -1
                    total_reward += reward
                    self.remember(state, next_state_pred, reward, next_state, done)  # Store the predicted next state
                    state = next_state
                    if done:
                        break
                    if len(self.memory) > batch_size:
                        self.replay(batch_size)


                rewards.append(total_reward)
                print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

        return rewards


# In[ ]:




