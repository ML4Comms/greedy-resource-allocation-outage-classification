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
from outage_loss import InfiniteOutageCoefficientLoss
from outage_loss import FiniteOutageCoefficientLoss

# TODO Make setting this less weird!
data_config = {}

class DQNLSTM:
    def __init__(self, qth:float,model_name=None, epochs=100,data_config= None,learning_rate=0.001,force_retrain: bool= True,lstm_units: int = 32):
        self.input_shape = (data_config["batch_size"], data_config["input_size"], 1)
        self.output_shape = (data_config["output_size"], 1)
        self.memory = []
        self.gamma = 0.9  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model_name = model_name
        self.epochs = epochs
        self.data_config = data_config
        self.qth = qth
        #self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.force_retrain = force_retrain
        self.lstm_units = lstm_units
        self.model = self.build_model()
        

    def build_model(self):
        model = Sequential()
        model.add(LSTM(self.lstm_units, input_shape=(self.input_shape[1], self.input_shape[2]),return_sequences= False))
        model.add(Dense(10, activation='PReLU'))
        model.add(Dense(1,activation='sigmoid'))  # Use the first element of output_shape as the number of units
        directory = f"models/{self.model_name}"
        filename = f"{directory}/model.keras"

        # Create the directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        if self.force_retrain or not os.path.exists(filename):
            if 'mse' in self.model_name:
                model.compile(loss=MeanSquaredError(), optimizer='adam',metrics=[tf.keras.metrics.MeanSquaredError()])
            elif 'binary_cross_entropy' in self.model_name:
                model.compile(loss=BinaryCrossentropy(from_logits=False), optimizer='adam',metrics=[tf.keras.metrics.Recall(), 
                                    tf.keras.metrics.Precision(),
                                    tf.keras.metrics.BinaryAccuracy()])
            elif 'mae' in self.model_name:
                model.compile(loss=MeanAbsoluteError(), optimizer='adam', metrics=[tf.keras.metrics.MeanAbsoluteError()])
            elif "fin_coef_loss" in self.model_name:
                model.compile(loss=FiniteOutageCoefficientLoss(S=self.data_config["batch_size"], qth=self.qth), optimizer='adam',metrics=[tf.keras.metrics.Recall(), 
                                    tf.keras.metrics.Precision(),
                                    tf.keras.metrics.BinaryAccuracy(),
                                    tf.keras.metrics.Accuracy()])
            else:
                raise ValueError(f"Invalid loss name: {self.model_name}")
            print(f"Training model: {self.model_name}")
            training_generator = OutageData(**self.data_config)
            callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=self.epochs, restore_best_weights=True)
            history = model.fit(training_generator, epochs=self.epochs, callbacks=[callback])
            return model
        else:
            print(f"Loading model: {self.model_name}")
            model = tf.keras.models.load_model(filename, compile = False)
            model.save(filename)
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

    def predict(self, *args, **kwargs):
        # Directly call the predict method of the encapsulated Keras model
        return self.model.predict(*args, **kwargs)

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
    @staticmethod
    def calculate_binary_nll(y_true, y_pred_probs, epsilon=1e-5):
        """Calculate Binary Negative Log-Likelihood (NLL)."""
        y_pred_probs = tf.clip_by_value(y_pred_probs, epsilon, 1 - epsilon)
        return -tf.reduce_mean(y_true * tf.math.log(y_pred_probs) + (1 - y_true) * tf.math.log(1 - y_pred_probs))

    def calibrate(self, data_input, method='platt', nll_function=calculate_binary_nll):
        temp = tf.Variable(initial_value=1.0, trainable=True, dtype=tf.float32)
        training_generator = OutageData(**data_input)
        optimizer = tf.optimizers.Adam(learning_rate=0.0005)
        
        if method == 'temp':
            def compute_temp_loss(y_pred, y):
                y_pred_model_w_temp = y_pred / temp
                loss = nll_function(tf.convert_to_tensor(y, dtype=tf.float32), tf.convert_to_tensor(y_pred_model_w_temp, dtype=tf.float32))
                return loss

            for i in range(10000):
                X, y = training_generator.__getitem__(0)
                y_pred = self.model.predict(X)

                with tf.GradientTape() as tape:
                    loss = compute_temp_loss(y_pred, y)
                grads = tape.gradient(loss, [temp])
                optimizer.apply_gradients(zip(grads, [temp]))

                if i % 1000 == 0:
                    print(f"Iteration {i}, Temperature: {temp.numpy()}, Loss: {loss.numpy()}")

            self.temp = temp.numpy()
            return self.temp

        elif method == 'platt':
            all_y_true = []
            all_y_pred = []
            for _ in range(10000):
                X, y = training_generator.__getitem__(0)
                y_pred = self.model.predict(X)
                all_y_true.extend(y)
                all_y_pred.extend(y_pred)

            lr = LogisticRegression()
            lr.fit(np.array(all_y_pred).reshape(-1, 1), np.array(all_y_true).reshape(-1))
            self.A = lr.coef_[0][0]
            self.B = lr.intercept_[0]
            return self.A, self.B

        elif method == 'beta':
            def nll_loss(params):
                a, b, c = params
                scaled_preds = 1 / (1 + np.exp(-a * np.log(np.array(all_y_pred)) - b * np.log(1 - np.array(all_y_pred)) - c))
                return nll_function(tf.convert_to_tensor(np.array(all_y_true), dtype=tf.float32), tf.convert_to_tensor(scaled_preds, dtype=tf.float32))

            all_y_true = []
            all_y_pred = []
            for _ in range(10000):
                X, y = training_generator.__getitem__(0)
                y_pred = self.model.predict(X)
                all_y_true.extend(y)
                all_y_pred.extend(y_pred)

            result = minimize(nll_loss, x0=[1.0, 1.0, 0.0], bounds=[(0.01, 10), (0.01, 10), (-10, 10)])
            self.a, self.b, self.c = result.x
            return self.a, self.b, self.c
        
        elif method == 'isotonic':
            all_y_true = []
            all_y_pred = []
            for _ in range(10000):
                X, y = training_generator.__getitem__(0)
                y_pred = self.model.predict(X)
                all_y_true.extend(y)
                all_y_pred.extend(y_pred)

            # Ensure correct shapes
            all_y_true = np.array(all_y_true).flatten()
            all_y_pred = np.array(all_y_pred).flatten()
            
            print(f"Length of all_y_true: {len(all_y_true)}")
            print(f"Length of all_y_pred: {len(all_y_pred)}")
            
            if len(all_y_true) != len(all_y_pred):
                raise ValueError("The lengths of all_y_true and all_y_pred do not match.")
            
            # Remove NaN or infinite values
            valid_mask = np.isfinite(all_y_true) & np.isfinite(all_y_pred)
            all_y_true = all_y_true[valid_mask]
            all_y_pred = all_y_pred[valid_mask]
            
            ir = IsotonicRegression(out_of_bounds='clip')
            self.isotonic_regressor = ir.fit(all_y_pred, all_y_true)
            return self.isotonic_regressor

        else:
            raise ValueError(f"Invalid calibration method: {method}")
    def isotonic_predict(self, y_pred):
        if self.isotonic_regressor is None:
            raise ValueError("Isotonic regressor is not trained. Please run calibrate method with 'isotonic' option first.")
        return self.isotonic_regressor.transform(y_pred)
