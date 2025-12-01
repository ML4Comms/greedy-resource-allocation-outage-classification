import tensorflow as tf
import numpy as np
from data_generator import OutageData
from outage_loss import InfiniteOutageCoefficientLoss, FiniteOutageCoefficientLoss
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,PReLU
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy, MeanAbsoluteError
from tensorflow.keras.callbacks import EarlyStopping
from outage_loss import InfiniteOutageCoefficientLoss, FiniteOutageCoefficientLoss
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize
from scipy.special import logit
from betacal import BetaCalibration


class DummyModel():
    def predict(self, X):
        return np.zeros(len(X))

# TODO Make setting this less weird!
data_config = {}

class TemperatureModel(tf.keras.Model):
    def __init__(self, lstm_units, temp, output_size):
        super().__init__()
        self.layer1 = tf.keras.layers.LSTM(lstm_units, return_sequences=False)
        self.layer2 = tf.keras.layers.Dense(output_size,
                                kernel_initializer=tf.initializers.HeUniform(), 
                                activation=tf.keras.layers.PReLU())
        self.layer3 = tf.keras.layers.Dense(1,
                                kernel_initializer=tf.initializers.HeUniform(), 
                                activation=tf.keras.activations.sigmoid)                            
    
    def call(self, inputs):
        x = self.layer3(self.layer2(self.layer1(inputs)))
        return x
    
    def calibrate(self, data_input, method='platt'):
        temp = tf.Variable(initial_value=1.0, trainable=True, dtype=tf.float32)
        training_generator = OutageData(**data_input)
        optimizer = tf.optimizers.Adam(learning_rate=0.0005)
        
        if method == 'temp':
            def compute_temp_loss(y_pred, y_true):
                y_pred_logits = logit(y_pred)
                scaled_logits = y_pred_logits / temp
                # Compute cross-entropy loss
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=scaled_logits))
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
            for _ in range(11000):
                X, y = training_generator.__getitem__(0)
                y_pred = self.predict(X)
                all_y_true.extend(y)
                all_y_pred.extend(y_pred)
                
            all_y_true = np.array(all_y_true).flatten()  # Flattening to ensure correct shape
            all_y_pred = np.array(all_y_pred).flatten()  # Flattening to ensure correct shape
            
            epsilon = 1e-7  # Safe threshold for float64
            all_y_pred = np.clip(all_y_pred, epsilon, 1 - epsilon)  # Clip probabilities
            
            # Logistic regression for Platt scaling is applied on logits, not probabilities
            logits = np.log(all_y_pred / (1 - all_y_pred))  # Convert probabilities to logits
            lr = LogisticRegression()
            lr.fit(logits.reshape(-1, 1), all_y_true)  # Fit on logits, not probabilities
            # Assign coefficients
            self.A = lr.coef_[0][0]  # Slope
            self.B = lr.intercept_[0]  # Intercept
            print(f"Fitted Platt scaling coefficients: A = {self.A}, B = {self.B}")
            return self.A, self.B

        elif method == 'beta':
            all_y_true = []
            all_y_pred = []
            for _ in range(9000):
                X, y = training_generator.__getitem__(0)
                y_pred = self.predict(X)
                all_y_true.extend(y)
                all_y_pred.extend(y_pred)

            beta_calibrator = BetaCalibration(parameters="abm")
            beta_calibrator.fit(all_y_pred, all_y_true)
            self.beta_calibrator = beta_calibrator
            return self.beta_calibrator
        
        elif method == 'isotonic':
            all_y_true = []
            all_y_pred = []
            for _ in range(11000):
                X, y = training_generator.__getitem__(0)
                y_pred = self.predict(X)
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
            
            # Clip to safe numeric range
            #epsilon = 1e-6
            #all_y_pred = np.clip(all_y_pred, epsilon, 1 - epsilon)
            
            #all_y_pred = np.concatenate([all_y_pred, [1e-15, 1e-15]])
            #all_y_true = np.concatenate([all_y_true, [1, 0]])

    
            ir = IsotonicRegression(out_of_bounds='clip')
            self.isotonic_regressor = ir.fit(all_y_pred, all_y_true)
           
            return self.isotonic_regressor

        else:
            raise ValueError(f"Invalid calibration method: {method}")
            
            
    def isotonic_predict(self, y_pred):
        if self.isotonic_regressor is None:
            raise ValueError("Isotonic regressor is not trained. Please run calibrate method with 'isotonic' option first.")
        return self.isotonic_regressor.transform(y_pred) 

def compute_loss():
    pass

def get_model(data_input, 
                model_name, 
                qth: float,
                lstm_units: int = 32):


    model = TemperatureModel(output_size=data_input["output_size"], temp = 1, lstm_units=lstm_units)
    print(f"Training model: {model_name}")
    if "mse" in model_name:
        model.compile(loss=tf.keras.losses.MeanSquaredError(),
                        optimizer=tf.keras.optimizers.Adam(),
                        metrics=[tf.keras.metrics.MeanAbsoluteError()])
    elif "bce" in model_name:
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                        optimizer=tf.keras.optimizers.Adam(),
                            metrics=[tf.keras.metrics.Recall(), 
                                    tf.keras.metrics.Precision(),
                                    tf.keras.metrics.BinaryAccuracy(),
                                    tf.keras.metrics.Accuracy()])
    elif "mae" in model_name:
        model.compile(loss=tf.keras.losses.MeanAbsoluteError(),
                        optimizer=tf.keras.optimizers.Adam(),
                        metrics=[tf.keras.metrics.MeanAbsoluteError()])
    elif "inf_coef_loss" in model_name:
        model.compile(loss=InfiniteOutageCoefficientLoss(qth= qth),
                        optimizer=tf.keras.optimizers.Adam(),
                            metrics=[tf.keras.metrics.Recall(), 
                                    tf.keras.metrics.Precision(),
                                    tf.keras.metrics.BinaryAccuracy(),
                                    tf.keras.metrics.Accuracy()])
    elif "olf" in model_name:
        model.compile(loss=FiniteOutageCoefficientLoss(S = data_input["batch_size"], qth = qth),
                        optimizer=tf.keras.optimizers.Adam(),
                            metrics=[tf.keras.metrics.Recall(), 
                                    tf.keras.metrics.Precision(),
                                    tf.keras.metrics.BinaryAccuracy(),
                                    tf.keras.metrics.Accuracy()])
    elif "dummy" in model_name:
        return DummyModel()
    return model
        

def get_fitted_model(data_input, 
                    model_name, 
                    qth: float,
                    epochs: int = 2, 
                    force_retrain: bool = False,
                    lstm_units: int = 32):
    # TODO In relation to the above "weird" comment, this needs fixing here
    training_generator = OutageData(**data_input)
    directory = f"models/{model_name}"
    filename = f"{directory}/model.keras"

    # Create directory if missing
    os.makedirs(directory, exist_ok=True)

    if force_retrain or not os.path.exists(filename):
        
        model = get_model(data_input, 
                    model_name, 
                    qth,
                    lstm_units)

        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=epochs, restore_best_weights=True)
        history = model.fit(training_generator, epochs=epochs,
                            callbacks=[callback])
        model.save(filename)
        return model

    else:
        print(f"Loading model: {model_name}")
        model = tf.keras.models.load_model(filename, compile = False)
        return model