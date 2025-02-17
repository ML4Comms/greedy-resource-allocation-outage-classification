from enum import Enum
import tensorflow as tf
import numpy as np
from data_generator import OutageData
from dqn_lstm import DQNLSTM
from outage_loss import InfiniteOutageCoefficientLoss, FiniteOutageCoefficientLoss
import os


from data_generator import OutageData

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

def compute_loss():
    pass

def calibrate(data_input, model: TemperatureModel):
    temp = tf.Variable(initial_value=1.0, trainable=True, dtype=tf.float32) 
    training_generator = OutageData(**data_input,)
    for i in range(10000):
        print(i)
        X, y = training_generator.__getitem__(0)
        y_pred = model.predict(X)       
        
        def compute_loss():
            y_pred_model_w_temp = tf.math.divide(y_pred, temp)
            print("y", y)
            print("y_pred", y_pred)
            print("y_pred_model_w_temp",y_pred_model_w_temp)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\
                                y, y_pred_model_w_temp))
            print('Temperature value: {}'.format(temp.numpy()))
            return loss
        
        optimizer = tf.optimizers.Adam(learning_rate=0.0005)
        opts = optimizer.minimize(compute_loss, var_list=[temp])
    return temp

class ModelType(Enum):
    TemperatureModel = 1
    DQNLSTM = 2

def get_model_and_loss(data_input, 
                model_name, 
                qth: float,
                lstm_units: int = 32,
                model_type: ModelType = ModelType.TemperatureModel):

    if model_type == ModelType.TemperatureModel:
        model = TemperatureModel(output_size=data_input["output_size"], temp = 1, lstm_units=lstm_units)
    elif model_type == ModelType.DQNLSTM:
        model = DQNLSTM(qth, epochs=data_input["epoch_size"], data_config=data_input, model_name=model_name, lstm_units=lstm_units)
    else:
        raise Exception(f"Invalid model type: {model_type}")

    print(f"Training model: {model_name}")
    if "mse" in model_name:
        loss = tf.keras.losses.MeanSquaredError()
        model.compile(loss=loss,
                        optimizer=tf.keras.optimizers.Adam(),
                        metrics=[tf.keras.metrics.MeanAbsoluteError()])
    elif "binary_cross_entropy" in model_name:
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        model.compile(loss=loss,
                        optimizer=tf.keras.optimizers.Adam(),
                            metrics=[tf.keras.metrics.Recall(), 
                                    tf.keras.metrics.Precision(),
                                    tf.keras.metrics.BinaryAccuracy(),
                                    tf.keras.metrics.Accuracy()])
    elif "mae" in model_name:
        loss = tf.keras.losses.MeanAbsoluteError()
        model.compile(loss=loss,
                        optimizer=tf.keras.optimizers.Adam(),
                        metrics=[tf.keras.metrics.MeanAbsoluteError()])
    elif "inf_coef_loss" in model_name:
        loss = InfiniteOutageCoefficientLoss(qth= qth)
        model.compile(loss=loss,
                        optimizer=tf.keras.optimizers.Adam(),
                            metrics=[tf.keras.metrics.Recall(), 
                                    tf.keras.metrics.Precision(),
                                    tf.keras.metrics.BinaryAccuracy(),
                                    tf.keras.metrics.Accuracy()])
    elif "fin_coef_loss" in model_name:
        loss = FiniteOutageCoefficientLoss(S = data_input["batch_size"], qth = qth)
        model.compile(loss=loss,
                        optimizer=tf.keras.optimizers.Adam(),
                            metrics=[tf.keras.metrics.Recall(), 
                                    tf.keras.metrics.Precision(),
                                    tf.keras.metrics.BinaryAccuracy(),
                                    tf.keras.metrics.Accuracy()])
    elif "dummy" in model_name:
        return DummyModel()
    return model, loss
        

def get_fitted_model(data_input, 
                    model_name, 
                    qth: float,
                    epochs: int = 2, 
                    force_retrain: bool = False,
                    lstm_units: int = 32,
                    model_type: ModelType = ModelType.TemperatureModel):
    # TODO In relation to the above "weird" comment, this needs fixing here
    training_generator = OutageData(**data_input)
    path = f"models/{model_name}"
    if force_retrain or not os.path.exists(path):
        
        model, loss = get_model_and_loss(data_input, 
                                         model_name, 
                                         qth,
                                         lstm_units,
                                         model_type=model_type)

        class PrintQthCallback(tf.keras.callbacks.Callback):
            def __init__(self, loss):
                super().__init__()
                self.switched_direction_count = 0
                self.max_switched_direction_count = 3
                self.shift_size = 0.1
                self.direction_up = False
                self.loss = loss  # Store reference to the loss function

            def on_epoch_end(self, epoch, logs=None):
                print("\n\non_epoch_end\n\n")
                if hasattr(self.loss, 'conditional_avg_y_pred'):
                    print(f"\n\nEqth = {self.loss.conditional_avg_y_pred.numpy()}\n\n")
                if hasattr(self.loss, 'qth'):
                    print(f"\n\nPinf = {self.loss.conditional_avg_M.numpy()}\n\n")
                previous_qth = self.loss.qth.numpy()
                self.loss.adjust_qth(shift=self.shift_size)
                if self.loss.qth.numpy() > previous_qth:
                    # Shifted upwards
                    if self.direction_up:
                        # Continuing in same direction
                        # do nothing
                        pass
                    else:
                        # Direction changed
                        self.switched_direction_count += 1
                    self.direction_up = True
                else:
                    self.direction_up = False

                if self.switched_direction_count > self.max_switched_direction_count:
                    self.shift_size /= 2
                    self.switched_direction_count = 0
                        
                if hasattr(self.loss, 'qth'):
                    print(f"\n\nEpoch {epoch+1}: qth = {self.loss.qth.numpy()}\n\n")
                else:
                    print("\n\nno qth\n\n")

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='loss', patience=epochs, restore_best_weights=True),
            PrintQthCallback(loss)
        ]

        history = model.fit(training_generator, epochs=epochs, callbacks=callbacks)
        
        model.save(path)
        return model

    else:
        print(f"Loading model: {model_name}")
        model = tf.keras.models.load_model(path, compile = False)
        return model
