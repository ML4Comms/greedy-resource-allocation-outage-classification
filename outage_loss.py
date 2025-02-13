import tensorflow as tf
from data_generator import OutageData

# Define global qth and epoch tracking variables
global_qth = tf.Variable(0.5, trainable=False, dtype=tf.float32)

# Variables for averaging per epoch
p_infty_epoch_sum = tf.Variable(0.0, dtype=tf.float32, trainable=False)
E_Q_less_qth_epoch_sum = tf.Variable(0.0, dtype=tf.float32, trainable=False)
batch_count = tf.Variable(0, dtype=tf.int32, trainable=False)

p_infty_avg = tf.Variable(0.0, dtype=tf.float32, trainable=False)  # Running average of P_infty
E_Q_less_qth_avg = tf.Variable(0.0, dtype=tf.float32, trainable=False)  # Running average of E[Q | Q < qth]
total_epochs = tf.Variable(0, dtype=tf.int32, trainable=False)  # Total number of completed epochs

# Lists to store values per epoch
p_infty_per_epoch = []
E_Q_less_qth_per_epoch = []
qth_per_epoch = []  # Track qth per epoch

def heaviside(x):
    return tf.experimental.numpy.heaviside(x,0)

def TPR(y_true, y_pred, step_function = heaviside):
    result = generalise_TP(y_true, y_pred, step_function) / (generalise_TP(y_true, y_pred, step_function) + generalise_FN(y_true, y_pred, step_function))
    if tf.math.is_nan(result):
        return tf.constant(0)
    else:
        return result

def FPR(y_true, y_pred, step_function = heaviside):
    result = generalise_FP( y_true, y_pred, step_function) / (generalise_FP(y_true, y_pred, step_function) + generalise_TN(y_true, y_pred, step_function))
    if tf.math.is_nan(result):
        return tf.constant(0)
    else:
        return result
    
def Precision(y_true, y_pred, step_function = heaviside):
    result = generalise_TP(y_true, y_pred, step_function) / (generalise_TP(y_true, y_pred, step_function) + generalise_FP(y_true, y_pred, step_function))
    if tf.math.is_nan(result):
        return tf.constant(0)
    else:
        return result

def generalise_TN(y_true, y_pred, step_function = heaviside):
    return tf.reduce_sum(tf.multiply(step_function(global_qth - y_pred), 1.0 - y_true))
def generalise_FN(y_true, y_pred, step_function = heaviside):
    return tf.reduce_sum(tf.multiply(step_function(global_qth - y_pred), y_true))
def generalise_TP(y_true, y_pred, step_function = heaviside):
    return tf.reduce_sum(tf.multiply(step_function(y_pred - global_qth), y_true))
def generalise_FP(y_true, y_pred, step_function = heaviside):
    return tf.reduce_sum(tf.multiply(step_function(y_pred - global_qth), 1.0 - y_true))


class MeanSquaredErrorTemperature(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name=None):
        super().__init__(reduction, name)

    def call(self, y_true, y_pred):
        return self.M(y_true, y_pred)

class InfiniteOutageCoefficientLoss(tf.keras.losses.Loss):
    def __init__(self, data_config, step_size=0.01, threshold=1e-4, reduction=tf.keras.losses.Reduction.AUTO, name=None):
        #self.qth = qth
        super().__init__(reduction, name)
        self.data_config = data_config
        self.step_size = step_size  # Step size for updating qth
        self.threshold = threshold  # Threshold for updating qth

    def squashed_sigmoid(self, input, factor: float = 5.0):
        return tf.divide(1, 1 + tf.math.exp(tf.multiply(-factor, input)))

    def TN(self, y_true, y_pred):
        return generalise_TN(y_true=y_true, y_pred=y_pred, step_function=self.squashed_sigmoid)
    def FN(self, y_true, y_pred):
        return generalise_FN(y_true=y_true, y_pred=y_pred, step_function=self.squashed_sigmoid)
    def TP(self, y_true, y_pred):
        return generalise_TP(y_true=y_true, y_pred=y_pred, step_function=self.squashed_sigmoid)
    def FP(self, y_true, y_pred):
        return generalise_FP(y_true=y_true, y_pred=y_pred, step_function=self.squashed_sigmoid)

    def M(self, y_true, y_pred):
        epsilon = 0.0001
        numerator = tf.multiply(self.FN(y_true, y_pred), self.TN(y_true, y_pred) + self.FN(y_true, y_pred) + self.TP(y_true, y_pred) + self.FP(y_true, y_pred))
        denominator =  epsilon + tf.multiply(self.TN(y_true, y_pred) + self.FN(y_true, y_pred), self.FN(y_true, y_pred) + self.TP(y_true, y_pred))
        return tf.divide(numerator, denominator)

    def call(self, y_true, y_pred):
        global global_qth, p_infty_epoch_sum, E_Q_less_qth_epoch_sum, batch_count
        global p_infty_avg, E_Q_less_qth_avg, total_epochs

        # Compute P_infty for the current batch
        P_infty_estimate = self.M(y_true, y_pred)

        # Compute E[Q | Q < qth] for the current batch
        mask = y_pred < global_qth
        valid_values = tf.boolean_mask(y_pred, mask)
        E_Q_less_qth = tf.reduce_mean(valid_values) if tf.size(valid_values) > 0 else 0.0  

        # Update accumulators for averaging per epoch
        p_infty_epoch_sum.assign_add(P_infty_estimate)
        E_Q_less_qth_epoch_sum.assign_add(E_Q_less_qth)
        batch_count.assign_add(1)

        # Check if it's the last batch of the epoch
        if tf.equal(tf.math.mod(batch_count, self.data_config["epoch_size"]), 0):
            total_epochs.assign_add(1)  # Correct way to increment tf.Variable


            # Compute the per-epoch averages
            p_infty_epoch_avg = p_infty_epoch_sum / tf.cast(self.data_config["epoch_size"], tf.float32)
            E_Q_less_qth_epoch_avg = E_Q_less_qth_epoch_sum / tf.cast(self.data_config["epoch_size"], tf.float32)

            # **Update running average over multiple epochs**
            p_infty_avg.assign((p_infty_avg * tf.cast(total_epochs - 1, tf.float32) + p_infty_epoch_avg) / tf.cast(total_epochs, tf.float32))

            E_Q_less_qth_avg.assign((E_Q_less_qth_avg * tf.cast(total_epochs - 1, tf.float32) + E_Q_less_qth_epoch_avg) / tf.cast(total_epochs, tf.float32))


            # **Update qth based on difference**
            diff = tf.abs(E_Q_less_qth_avg - p_infty_avg)
            if diff > self.threshold:
                if E_Q_less_qth_avg > p_infty_avg:
                    global_qth.assign(tf.maximum(1e-5, global_qth - self.step_size))  
                else:
                    global_qth.assign(tf.minimum(0.5, global_qth + self.step_size))

            # Store per-epoch values safely using tf.py_function
            tf.py_function(lambda x: p_infty_per_epoch.append(float(x.numpy())), [p_infty_avg], [])
            tf.py_function(lambda x: E_Q_less_qth_per_epoch.append(float(x.numpy())), [E_Q_less_qth_avg], [])
            tf.py_function(lambda x: qth_per_epoch.append(float(x.numpy())), [global_qth], [])

            # Reset accumulators for the next epoch
            p_infty_epoch_sum.assign(0.0)
            E_Q_less_qth_epoch_sum.assign(0.0)
            batch_count.assign(0)

        return P_infty_estimate

class FiniteOutageCoefficientLoss(InfiniteOutageCoefficientLoss):
    def __init__(self, S: int, reduction=tf.keras.losses.Reduction.AUTO, name=None):
        #self.qth = qth
        self.S = S
        super().__init__(reduction=tf.keras.losses.Reduction.AUTO, name=None)

    def q(self, y_true, y_pred):
        epsilon = 0.0001
        numerator = self.TP(y_true, y_pred) + self.FP(y_true, y_pred)
        denominator =  epsilon + self.TP(y_true, y_pred) + self.FP(y_true, y_pred) + self.TN(y_true, y_pred) + self.FN(y_true, y_pred)
        return tf.divide(numerator, denominator)

    def call(self, y_true, y_pred):
        P_infty_estimate = super().call(y_true, y_pred)
        M = self.M(y_true, y_pred)
        return M - tf.multiply(tf.pow(self.q(y_true, y_pred), self.S - 1), M - 1)

if __name__ == "__main__":
    loss = InfiniteOutageCoefficientLoss()
    path_name = "fin_coef_loss_model_rt-1.05_b-5_lstm-32"
    inputs = {
                                "taps": 1024,
                                "padding": 0,
                                "input_size": 100,
                                "output_size": 10,
                                "batch_size": 5,
                                "epoch_size": 1,
                                "phase_shift": 0.1,
                                "rate_threshold": 1.05
                            }
    training_generator = OutageData(**inputs,)
    multi_lstm_model = tf.keras.models.load_model(path_name, compile = False)
    X, y = training_generator.__getitem__(0)
    print(multi_lstm_model.predict(X))
    print(y)