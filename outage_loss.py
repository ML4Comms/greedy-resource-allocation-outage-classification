import tensorflow as tf
from data_generator import OutageData


# Define a global qth variable that will be dynamically updated
global_qth = tf.Variable(0.5, trainable=False, dtype=tf.float32)  # Initial qth = 0.5
p_infty_history = []
E_Q_less_qth_history = []
qth_history = []


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

def calculate_P1(y_true, y_pred):
    TP = generalise_TP(y_true, y_pred)
    FN = generalise_FN(y_true, y_pred)
    TN = generalise_TN(y_true, y_pred)
    FP = generalise_FP(y_true, y_pred)
    denominator = TN + FN + TP + FP
    return tf.divide(TP + FN, denominator + 1e-4)  # Avoid division by zero

class MeanSquaredErrorTemperature(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name=None):
        super().__init__(reduction, name)

    def call(self, y_true, y_pred):
        return self.M(y_true, y_pred)

class InfiniteOutageCoefficientLoss(tf.keras.losses.Loss):
    def __init__(self, data_config, step_size=0.01, threshold=1e-4, reduction=tf.keras.losses.Reduction.AUTO, name=None):
        super().__init__(reduction, name)
        self.data_config = data_config
        self.step_size = step_size
        self.threshold = threshold

        # Variables to store per-epoch moving averages
        self.p_infty_avg = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self.E_Q_less_qth_avg = tf.Variable(0.0, dtype=tf.float32, trainable=False)

        # Running accumulators for batch-level tracking
        self.p_infty_epoch_sum = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self.E_Q_less_qth_epoch_sum = tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self.batch_count = tf.Variable(0, dtype=tf.int32, trainable=False)

        # Sliding window buffers
        self.p_infty_window = []
        self.E_Q_less_qth_window = []
        
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
        global global_qth, p_infty_history, E_Q_less_qth_history, qth_history  

        #  Compute P_infty per batch
        P_infty_estimate = self.M(y_true, y_pred)

        # Compute E[Q | Q < qth] for the batch
        mask = tf.cast(y_pred < global_qth, tf.float32)
        valid_values = tf.boolean_mask(y_pred, mask)
        E_Q_less_qth = tf.reduce_mean(valid_values) if tf.size(valid_values) > 0 else 0.0  

        #  Accumulate values for running averages
        self.p_infty_epoch_sum.assign_add(P_infty_estimate)
        self.E_Q_less_qth_epoch_sum.assign_add(E_Q_less_qth)
        self.batch_count.assign_add(1)

        #  Compute per-batch running average
        p_infty_running_avg = self.p_infty_epoch_sum / tf.cast(self.batch_count, tf.float32)
        E_Q_less_qth_running_avg = self.E_Q_less_qth_epoch_sum / tf.cast(self.batch_count, tf.float32)

        #  At the end of an epoch, compute final per-epoch averages
        if tf.equal(tf.math.mod(self.batch_count, self.data_config["epoch_size"]), 0):
            p_infty_epoch_avg = self.p_infty_epoch_sum / tf.cast(self.data_config["epoch_size"], tf.float32)
            E_Q_less_qth_epoch_avg = self.E_Q_less_qth_epoch_sum / tf.cast(self.data_config["epoch_size"], tf.float32)

            #  Adjust sliding window size dynamically
            N = tf.cast(tf.math.maximum(1, 10 / (p_infty_epoch_avg * self.data_config["epoch_size"] + 1e-4)), tf.int32)

            #  Update sliding window
            self.p_infty_window.append(float(p_infty_epoch_avg.numpy()))
            self.p_infty_window = self.p_infty_window[-N.numpy():]

            self.E_Q_less_qth_window.append(float(E_Q_less_qth_epoch_avg.numpy()))
            self.E_Q_less_qth_window = self.E_Q_less_qth_window[-N.numpy():]

            #  Compute final sliding window averages
            self.p_infty_avg.assign(sum(self.p_infty_window) / len(self.p_infty_window))
            self.E_Q_less_qth_avg.assign(sum(self.E_Q_less_qth_window) / len(self.E_Q_less_qth_window))

            #  Store values for tracking per epoch
            p_infty_history.append(float(self.p_infty_avg.numpy()))
            E_Q_less_qth_history.append(float(self.E_Q_less_qth_avg.numpy()))
            qth_history.append(float(global_qth.numpy()))

            # Reset accumulators for next epoch
            self.p_infty_epoch_sum.assign(0.0)
            self.E_Q_less_qth_epoch_sum.assign(0.0)
            self.batch_count.assign(0)

            tf.print("Updating history:", self.p_infty_avg, self.E_Q_less_qth_avg, global_qth)
        return P_infty_estimate
    
class FiniteOutageCoefficientLoss(InfiniteOutageCoefficientLoss):
    def __init__(self, data_config, S: int, reduction=tf.keras.losses.Reduction.AUTO, name=None):
        super(FiniteOutageCoefficientLoss, self).__init__(data_config=data_config, reduction=reduction, name=name)
        self.S = S
        
        # Ensure attributes are inherited correctly
        self.p_infty_avg = self.p_infty_avg if hasattr(self, 'p_infty_avg') else tf.Variable(0.0, dtype=tf.float32, trainable=False)
        self.E_Q_less_qth_avg = self.E_Q_less_qth_avg if hasattr(self, 'E_Q_less_qth_avg') else tf.Variable(0.0, dtype=tf.float32, trainable=False)

        
    def q(self, y_true, y_pred):
        epsilon = 1e-4  # Avoid division by zero
        numerator = self.TP(y_true, y_pred) + self.FP(y_true, y_pred)
        denominator = epsilon + self.TP(y_true, y_pred) + self.FP(y_true, y_pred) + self.TN(y_true, y_pred) + self.FN(y_true, y_pred)
        return tf.divide(numerator, denominator)

    def call(self, y_true, y_pred):
        P_infty_estimate = super().call(y_true, y_pred)
        M = self.M(y_true, y_pred)
        P1_estimate = calculate_P1(y_true, y_pred)
        return M - tf.multiply(tf.pow(self.q(y_true, y_pred), self.S - 1), M - 1) + tf.square(self.E_Q_less_qth_avg - P1_estimate)

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