import tensorflow as tf
from data_generator import OutageData

def heaviside(x):
    return tf.experimental.numpy.heaviside(x,0)

def TPR(qth, y_true, y_pred, step_function = heaviside):
    result = generalise_TP(qth, y_true, y_pred, step_function) / (generalise_TP(qth, y_true, y_pred, step_function) + generalise_FN(qth, y_true, y_pred, step_function))
    if tf.math.is_nan(result):
        return tf.constant(0)
    else:
        return result

def FPR(qth, y_true, y_pred, step_function = heaviside):
    result = generalise_FP(qth, y_true, y_pred, step_function) / (generalise_FP(qth, y_true, y_pred, step_function) + generalise_TN(qth, y_true, y_pred, step_function))
    if tf.math.is_nan(result):
        return tf.constant(0)
    else:
        return result
    
def Precision(qth, y_true, y_pred, step_function = heaviside):
    result = generalise_TP(qth, y_true, y_pred, step_function) / (generalise_TP(qth, y_true, y_pred, step_function) + generalise_FP(qth, y_true, y_pred, step_function))
    if tf.math.is_nan(result):
        return tf.constant(0)
    else:
        return result

def generalise_TN(qth, y_true, y_pred, step_function = heaviside):
    return tf.reduce_sum(tf.multiply(step_function(qth - y_pred), 1.0 - y_true))
def generalise_FN(qth, y_true, y_pred, step_function = heaviside):
    return tf.reduce_sum(tf.multiply(step_function(qth - y_pred), y_true))
def generalise_TP(qth, y_true, y_pred, step_function = heaviside):
    return tf.reduce_sum(tf.multiply(step_function(y_pred - qth), y_true))
def generalise_FP(qth, y_true, y_pred, step_function = heaviside):
    return tf.reduce_sum(tf.multiply(step_function(y_pred - qth), 1.0 - y_true))


class MeanSquaredErrorTemperature(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name=None):
        super().__init__(reduction, name)

    def call(self, y_true, y_pred):
        return self.M(y_true, y_pred)

class InfiniteOutageCoefficientLoss(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name=None, qth = 0.5):
        self.qth = qth
        super().__init__(reduction, name)

    def squashed_sigmoid(self, input, factor: float = 5.0):
        return tf.divide(1, 1 + tf.math.exp(tf.multiply(-factor, input)))

    def TN(self, y_true, y_pred):
        return generalise_TN(qth=self.qth, y_true=y_true, y_pred=y_pred, step_function=self.squashed_sigmoid)
    def FN(self, y_true, y_pred):
        return generalise_FN(qth= self.qth, y_true=y_true, y_pred=y_pred, step_function=self.squashed_sigmoid)
    def TP(self, y_true, y_pred):
        return generalise_TP(qth=self.qth, y_true=y_true, y_pred=y_pred, step_function=self.squashed_sigmoid)
    def FP(self, y_true, y_pred):
        return generalise_FP(qth=self.qth, y_true=y_true, y_pred=y_pred, step_function=self.squashed_sigmoid)

    def M(self, y_true, y_pred):
        epsilon = 0.0001
        numerator = tf.multiply(self.FN(y_true, y_pred), self.TN(y_true, y_pred) + self.FN(y_true, y_pred) + self.TP(y_true, y_pred) + self.FP(y_true, y_pred))
        denominator =  epsilon + tf.multiply(self.TN(y_true, y_pred) + self.FN(y_true, y_pred), self.FN(y_true, y_pred) + self.TP(y_true, y_pred))
        return tf.divide(numerator, denominator)

    def call(self, y_true, y_pred):
        return self.M(y_true, y_pred)

class FiniteOutageCoefficientLoss(InfiniteOutageCoefficientLoss):
    def __init__(self, S: int, reduction=tf.keras.losses.Reduction.AUTO, name=None, qth = 0.5):
        self.qth = qth
        self.S = S
        super().__init__(reduction=tf.keras.losses.Reduction.AUTO, name=None, qth = 0.5)

    def q(self, y_true, y_pred):
        epsilon = 0.0001
        numerator = self.TP(y_true, y_pred) + self.FP(y_true, y_pred)
        denominator =  epsilon + self.TP(y_true, y_pred) + self.FP(y_true, y_pred) + self.TN(y_true, y_pred) + self.FN(y_true, y_pred)
        return tf.divide(numerator, denominator)

    def call(self, y_true, y_pred):
        y_true_element = y_true
        y_pred_element = y_pred
        M = self.M(y_true_element, y_pred_element)
        return M - tf.multiply(tf.pow(self.q(y_true_element, y_pred_element), self.S - 1), M - 1)

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