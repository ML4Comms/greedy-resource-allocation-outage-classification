import tensorflow as tf
from data_generator import OutageData


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
        return tf.reduce_sum(tf.multiply(self.squashed_sigmoid(self.qth - y_pred), 1.0 - y_true))
    def FN(self, y_true, y_pred):
        return tf.reduce_sum(tf.multiply(self.squashed_sigmoid(self.qth - y_pred), y_true))
    def TP(self, y_true, y_pred):
        return tf.reduce_sum(tf.multiply(self.squashed_sigmoid(y_pred - self.qth), y_true))
    def FP(self, y_true, y_pred):
        return tf.reduce_sum(tf.multiply(self.squashed_sigmoid(y_pred - self.qth), 1.0 - y_true))

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