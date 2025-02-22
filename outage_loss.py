import tensorflow as tf
from data_generator import OutageData
import sklearn

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
        self.qth = tf.Variable(qth)
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
        numerator = self.FN(y_true, y_pred)
        denominator =  epsilon + self.TN(y_true, y_pred) + self.FN(y_true, y_pred)
        return tf.divide(numerator, denominator)

    def call(self, y_true, y_pred):
        return self.M(y_true, y_pred)

class FiniteOutageCoefficientLoss(InfiniteOutageCoefficientLoss):
    def __init__(self, S: int, reduction=tf.keras.losses.Reduction.AUTO, name=None, qth=0.5):

        self.S = S

        # Running averages
        self.conditional_avg_y_pred = tf.Variable(0.0, trainable=False, dtype=tf.float32)  
        self.conditional_avg_M = tf.Variable(0.0, trainable=False, dtype=tf.float32)  
        self.count_y_pred = tf.Variable(0, trainable=False, dtype=tf.float32)  
        self.count_M = tf.Variable(0, trainable=False, dtype=tf.float32)

        self.bce = tf.keras.losses.BinaryCrossentropy()

        super().__init__(reduction=reduction, name=name, qth=qth)

    def q(self, y_true, y_pred):
        epsilon = 1e-4
        numerator = self.TP(y_true, y_pred) + self.FP(y_true, y_pred)
        denominator = epsilon + self.TP(y_true, y_pred) + self.FP(y_true, y_pred) + self.TN(y_true, y_pred) + self.FN(y_true, y_pred)
        return tf.divide(numerator, denominator)


    def conditional_average(self, y_pred, threshold):
        mask_y_pred = tf.cast(y_pred <= threshold, tf.float32)  
        valid_y_pred = y_pred * mask_y_pred  
        count_valid_y_pred = tf.reduce_sum(mask_y_pred)  
        sum_valid_y_pred = tf.reduce_sum(valid_y_pred)
        return tf.math.divide_no_nan(
            sum_valid_y_pred,
            tf.cast(count_valid_y_pred, tf.float32)
        )

    def P1(self, y_true, y_pred):
        epsilon = 0.0000001
        numerator = self.TP(y_true, y_pred) + self.FN(y_true, y_pred)
        denominator =  epsilon + self.TN(y_true, y_pred) + self.FN(y_true, y_pred) + self.TP(y_true, y_pred) + self.FP(y_true, y_pred)
        return tf.divide(numerator, denominator)

    def FQ(self, y_true, y_pred):
        epsilon = 0.0000001
        numerator = self.TN(y_true, y_pred) + self.FN(y_true, y_pred)
        denominator =  epsilon + self.TN(y_true, y_pred) + self.FN(y_true, y_pred) + self.TP(y_true, y_pred) + self.FP(y_true, y_pred)
        return tf.divide(numerator, denominator)

    def update_conditional_avg(self, y_pred, M):
        """ Updates running averages for y_pred (conditional) and M """

        # Update for y_pred where y_pred <= qth
        mask_y_pred = tf.cast(y_pred <= self.qth, tf.float32)  
        valid_y_pred = y_pred * mask_y_pred  
        count_valid_y_pred = tf.reduce_sum(mask_y_pred)  
        sum_valid_y_pred = tf.reduce_sum(valid_y_pred)

        new_count_y_pred = tf.cast(self.count_y_pred, tf.float32) + count_valid_y_pred
        new_avg_y_pred = tf.math.divide_no_nan(
            self.conditional_avg_y_pred * tf.cast(self.count_y_pred, tf.float32) + sum_valid_y_pred,
            tf.cast(new_count_y_pred, tf.float32)
        )

        # Assign updates for y_pred tracking
        self.conditional_avg_y_pred.assign(new_avg_y_pred)
        self.count_y_pred.assign(new_count_y_pred)

        # Update for M
        count_valid_M = tf.size(M, out_type=tf.float32)  
        sum_valid_M = tf.reduce_sum(M)

        new_count_M = self.count_M + count_valid_M
        new_avg_M = tf.math.divide_no_nan(
            self.conditional_avg_M * tf.cast(self.count_M, tf.float32) + sum_valid_M,
            tf.cast(new_count_M, tf.float32)
        )

        # Assign updates for M tracking
        self.conditional_avg_M.assign(new_avg_M)
        self.count_M.assign(new_count_M)

    def adjust_qth(self, shift):
        """ Dynamically updates qth based on the relationship between conditional_avg_y_pred and conditional_avg_M """

        # Shift qth down if conditional_avg_y_pred > conditional_avg_M, otherwise shift up
        shift = tf.cond(
            self.conditional_avg_y_pred > self.conditional_avg_M,
            lambda: -shift,  # Decrease qth
            lambda: shift    # Increase qth
        )
        self.qth.assign_add(shift)
        if self.qth >= 0.999:
            self.qth.assign(0.999)
        self.conditional_avg_y_pred.assign(0)
        self.conditional_avg_M.assign(0)
        self.count_M.assign(0)
        self.count_y_pred.assign(0)

    def call(self, y_true, y_pred):
        """ Computes the loss, updates the conditional averages, and adjusts qth dynamically """

        y_true_element = y_true
        y_pred_element = y_pred
        M = self.M(y_true_element, y_pred_element)

        # Update running averages
        self.update_conditional_avg(y_pred_element, M)
        return M - tf.multiply(tf.pow(self.q(y_true, y_pred), self.S - 1), M - 1) + tf.square((tf.maximum(self.qth - y_pred, 0)))
        # return self.P1(y_true,y_pred) * (tf.pow(1-self.FQ(y_true_element, y_pred_element), self.S - 1)) + self.M(y_true,y_pred) * (1-(tf.pow(1-self.FQ(y_true_element, y_pred_element), self.S - 1))) + self.bce(y_true, y_pred)#+ tf.square(self.conditional_average(y_pred, self.qth) - self.M(y_true, y_pred)) + tf.square(self.conditional_average(y_pred, 1) - self.P1(y_true, y_pred))

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