import tensorflow as tf
import matplotlib.pyplot as plt


class FrequencyResponseGenerator(tf.keras.utils.Sequence):
    def __init__(self, taps: int, padding: int = 100, phase_shift: float = 0.1, batch_size: int = 1, epoch_size: int = 1000, input_size: int = 100, output_size: int = 10):
        self.signal_length = padding + taps

        assert(self.signal_length >= batch_size)

        time_doman_samples = tf.complex(tf.random.normal((1, taps), mean = 0, stddev = tf.sqrt(1/float(taps))), tf.random.normal((1, taps), mean = 0, stddev = tf.sqrt(1/float(taps))))
        padding_signal = tf.complex(tf.zeros((1, padding)), tf.zeros((1, padding)))

        self.padded_signal = tf.concat([time_doman_samples, padding_signal], 1)
        self.frequency_domain = tf.signal.fft(self.padded_signal)/tf.sqrt(tf.cast(self.signal_length, dtype=tf.complex64))
        self.phase_shift = phase_shift
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.input_size = input_size
        self.output_size = output_size
        self.taps = taps
    #updating freq response
    def update_frequency_response(self):
        arg_shifter = tf.math.exp(tf.complex(0.0, 1.0) * tf.complex(tf.random.uniform(tf.shape(self.padded_signal), minval = -self.phase_shift, maxval = self.phase_shift), 0.0))
        self.padded_signal = tf.math.multiply(self.padded_signal, arg_shifter)
        self.frequency_domain = tf.signal.fft(self.padded_signal)

    #displaying updated freq response in an animation loop
    def run_animation(self, time: int = 10, ):
        _, ax = plt.subplots()
        plt.ion()
        plt.show(block = False)
        frame_period = 0.05
        number_of_frames = time / frame_period

        for _ in range(int(number_of_frames)):
            self.update_frequency_response()
            ax.clear()
            #plotting instantaneous capacity at each frequency point: log2(1 + freq points)
            ax.plot(tf.math.log(1 + tf.abs(tf.multiply(self.frequency_domain[0], tf.math.conj(self.frequency_domain[0])))/tf.math.log(2.0))) 
            ax.set_ylim([-5, 5])
            plt.pause(frame_period)

     #from frequency domain vector extract |R|<=1024 equally spaced samples   
    def __get_slice(self):
        return self.frequency_domain[0, 0 : self.batch_size * int(self.signal_length / self.batch_size) : int(self.signal_length / self.batch_size)]

    def get_data(self, index):
        reshape_vector = [self.batch_size, 1, 1]
        # running the loop k times
        X =self.__get_slice()
        X = tf.reshape(X, reshape_vector)
        for _ in range(self.input_size - 1):
            self.update_frequency_response()
            X = tf.concat([X, tf.reshape(self.__get_slice(), reshape_vector)], 1)
        #running the loop l times
        y = self.__get_slice()
        y = tf.reshape(y, reshape_vector)
        for _ in range(self.output_size - 1):
            self.update_frequency_response()
            y = tf.concat([y, tf.reshape(self.__get_slice(), reshape_vector)], 1)
        return tf.abs(X), tf.abs(y) #X = k samples and y = l samples

    # tensorflow methods you have to override when creating a 'tf.keras.utils.Sequence' child
    def __getitem__(self, index):
        return self.get_data(index)

    def __len__(self):
        return self.epoch_size

class OutageData(FrequencyResponseGenerator):
    def __init__(self, rate_threshold: float, taps: int, padding: int = 100, phase_shift: float = 0.1, batch_size: int = 1, epoch_size: int = 1000, input_size: int = 100, output_size: int = 10):
        self.rate_threshold = rate_threshold
        super().__init__(taps, padding, phase_shift, batch_size, epoch_size, input_size, output_size)
    def __getitem__(self, index):
        # X, [0, 1] success follows
        # X, [1, 0] outage follows
        X, y = self.get_data(index)
        y = tf.reduce_mean(tf.multiply(y,y), axis = 1) #mean squared magnitude of the output samples
        reshape_vector = [self.batch_size, 1]
        y = tf.reshape(y, reshape_vector)
        success = tf.cast(tf.math.log(1+y) / tf.math.log(2.0) > self.rate_threshold, tf.float32)
        return X,  tf.concat([1-success, success], axis=-1)


    def __len__(self):
        return self.epoch_size

if __name__ == "__main__":
    fre_response = FrequencyResponseGenerator(taps = 25, padding=100, batch_size=2, phase_shift = 0.1)
    fre_response.run_animation(10)
    print(fre_response.padded_signal)

    outage_data = OutageData(taps = 25, padding=10, batch_size=3, phase_shift = 0.1, rate_threshold = 1.05)
    print(outage_data.__getitem__(0))