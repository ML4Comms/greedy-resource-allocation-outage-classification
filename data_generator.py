import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class FrequencyResponseGenerator(tf.keras.utils.Sequence):
    def __init__(self, taps: int, padding: int = 100, phase_shift: float = 0.1,
                 batch_size: int = 1, epoch_size: int = 1000, input_size: int = 100,
                 output_size: int = 10, snr: float = 1.0, model_type='sinc',
                 fading_type='rician', K_factor: float = 1.0, doppler_freq: float = 0.01,
                 los_mode: str = 'first'):
        
        self.signal_length = padding + taps
        self.taps = taps
        self.padding = padding
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.input_size = input_size
        self.output_size = output_size
        self.phase_shift = phase_shift
        self.model_type = model_type
        self.fading_type = fading_type
        self.K_factor = K_factor
        self.doppler_freq = doppler_freq
        self.los_mode = los_mode
        self.snr = float(snr)

        self.time_step = 0

        self._initialize_phase_evolution()
        self._generate_initial_channel()

    def _initialize_phase_evolution(self):
        self.nlos_theta = tf.random.uniform((1, self.taps), minval=-self.phase_shift, maxval=self.phase_shift)

    def _generate_initial_channel(self):
        if self.los_mode == 'first':
            self.los_phase = tf.random.uniform([], 0.0, 2 * np.pi)  # ONE shared phase
        elif self.los_mode == 'all':
            self.los_phase_vector = tf.random.uniform((1, self.taps), 0.0, 2 * np.pi)
        else:
                raise ValueError("los_mode must be 'first' or 'all'")

        if self.fading_type == 'rician':
            K = self.K_factor
            # Original LOS amplitude for total LOS power K/(K+1)
            base_los_amplitude = tf.sqrt(K / (K + 1))

            if self.los_mode == 'first':
                self.los_amplitude = base_los_amplitude
            elif self.los_mode == 'all':
                # Distribute the total LOS power across all taps
                self.los_amplitude = base_los_amplitude / tf.sqrt(float(self.taps))
            else:
                raise ValueError("los_mode must be 'first' or 'all'")

            scatter_stddev = tf.sqrt(1 / ((K + 1) * self.taps))
        else:
            self.los_amplitude = tf.constant(0.0)
            scatter_stddev = tf.sqrt(1 / 2*self.taps)

        real = tf.random.normal((1, self.taps), stddev=scatter_stddev)
        imag = tf.random.normal((1, self.taps), stddev=scatter_stddev)
        self.scattered_taps_initial = tf.complex(real, imag)
        self.scattered_taps = tf.identity(self.scattered_taps_initial)

        self._update_los_vector()

        signal = (self.los_vector + self.scattered_taps) * tf.complex(tf.sqrt(self.snr), 0.0)
        padding = tf.complex(tf.zeros((1, self.padding)), tf.zeros((1, self.padding)))
        self.padded_signal = tf.concat([signal, padding], axis=1)
        self.frequency_domain = tf.signal.fft(self.padded_signal) / tf.sqrt(tf.cast(self.signal_length, tf.complex64))

    def _update_los_vector(self):

        if self.los_mode == 'first':
            phase = 2 * np.pi * self.doppler_freq * self.time_step + self.los_phase
            los_val = tf.cast(self.los_amplitude, tf.complex64) * tf.exp(tf.complex(0.0, phase))
            self.los_vector = tf.concat([tf.reshape(los_val, [1, 1]),tf.zeros((1, self.taps - 1), dtype=tf.complex64)], axis=1)
        elif self.los_mode == 'all':
            phase = 2 * np.pi * self.doppler_freq * self.time_step + self.los_phase_vector
            self.los_vector = tf.cast(self.los_amplitude, tf.complex64) * tf.exp(tf.complex(0.0, phase))

        else:
            raise ValueError("los_mode must be 'first' or 'all'")

    def update_frequency_response(self):
        self.time_step += 1

        if self.model_type == 'sinc':#GLOBECOM WKSHP
            cumulative_phase = self.time_step * self.nlos_theta
            phase = tf.complex(tf.zeros_like(cumulative_phase), cumulative_phase)
            self.scattered_taps = self.scattered_taps_initial * tf.exp(phase)

        elif self.model_type == '3D':#TMLCN
            random_phase = tf.random.uniform((1, self.taps), minval=-self.phase_shift, maxval=self.phase_shift)
            phase = tf.complex(tf.zeros_like(random_phase), random_phase)
            self.scattered_taps *= tf.exp(phase)

        elif self.model_type == 'fallback':#TNSE
            theta = tf.random.uniform((1, self.taps), minval=-np.pi / 2, maxval=np.pi / 2)
            phase_vals = self.phase_shift * tf.sin(theta)
            phase = tf.complex(tf.zeros_like(phase_vals), phase_vals)
            self.scattered_taps *= tf.exp(phase)

        else:
            raise ValueError("Invalid model_type")

        if self.fading_type == 'rician':
            self._update_los_vector()
        else:
            self.los_vector = tf.zeros((1, self.taps), dtype=tf.complex64)

        time_signal = (self.los_vector + self.scattered_taps) * tf.complex(tf.sqrt(self.snr), 0.0)
        padding = self.padded_signal[:, self.taps:]
        self.padded_signal = tf.concat([time_signal, padding], axis=1)
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
    def __init__(self, rate_threshold: float, *args, **kwargs):
        self.rate_threshold = rate_threshold
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        # X, [0, 1] success follows
        # X, [1, 0] outage follows
        X, y = self.get_data(index)
        y = tf.math.log(1 + tf.math.square(tf.math.abs(y,y))) / tf.math.log(2.0)
        reshape_vector = [self.batch_size, 1]
        success = tf.cast(tf.reduce_mean(y, axis = 1) > self.rate_threshold, tf.float32)
        success = tf.reshape(success, reshape_vector)
        return X, 1-success


if __name__ == "__main__":
    fre_response = FrequencyResponseGenerator(
        taps=25, padding=100, batch_size=4, phase_shift=0.1,
        K_factor=1.0, doppler_freq=0.01, los_mode='all', model_type='sinc'
    )
    fre_response.run_animation(5)

    outage_data = OutageData(
        rate_threshold=1.0, taps=25, padding=10, batch_size=4, phase_shift=0.1,
        K_factor=1.0, doppler_freq=0.01, los_mode='all', model_type='sinc'
    )
    print(outage_data.__getitem__(0))