import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import matplotlib.pyplot as plt


class FrequencyResponseGenerator(torch.utils.data.Dataset):
    def __init__(self, taps: int, padding: int = 100, phase_shift: float = 0.1,
                 batch_size: int = 1, epoch_size: int = 1000, input_size: int = 70,
                 output_size: int = 30, snr: float = 1.0, model_type='bessel',
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
        # Similar to TF version: uniform(-pi/2, pi/2)
        self.nlos_theta = torch.empty((1, self.taps)).uniform_(-np.pi / 2, np.pi / 2)

    def _generate_initial_channel(self):
        if self.los_mode == 'first':
            self.los_phase = torch.empty([]).uniform_(0.0, 2 * np.pi)
        elif self.los_mode == 'all':
            self.los_phase_vector = torch.empty((1, self.taps)).uniform_(0.0, 2 * np.pi)
        else:
            raise ValueError("los_mode must be 'first' or 'all'")

        if self.fading_type == 'rician':
            K = self.K_factor
            base_los_amplitude = torch.sqrt(torch.tensor(2 * K / (K + 1)))
            if self.los_mode == 'first':
                self.los_amplitude = base_los_amplitude
            elif self.los_mode == 'all':
                self.los_amplitude = base_los_amplitude / torch.sqrt(torch.tensor(float(self.taps)))
            scatter_stddev = torch.sqrt(torch.tensor(1 / ((K + 1) * self.taps)))
        else:
            self.los_amplitude = torch.tensor(0.0)
            scatter_stddev = torch.sqrt(torch.tensor(1 / self.taps))

        real = torch.randn((1, self.taps)) * scatter_stddev
        imag = torch.randn((1, self.taps)) * scatter_stddev
        self.scattered_taps_initial = real + 1j * imag
        self.scattered_taps = self.scattered_taps_initial.clone()

        self._update_los_vector()

        signal = (self.los_vector + self.scattered_taps) * torch.sqrt(torch.tensor(self.snr))
        padding = torch.zeros((1, self.padding), dtype=torch.complex64)
        self.padded_signal = torch.cat([signal, padding], axis=1)
        self.frequency_domain = torch.fft.fft(self.padded_signal) / torch.sqrt(
            torch.tensor(self.signal_length, dtype=torch.complex64))

    def _update_los_vector(self):
        if self.los_mode == 'first':
            phase = 2 * np.pi * self.doppler_freq * self.time_step + self.los_phase
            los_val = self.los_amplitude * torch.exp(1j * phase)
            self.los_vector = torch.cat(
                [los_val.reshape(1, 1),
                 torch.zeros((1, self.taps - 1), dtype=torch.complex64)], axis=1)
        elif self.los_mode == 'all':
            phase = 2 * np.pi * self.doppler_freq * self.time_step + self.los_phase_vector
            self.los_vector = self.los_amplitude * torch.exp(1j * phase)
        else:
            raise ValueError("los_mode must be 'first' or 'all'")

    def update_frequency_response(self):
        self.time_step += 1

        if self.model_type == 'sinc':
            cumulative_phase = self.time_step * self.nlos_theta
            self.scattered_taps = self.scattered_taps_initial * torch.exp(1j * cumulative_phase)

        elif self.model_type == 'bessel':
            # Bessel-type phase evolution: phase_shift * t * sin(theta)
            phase_vals = self.phase_shift * self.time_step * torch.sin(self.nlos_theta)
            self.scattered_taps = self.scattered_taps_initial * torch.exp(1j * phase_vals)

        elif self.model_type == '3D':
            random_phase = torch.empty((1, self.taps)).uniform_(-self.phase_shift, self.phase_shift)
            self.scattered_taps *= torch.exp(1j * random_phase)

        elif self.model_type == 'fallback':
            theta = torch.empty((1, self.taps)).uniform_(-np.pi / 2, np.pi / 2)
            phase_vals = self.phase_shift * torch.sin(theta)
            self.scattered_taps *= torch.exp(1j * phase_vals)

        else:
            raise ValueError("Invalid model_type")

        if self.fading_type == 'rician':
            self._update_los_vector()
        else:
            self.los_vector = torch.zeros((1, self.taps), dtype=torch.complex64)

        time_signal = (self.los_vector + self.scattered_taps) * torch.sqrt(torch.tensor(self.snr))
        padding = self.padded_signal[:, self.taps:]
        self.padded_signal = torch.cat([time_signal, padding], axis=1)
        self.frequency_domain = torch.fft.fft(self.padded_signal)

    def run_animation(self, time: int = 10):
        _, ax = plt.subplots()
        plt.ion()
        plt.show(block=False)
        frame_period = 0.05
        number_of_frames = int(time / frame_period)

        for _ in range(number_of_frames):
            self.update_frequency_response()
            ax.clear()
            capacity = torch.log2(1 + torch.abs(self.frequency_domain[0]).pow(2))
            ax.plot(capacity.numpy())
            ax.set_ylim([0, 5])
            plt.pause(frame_period)

    def __get_slice(self):
        return self.frequency_domain[0,
               0: self.batch_size * int(self.signal_length / self.batch_size):
               int(self.signal_length / self.batch_size)]

    def get_data(self, index):
        reshape_vector = [self.batch_size, 1, 1]

        X = self.__get_slice().reshape(reshape_vector)
        for _ in range(self.input_size - 1):
            self.update_frequency_response()
            X = torch.cat([X, self.__get_slice().reshape(reshape_vector)], 1)

        y = self.__get_slice().reshape(reshape_vector)
        for _ in range(self.output_size - 1):
            self.update_frequency_response()
            y = torch.cat([y, self.__get_slice().reshape(reshape_vector)], 1)

        return torch.abs(X), torch.abs(y)

    def __getitem__(self, index):
        return self.get_data(index)

    def __len__(self):
        return self.epoch_size


class OutageData(FrequencyResponseGenerator):
    def __init__(self, rate_threshold: float, *args, **kwargs):
        self.rate_threshold = rate_threshold
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        X, y = self.get_data(index)
        capacity = torch.log2(1 + torch.abs(y).pow(2))
        success = (torch.mean(capacity, axis=1) > self.rate_threshold).float()
        success = success.reshape([self.batch_size, 1])
        return X, 1 - success


if __name__ == "__main__":
    fre_response = FrequencyResponseGenerator(
        taps=25, padding=100, batch_size=4, phase_shift=0.1,
        K_factor=1.0, doppler_freq=0.01, los_mode='all', model_type='bessel'
    )
    fre_response.run_animation(5)

    outage_data = OutageData(
        rate_threshold=1.0, taps=25, padding=10, batch_size=4, phase_shift=0.1,
        K_factor=1.0, doppler_freq=0.01, los_mode='all', model_type='bessel'
    )
    print(outage_data.__getitem__(0))
