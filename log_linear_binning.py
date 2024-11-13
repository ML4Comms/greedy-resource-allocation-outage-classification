#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from data_generator import OutageData
import matplotlib.pyplot as plt
from scipy.special import logit, expit

def get_linear_bin_index(probability, number_of_bins):
    """Calculate the bin index for linear binning."""
    answer = int(probability * number_of_bins)
    return answer if answer < number_of_bins else answer - 1

def get_log_bin_index(probability, bin_edges):
    """Calculate the bin index for logarithmic binning."""
    return np.digitize(probability, bin_edges, right=True) - 1

def calculate_ece(accuracies, confidences, counts):
    """Calculate Expected Calibration Error (ECE)."""
    total = np.sum(counts)
    return np.sum(np.abs(accuracies - confidences) * (counts / total)) if total > 0 else 0

def calculate_mce(accuracies, confidences):
    """Calculate Maximum Calibration Error (MCE)."""
    return np.max(np.abs(accuracies - confidences))

def calculate_binary_nll(y_true, y_pred_probs, epsilon=1e-5):
    """Calculate Binary Negative Log-Likelihood (NLL)."""
    y_pred_probs = np.clip(y_pred_probs, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred_probs) + (1 - y_true) * np.log(1 - y_pred_probs))

def setup_bins(binning_type, number_of_bins, epsilon=1e-5):
    """Set up bins for either linear or logarithmic binning."""
    if binning_type == 'linear':
        return np.linspace(0, 1, number_of_bins + 1)
    elif binning_type == 'log':
        return np.logspace(np.log10(epsilon), 0, num=number_of_bins + 1, base=10)

if __name__ == "__main__":
    binning_type = input("Enter 'log' for logarithmic binning or 'linear' for linear binning: ")

    # Configuration parameters for simulation
    number_of_bins = 10
    samples = 10
    SNRs = [1.58]
    epsilon = 1e-5
    bin_edges = setup_bins(binning_type, number_of_bins, epsilon)

    # Main loop for simulation
    for run in range(6):
        for snr in SNRs:
            for resources in [4]:
                for lstm_size in [32]:
                    for model in ["fin_coef_loss", "bce"]:
                        for rt in [0.5]:
                            for qth in [0.5]:
                                uncalibrated_count = np.zeros(number_of_bins)
                                uncalibrated_accuracy = np.zeros(number_of_bins)
                                uncalibrated_confdistribution = np.zeros(number_of_bins)

                                all_y_true = []
                                all_y_pred_probs_uncalibrated = []

                                path_name = f"models/{model}_snr-{snr}_rt-{rt}_r-{resources}_qth--{qth}_lstm-{lstm_size}_out-10_phase-0.1"
                                inputs = {
                                    "taps": 1024,
                                    "padding": 0,
                                    "input_size": 100,
                                    "output_size": 10,
                                    "batch_size": resources,
                                    "epoch_size": 150,
                                    "phase_shift": 0.1,
                                    "rate_threshold": rt,
                                    "snr": snr
                                }

                                training_generator = OutageData(**inputs)
                                dqn_lstm_model = tf.keras.models.load_model(path_name, compile=False)

                                # Collect and store uncalibrated results
                                for _ in range(samples):
                                    X, y = training_generator.__getitem__(0)
                                    y_pred_vector = dqn_lstm_model.predict(X)
                                    for idx, pred in enumerate(y_pred_vector):
                                        outage = int(y[idx][0] > 0.5)
                                        pred_prob_uncalibrated = pred[0]
                                        bin_index = get_linear_bin_index(pred_prob_uncalibrated, number_of_bins) if binning_type == 'linear' else get_log_bin_index(pred_prob_uncalibrated, bin_edges)
                                        uncalibrated_count[bin_index] += 1
                                        uncalibrated_accuracy[bin_index] += outage
                                        uncalibrated_confdistribution[bin_index] += pred_prob_uncalibrated

                                        all_y_true.append(y[idx][0])
                                        all_y_pred_probs_uncalibrated.append(pred_prob_uncalibrated)

                                for i in range(number_of_bins):
                                    if uncalibrated_count[i] > 0:
                                        uncalibrated_accuracy[i] /= uncalibrated_count[i]
                                        uncalibrated_confdistribution[i] /= uncalibrated_count[i]

                                # Metrics calculation
                                ece_uncalibrated = calculate_ece(uncalibrated_accuracy, uncalibrated_confdistribution, uncalibrated_count)
                                mce_uncalibrated = calculate_mce(uncalibrated_accuracy, uncalibrated_confdistribution)
                                nll_uncalibrated = calculate_binary_nll(np.array(all_y_true), np.array(all_y_pred_probs_uncalibrated), epsilon)

                                # Save results
                                filename = f"results_{binning_type}_uncalibrated_4db.txt"
                                with open(filename, 'a') as file1:
                                    file1.write(f"Run: {run}, SNR: {snr}, Resources: {resources}, Model: {model}, Rate Threshold: {rt}, Q-threshold: {qth}\n")
                                    file1.write("Uncalibrated Results:\n")
                                    file1.write(f"ECE: {ece_uncalibrated:.4f}, MCE: {mce_uncalibrated:.4f}, NLL: {nll_uncalibrated:.4f}\n")
                                    file1.write(f"Accuracy: {np.array2string(uncalibrated_accuracy, separator=',')}\n")
                                    file1.write(f"Conf Distribution: {np.array2string(uncalibrated_confdistribution, separator=',')}\n")
                                    file1.write("\n")

                                # Debug info
                                print(f"Uncalibrated predicted probabilities (sample): {np.array(all_y_pred_probs_uncalibrated[:10])}")
                                print("Results saved for uncalibrated model.")


# In[ ]:




