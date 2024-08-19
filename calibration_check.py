#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from data_generator import OutageData
from dqn_lstm import DQNLSTM
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize
import matplotlib.pyplot as plt

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

def calculate_modified_nll(y_true, y_pred_probs, qth, critical_only=False, epsilon=1e-5):
    """Calculate NLL for predictions above a certain threshold if critical_only is True."""
    y_pred_probs = np.clip(y_pred_probs, epsilon, 1 - epsilon)
    if critical_only:
        mask = y_pred_probs >= qth
        if not mask.any():
            return float('nan')
        y_true = y_true[mask]
        y_pred_probs = y_pred_probs[mask]

    if len(y_true) == 0:
        return float('nan')

    return -np.mean(y_true * np.log(y_pred_probs) + (1 - y_true) * np.log(1 - y_pred_probs))

def calculate_weighted_nll(y_true, y_pred_probs, qth, epsilon=1e-5, weight_factor=10):
    """Calculate weighted NLL where predictions above qth are given more weight."""
    y_pred_probs = np.clip(y_pred_probs, epsilon, 1 - epsilon)
    weights = np.where(y_pred_probs >= qth, weight_factor, 1)
    return -np.mean(weights * (y_true * np.log(y_pred_probs) + (1 - y_true) * np.log(1 - y_pred_probs)))

def setup_bins(binning_type, number_of_bins, epsilon=1e-5):
    """Set up bins for either linear or logarithmic binning."""
    if binning_type == 'linear':
        return np.linspace(0, 1, number_of_bins + 1)
    elif binning_type == 'log':
        return np.logspace(np.log10(epsilon), 0, num=number_of_bins + 1, base=10)
    
#def fit_platt_scaling(y_true, y_pred_probs):
    """Fit Platt scaling coefficients using weighted logistic regression."""
 #   y_true = np.array(y_true)
  #  y_pred_probs = np.array(y_pred_probs)

    # Calculate class weights
   # class_weights = {
    #    0: len(y_true) / (2 * np.sum(y_true == 0)),
     #   1: len(y_true) / (2 * np.sum(y_true == 1))
    #}
    
    #sample_weights = np.array([class_weights[cls] for cls in y_true])

    # Fit logistic regression model with weights
    #lr = LogisticRegression(solver='lbfgs', max_iter=1000)
    #lr.fit(y_pred_probs.reshape(-1, 1), y_true, sample_weight=sample_weights)
    
    #A, B = lr.coef_[0][0], lr.intercept_[0]
    #print(f"Fitted Platt scaling coefficients: A = {A}, B = {B}")
    #return A, B
    
def fit_platt_scaling(y_true, y_pred_probs):
    """Fit Platt scaling coefficients using logistic regression."""
    # Ensure y_true and y_pred_probs are numpy arrays
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    
    # Fit logistic regression model
    lr = LogisticRegression(solver='lbfgs')
    lr.fit(y_pred_probs.reshape(-1, 1), y_true)
    
    # Extract coefficients
    A, B = lr.coef_[0][0], lr.intercept_[0]
    
    print(f"Fitted Platt scaling coefficients: A = {A}, B = {B}")
    return A, B

def fit_isotonic_scaling(y_true, y_pred_probs):
    """Fit Isotonic Regression model for calibration."""
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(y_pred_probs, y_true)
    print("Fitted isotonic regression model.")
    return iso_reg

def fit_temp_scaling(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    """Fit temperature scaling parameter using NLL."""
    def nll_loss(temp):
        scaled_preds = y_pred / temp
        return calculate_binary_nll(y_true, scaled_preds)
    result = minimize(nll_loss, x0=1.0, bounds=[(0.01, 100)])
    return result.x[0]

def fit_beta_scaling(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    """Fit beta scaling parameters using NLL."""
    def nll_loss(params):
        a, b, c = params
        scaled_preds = 1 / (1 + np.exp(-a * np.log(y_pred) - b * np.log(1 - y_pred) - c))
        return calculate_binary_nll(y_true, scaled_preds)
    result = minimize(nll_loss, x0=[1.0, 1.0, 0.0], bounds=[(0.01, 10), (0.01, 10), (-10, 10)])
    return result.x

# Function to select which NLL calculation to use
def get_nll_function():
    choice = input("Choose NLL calculation: 1 - Standard, 2 - Critical Only, 3 - Weighted: ").strip()
    if choice == '1':
        return calculate_binary_nll
    qth = float(input("Enter the threshold qth for critical or weighted NLL: "))
    if choice == '2':
        return lambda y_true, y_pred_probs, epsilon=1e-5: calculate_modified_nll(y_true, y_pred_probs, qth, critical_only=True, epsilon=epsilon)
    elif choice == '3':
        weight_factor = float(input("Enter weight factor for critical predictions: "))
        return lambda y_true, y_pred_probs, epsilon=1e-5: calculate_weighted_nll(y_true, y_pred_probs, qth, epsilon=epsilon, weight_factor=weight_factor)

if __name__ == "__main__":
    binning_type = input("Enter 'log' for logarithmic binning or 'linear' for linear binning: ")
    scaling_type = input("Enter 'temp' for temperature scaling, 'platt' for Platt scaling, 'beta' for Beta scaling, or 'isotonic' for Isotonic regression: ")

    A, B, T, a, b, c, iso_reg = None, None, None, None, None, None, None  # Initialize parameters
    fit_platt, fit_temp, fit_beta, fit_iso = False, False, False, False
    if scaling_type == 'platt':
        fit_platt = input("Do you want to fit Platt scaling coefficients? (yes/no): ").strip().lower() == 'yes'
        if not fit_platt:
            A = float(input("Enter Platt scaling coefficient A: "))
            B = float(input("Enter Platt scaling coefficient B: "))
    elif scaling_type == 'temp':
        fit_temp = input("Do you want to fit Temperature scaling parameter? (yes/no): ").strip().lower() == 'yes'
        if not fit_temp:
            T = float(input("Enter Temperature scaling parameter T: "))
    elif scaling_type == 'beta':
        fit_beta = input("Do you want to fit Beta scaling parameters? (yes/no): ").strip().lower() == 'yes'
        if not fit_beta:
            a = float(input("Enter Beta scaling parameter a: "))
            b = float(input("Enter Beta scaling parameter b: "))
            c = float(input("Enter Beta scaling parameter c: "))
    elif scaling_type == 'isotonic':
        fit_iso = input("Do you want to fit Isotonic scaling? (yes/no): ").strip().lower() == 'yes'

    nll_function = get_nll_function()  # Get the NLL function based on user choice    
    filename = f"results_{binning_type}_{scaling_type}______.txt"
    out=10
    number_of_bins = 10
    samples = 10000
    SNRs = [3.5]
    epsilon = 1e-5
    bin_edges = setup_bins(binning_type, number_of_bins, epsilon)
    
    for run in range(1, 6):
        for snr in SNRs:
            for resources in [4]:
                for lstm_size in [32]:
                    for model in ["bce"]:
                        for rt in [0.5]:
                            for qth in [0.5]:
                               # Initialize metrics
                                uncalibrated_count = np.zeros(number_of_bins)
                                uncalibrated_accuracy = np.zeros(number_of_bins)
                                uncalibrated_confdistribution = np.zeros(number_of_bins)
                                calibrated_count = np.zeros(number_of_bins)
                                calibrated_accuracy = np.zeros(number_of_bins)
                                calibrated_confdistribution = np.zeros(number_of_bins)

                                all_y_true = []
                                all_y_pred_probs_uncalibrated = []

                                path_name = f"models/{model}_snr-{snr}_rt-{rt}_r-{resources}_qth--{qth}_lstm-{lstm_size}_out-{10}_phase-{0.1}"
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

                                        # Uncalibrated results
                                        bin_index = get_linear_bin_index(pred_prob_uncalibrated, number_of_bins) if binning_type == 'linear' else get_log_bin_index(pred_prob_uncalibrated, bin_edges)
                                        uncalibrated_count[bin_index] += 1
                                        uncalibrated_accuracy[bin_index] += outage
                                        uncalibrated_confdistribution[bin_index] += pred_prob_uncalibrated

                                        # Track the uncalibrated predictions
                                        all_y_true.append(y[idx][0])
                                        all_y_pred_probs_uncalibrated.append(pred_prob_uncalibrated)

                                # Normalize uncalibrated results
                                for i in range(number_of_bins):
                                    if uncalibrated_count[i] > 0:
                                        uncalibrated_accuracy[i] /= uncalibrated_count[i]
                                        uncalibrated_confdistribution[i] /= uncalibrated_count[i]

                                # Fit calibration models if necessary
                                if fit_platt and (A is None or B is None):
                                    A, B = fit_platt_scaling(all_y_true, all_y_pred_probs_uncalibrated)

                                if fit_temp and T is None:
                                    T = fit_temp_scaling(all_y_true, all_y_pred_probs_uncalibrated)

                                if fit_beta and (a is None or b is None or c is None):
                                    a, b, c = fit_beta_scaling(all_y_true, all_y_pred_probs_uncalibrated)

                                if fit_iso and iso_reg is None:
                                    iso_reg = fit_isotonic_scaling(all_y_true, all_y_pred_probs_uncalibrated)

                                all_y_pred_probs_calibrated = []

                                for i, pred_prob_uncalibrated in enumerate(all_y_pred_probs_uncalibrated):
                                    # Apply calibration
                                    if scaling_type == 'temp' and T:
                                        pred_prob_calibrated = pred_prob_uncalibrated / T
                                    elif scaling_type == 'platt' and A is not None and B is not None:
                                        pred_prob_calibrated = 1 / (1 + np.exp(-(pred_prob_uncalibrated * A + B)))
                                    elif scaling_type == 'beta' and a is not None and b is not None and c is not None:
                                        pred_prob_calibrated = 1 / (1 + np.exp(-(a * np.log(pred_prob_uncalibrated) + b * np.log(1 - pred_prob_uncalibrated) + c)))
                                    elif scaling_type == 'isotonic' and iso_reg is not None:
                                        pred_prob_calibrated = iso_reg.transform([pred_prob_uncalibrated])[0]
                                    else:
                                        pred_prob_calibrated = pred_prob_uncalibrated

                                    all_y_pred_probs_calibrated.append(pred_prob_calibrated)

                                    # Calibrated results
                                    bin_index = get_linear_bin_index(pred_prob_calibrated, number_of_bins) if binning_type == 'linear' else get_log_bin_index(pred_prob_calibrated, bin_edges)
                                    calibrated_count[bin_index] += 1
                                    calibrated_accuracy[bin_index] += int(all_y_true[i] > 0.5)
                                    calibrated_confdistribution[bin_index] += pred_prob_calibrated

                                # Normalize calibrated results
                                for i in range(number_of_bins):
                                    if calibrated_count[i] > 0:
                                        calibrated_accuracy[i] /= calibrated_count[i]
                                        calibrated_confdistribution[i] /= calibrated_count[i]

                                # Calculate metrics
                                ece_uncalibrated = calculate_ece(uncalibrated_accuracy, uncalibrated_confdistribution, uncalibrated_count)
                                mce_uncalibrated = calculate_mce(uncalibrated_accuracy, uncalibrated_confdistribution)
                                nll_uncalibrated = nll_function(np.array(all_y_true[:len(all_y_pred_probs_uncalibrated)]), np.array(all_y_pred_probs_uncalibrated), epsilon)

                                ece_calibrated = calculate_ece(calibrated_accuracy, calibrated_confdistribution, calibrated_count)
                                mce_calibrated = calculate_mce(calibrated_accuracy, calibrated_confdistribution)
                                nll_calibrated = nll_function(np.array(all_y_true), np.array(all_y_pred_probs_calibrated), epsilon)

                                # Save results after each run
                                with open(filename, 'a') as file1:
                                    file1.write(f"Run: {run}, SNR: {snr}, Resources: {resources}, Model: {model}, Rate Threshold: {rt}, Q-threshold: {qth}, Scaling: {scaling_type}\n")
                                    file1.write("Uncalibrated Results:\n")
                                    file1.write(f"ECE: {ece_uncalibrated:.4f}, MCE: {mce_uncalibrated:.4f}, NLL: {nll_uncalibrated:.4f}\n")
                                    file1.write(f"Accuracy: {np.array2string(uncalibrated_accuracy, separator=',')}\n")
                                    file1.write(f"Conf Distribution: {np.array2string(uncalibrated_confdistribution, separator=',')}\n")

                                    file1.write("Calibrated Results:\n")
                                    file1.write(f"ECE: {ece_calibrated:.4f}, MCE: {mce_calibrated:.4f}, NLL: {nll_calibrated:.4f}\n")
                                    file1.write(f"Accuracy: {np.array2string(calibrated_accuracy, separator=',')}\n")
                                    file1.write(f"Conf Distribution: {np.array2string(calibrated_confdistribution, separator=',')}\n")
                                    file1.write("\n")

                                # Print debug info
                                print(f"Uncalibrated predicted probabilities (sample): {np.array(all_y_pred_probs_uncalibrated[:10])}")
                                print(f"Calibrated predicted probabilities (sample): {np.array(all_y_pred_probs_calibrated[:10])}")
                                print(f"True labels distribution (uncalibrated): {np.bincount(np.array(all_y_true, dtype=int))}")
                                print(f"True labels distribution (calibrated): {np.bincount(np.array(all_y_true, dtype=int))}")
                                print("Results saved for scaling type:", scaling_type)


# In[ ]:





# In[ ]:




