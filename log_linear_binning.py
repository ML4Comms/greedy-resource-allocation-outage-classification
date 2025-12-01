import numpy as np
import tensorflow as tf
import toy_models
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.special import logit, expit
from betacal import BetaCalibration
from data_generator import OutageData

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

def calculate_brier_score(y_true, y_pred_probs):
    """Calculate Brier Score."""
    return np.mean((y_true - y_pred_probs) ** 2)

def setup_bins(binning_type, number_of_bins, epsilon=1e-5):
    """Set up bins for either linear or logarithmic binning."""
    if binning_type == 'linear':
        return np.linspace(0, 1, number_of_bins + 1)
    elif binning_type == 'log':
        return np.logspace(np.log10(epsilon), 0, num=number_of_bins + 1, base=10)
    
def fit_platt_scaling(y_true, y_pred_probs):
    """Fit Platt scaling coefficients by converting probabilities to logits using logistic regression."""
    y_true = np.array(y_true)
    y_pred_logits = logit(np.clip(y_pred_probs, 1e-5, 1 - 1e-5))  # Convert probabilities to logits

    lr = LogisticRegression(solver='lbfgs')
    lr.fit(y_pred_logits.reshape(-1, 1), y_true)
    
    A, B = lr.coef_[0][0], lr.intercept_[0]
    print(f"Fitted Platt scaling coefficients: A = {A}, B = {B}")
    return A, B

def fit_isotonic_scaling(y_true, y_pred_probs):
    """Fit Isotonic Regression model for calibration."""
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(y_pred_probs, y_true)
    print("Fitted isotonic regression model.")
    return iso_reg

def fit_temp_scaling(y_true, y_pred_probs):
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    # Temperature scaling applies to logits
    y_pred_logits = logit(y_pred_probs)

    def nll_loss(temp):
        scaled_logits = y_pred_logits / temp
        scaled_probs = expit(scaled_logits)
        return calculate_binary_nll(y_true, scaled_probs)
    
    result = minimize(nll_loss, x0=1.0, bounds=[(0.01, 100)])
    T = result.x[0]
    print(f"Fitted temperature parameter T = {T}")
    return T

def get_nll_function():
    return calculate_binary_nll

if __name__ == "__main__":
    binning_type = input("Enter 'log' for logarithmic binning or 'linear' for linear binning: ")
    scaling_type = input("Enter 'temp' for temp scaling, 'platt' for Platt scaling, 'beta' for Beta scaling, or 'isotonic' for Isotonic regression: ")

    # Initialize parameters based on calibration method
    A, B, T, a, b, c, iso_reg = None, None, None, None, None, None, None
    fit_platt, fit_temp, fit_beta, fit_iso = False, False, False, False
    
    # Configuration parameters for simulation
    out = 10
    number_of_bins = 10
    samples = 10000
    SNRs = [6.31]
    epsilon = 1e-5
    bin_edges = setup_bins(binning_type, number_of_bins, epsilon)

    # Main loop for simulation and calibration
    for run in range(15):
        for snr_db in SNRs:
            for resources in [4]:
                for lstm_size in [32]:
                    for model in ["bce"]:
                        for rt in [0.5]:
                            for qth in [0.5]:
                                uncalibrated_count = np.zeros(number_of_bins)
                                uncalibrated_accuracy = np.zeros(number_of_bins)
                                uncalibrated_confdistribution = np.zeros(number_of_bins)
                                calibrated_count = np.zeros(number_of_bins)
                                calibrated_accuracy = np.zeros(number_of_bins)
                                calibrated_confdistribution = np.zeros(number_of_bins)

                                all_y_true = []
                                all_y_pred_probs_uncalibrated = []

                                path_name = f"models/L_{model}_snr-{snr_db}_rt-{rt}_r-{resources}_qth--{qth}_lstm-{lstm_size}_out-{10}_phase-{0.1}"
                                inputs = {
                                    "taps": 1024,
                                    "padding": 0,
                                    "input_size": 100,
                                    "output_size": 10,
                                    "batch_size": resources,
                                    "epoch_size": 150,
                                    "phase_shift": 0.1,
                                    "rate_threshold": rt,
                                    "snr": snr_db
                                }

                                training_generator = OutageData(**inputs)
                                lstm_model = tf.keras.models.load_model(path_name, compile=False)

                                # Collect and store uncalibrated results
                                for _ in range(samples):
                                    X, y = training_generator.__getitem__(0)
                                    y_pred_vector = lstm_model.predict(X)
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

                                # Apply calibration
                                if scaling_type == 'platt':
                                    A, B = fit_platt_scaling(all_y_true, all_y_pred_probs_uncalibrated)
                                elif scaling_type == 'temp':
                                    T = fit_temp_scaling(all_y_true, all_y_pred_probs_uncalibrated)
                                elif scaling_type == 'beta':
                                    beta_calibrator = BetaCalibration(parameters="abm")
                                    beta_calibrator.fit(all_y_pred_probs_uncalibrated, all_y_true)
                                elif scaling_type == 'isotonic':
                                    iso_reg = fit_isotonic_scaling(all_y_true, all_y_pred_probs_uncalibrated)

                                all_y_pred_probs_calibrated = []

                                for i, pred_prob_uncalibrated in enumerate(all_y_pred_probs_uncalibrated):
                                    if scaling_type == 'platt' and A is not None and B is not None:
                                        pred_prob_calibrated = expit(A * logit(pred_prob_uncalibrated) + B)
                                    elif scaling_type == 'temp':
                                        pred_prob_calibrated = expit(logit(pred_prob_uncalibrated) / T)
                                    elif scaling_type == 'beta':
                                        pred_prob_calibrated = beta_calibrator.predict([pred_prob_uncalibrated])[0]  # Calibrate the probability
                                    elif scaling_type == 'isotonic' and iso_reg is not None:
                                        pred_prob_calibrated = iso_reg.transform([pred_prob_uncalibrated])[0]
                                    else:
                                        pred_prob_calibrated = pred_prob_uncalibrated

                                    all_y_pred_probs_calibrated.append(pred_prob_calibrated)

                                    # Calculate metrics
                                    bin_index = get_linear_bin_index(pred_prob_calibrated, number_of_bins) if binning_type == 'linear' else get_log_bin_index(pred_prob_calibrated, bin_edges)
                                    calibrated_count[bin_index] += 1
                                    calibrated_accuracy[bin_index] += int(all_y_true[i] > 0.5)
                                    calibrated_confdistribution[bin_index] += pred_prob_calibrated

                                for i in range(number_of_bins):
                                    if calibrated_count[i] > 0:
                                        calibrated_accuracy[i] /= calibrated_count[i]
                                        calibrated_confdistribution[i] /= calibrated_count[i]


                                ece_uncalibrated = calculate_ece(uncalibrated_accuracy, uncalibrated_confdistribution, uncalibrated_count)
                                mce_uncalibrated = calculate_mce(uncalibrated_accuracy, uncalibrated_confdistribution)
                                nll_uncalibrated = calculate_binary_nll(np.array(all_y_true), np.array(all_y_pred_probs_uncalibrated), epsilon)
                                brier_uncalibrated = calculate_brier_score(np.array(all_y_true), np.array(all_y_pred_probs_uncalibrated))

                                ece_calibrated = calculate_ece(calibrated_accuracy, calibrated_confdistribution, calibrated_count)
                                mce_calibrated = calculate_mce(calibrated_accuracy, calibrated_confdistribution)
                                nll_calibrated = calculate_binary_nll(np.array(all_y_true), np.array(all_y_pred_probs_calibrated), epsilon)
                                brier_calibrated = calculate_brier_score(np.array(all_y_true), np.array(all_y_pred_probs_calibrated))


                                filename = f"LSTM_results_{binning_type}_{scaling_type}_snr-{snr_db}.txt"
                                with open(filename, 'a') as file1:
                                    file1.write(f"Run: {run}, snr: {snr_db}, Resources: {resources}, Model: {model}, Rate Threshold: {rt}, Q-threshold: {qth}, Scaling: {scaling_type}\n")
                                    file1.write("Uncalibrated Results:\n")
                                    file1.write(f"ECE: {ece_uncalibrated:.4f}, MCE: {mce_uncalibrated:.4f}, NLL: {nll_uncalibrated:.4f},Brier Score: {brier_uncalibrated:.4f}\n")
                                    file1.write(f"Accuracy: {np.array2string(uncalibrated_accuracy, separator=',')}\n")
                                    file1.write(f"Conf Distribution: {np.array2string(uncalibrated_confdistribution, separator=',')}\n")
                                    file1.write("Calibrated Results:\n")
                                    file1.write(f"ECE: {ece_calibrated:.4f}, MCE: {mce_calibrated:.4f}, NLL: {nll_calibrated:.4f}, Brier Score: {brier_calibrated:.4f}\n")
                                    file1.write(f"Accuracy: {np.array2string(calibrated_accuracy, separator=',')}\n")
                                    file1.write(f"Conf Distribution: {np.array2string(calibrated_confdistribution, separator=',')}\n")
                                    file1.write("\n")

                                print(f"Uncalibrated predicted probabilities (sample): {np.array(all_y_pred_probs_uncalibrated[:10])}")
                                print(f"Calibrated predicted probabilities (sample): {np.array(all_y_pred_probs_calibrated[:10])}")
                                print("Results saved for scaling type:", scaling_type)