import json
import tensorflow as tf
import numpy as np
from outage_loss import InfiniteOutageCoefficientLoss, TPR, FPR, Precision
import toy_models
from dqn_lstm import DQNLSTM
from scipy.optimize import minimize
from scipy.special import expit, logit
from data_generator import OutageData

def bubble_sort_indices(arr):
    n = len(arr)
    indices = list(range(n))
    for i in range(n-1):
        for j in range(0, n-i-1):
            if arr[indices[j]] > arr[indices[j+1]]:
                indices[j], indices[j+1] = indices[j+1], indices[j]
    return indices

def mean_of_values_at_indices(arr, indices):
    total = sum(arr[i] for i in indices)
    return total / len(indices)

cdf = {}
P_inf = {}
P_1 = {}
P_R_critical = {}
P_R = {}
P_best_N = {}
tpr = {}
fpr = {}
precision = {}
resources_used = {}
P_R_critical_average_outage_counters = {}
P_R_average_outage_counters = {}
best_N_average_outage_counters = {}
P_1_average_outage_counters = {}
P_inf_average_outage_counters = {}
cdf_average_outage_counters = {}
average_resources_used = {}
tpr_average_outage_counters = {}
fpr_average_outage_counters = {}
precision_average_outage_counters = {}
number_of_training_routines_per_model = 2
number_to_discard = 0
out = 10
number_of_tests = 11
SNRs = [1.58]
qth_range = [0.5]
phase_shift = 0.1
epochs = 3
epoch_size = 15
resources = [4]
rates = [1.0]
K_factor = 0.0
model_prefix_names = ["bce"]
#model_prefix_names = ["fin_coef_loss","binary_cross_entropy"]
force_retrain_models = True

scaling_method = input("Enter the scaling method ('temp','platt','beta', 'isotonic','none'): ").strip().lower()

set_params = False
if scaling_method != 'none':
    set_params = input(f"Do you want to set hard values for {scaling_method} scaling? (yes/no): ").strip().lower() == 'yes'

if scaling_method == 'platt' and set_params:
    platt_A = float(input("Enter Platt scaling coefficient A: "))
    platt_B = float(input("Enter Platt scaling coefficient B: "))

elif scaling_method == 'temp' and set_params:
    temp_T = float(input("Enter Temperature scaling parameter T: "))

# Prompt for model type once
use_model = input("Press 1 to use LSTM, any other key for DQN-LSTM: ")

nll_function = DQNLSTM.calculate_binary_nll

# Define function to compute Eqth
def conditional_average(y_pred, qth):
    y_pred = np.array(y_pred).flatten()  # Ensure proper shape
    mask = y_pred <= qth  # Select values below threshold
    if np.sum(mask) == 0:
        return 0  # Avoid division by zero
    return np.mean(y_pred[mask])  # Compute mean of valid values

# Initialize Eqth dictionaries
Eqth_values = {}
Eqth_average_outage_counters = {}

for snr_db in SNRs:
    for qth in qth_range:
        for lstm_size in [32]:
            for resource in resources:
                for model_prefix in model_prefix_names:
                    for rate_threshold in rates:
                        k_label = f"k-{K_factor}" if K_factor is not None else "rayleigh"
                        model_result = f"{model_prefix}_{k_label}_snr-{snr_db}_rt-{rate_threshold}_r-{resource}_qth--{qth}_out-{out}_phase-{phase_shift}"

                        #model_result = f"{model_prefix}_snr-{snr_db}_rt-{rate_threshold}_r-{resource}_qth--{qth}_lstm-{lstm_size}_out-{out}_phase-{phase_shift}"
                        model_name = f"{model_prefix}_{k_label}_snr-{snr_db}_rt-{rate_threshold}_r-{resource}_qth--{qth}_out-{out}_phase-{phase_shift}"


                        P_R_critical_average_outage_counters[model_result] = []
                        P_R_average_outage_counters[model_result] = []
                        average_resources_used[model_result] = []
                        best_N_average_outage_counters[model_result] = []
                        P_inf_average_outage_counters[model_result] = []
                        P_1_average_outage_counters[model_result] = 0
                        cdf_average_outage_counters[model_result] = []

                        tpr_average_outage_counters[model_result] = []
                        fpr_average_outage_counters[model_result] = []
                        precision_average_outage_counters[model_result] = []

                        for _ in range(number_of_training_routines_per_model):
                            data_config = {
                                "taps": 1024,
                                "padding": 0,
                                "input_size": 100,
                                "output_size": out,
                                "batch_size": resource,
                                "epoch_size": epoch_size,
                                "phase_shift": phase_shift,
                                "rate_threshold": rate_threshold,
                                "snr": snr_db}

                            # Only set fading-related parameters depending on K-factor
                            if K_factor is None:
                                data_config["fading_type"] = "rayleigh"
                            else:
                                data_config["fading_type"] = "rician"
                                data_config["K_factor"] = K_factor

                                

                            if use_model == '1':
                                model = toy_models.get_fitted_model(data_input=data_config, 
                                                                    model_name=model_name, 
                                                                    epochs=epochs, 
                                                                    force_retrain=force_retrain_models, 
                                                                    lstm_units=lstm_size,
                                                                    qth=qth_range)
                            else:
                                model = DQNLSTM(qth, epochs=epochs, data_config=data_config, model_name=model_name, lstm_units=lstm_size)

                            training_generator = OutageData(**data_config,)

                            P_best_N[model_result] = 0.0
                            P_R[model_result] = 0.0
                            P_R_critical[model_result] = 0.0
                            resources_used[model_result] = 0.0
                            tpr[model_result] = 0.0
                            fpr[model_result] = 0.0
                            precision[model_result] = 0.0

                            P_inf[model_result] = 0.0
                            cdf[model_result] = 0.0
                            P_1[model_result] = 0.0

                            P_1_counter = 0
                            P_inf_counter = 0
                            cdf_counter = 0

                            if scaling_method != 'none':
                                if scaling_method == 'platt':
                                    if set_params:
                                        model.A, model.B = platt_A, platt_B
                                    else:
                                        model.calibrate(data_config, method=scaling_method)
                                        platt_A, platt_B = model.A, model.B
                                elif scaling_method == 'temp':
                                    if set_params:
                                        model.temp = temp_T
                                    else:
                                        model.calibrate(data_config, method=scaling_method)
                                        temp_T = model.temp
                                elif scaling_method == 'beta':
                                    model.calibrate(data_config, method='beta')
                                elif scaling_method == 'isotonic':
                                    model.calibrate(data_config, method='isotonic')
                                else:
                                    model.calibrate(data_config, method=scaling_method)

                            # Initialize storage
                            Eqth_values[model_result] = []
                            Eqth_average_outage_counters[model_result] = []

                            for _ in range(number_of_tests):
                                X, y_label = training_generator.__getitem__(0)
                                # Get the initial predictions
                                Y_pred = model.predict(X)

                                # Apply calibration
                                if scaling_method == 'none':
                                    pass  # No changes to Y_pred
                                elif scaling_method == 'platt':
                                    Y_pred = expit(model.A * logit(Y_pred) + model.B)
                                elif scaling_method == 'temp':
                                    Y_pred = expit(logit(Y_pred) / model.temp)
                                elif scaling_method == 'beta':
                                    Y_pred = model.beta_calibrator.predict(Y_pred).reshape(-1, 1).astype(np.float32)
                                if scaling_method == 'isotonic':
                                    #print("Before Isotonic Scaling:", Y_pred[:5])  # Print first 5 values
                                    Y_pred = model.isotonic_predict(Y_pred).astype(np.float32).reshape(-1, 1)
                                    #print("After Isotonic Scaling:", Y_pred[:5])  # Print first 5 values

                                # Compute Eqth
                                Eqth_val = conditional_average(Y_pred,0.1)
                                Eqth_values[model_result].append(Eqth_val)

                                resource_used = 0

                                should_count = True
                                lowest_y_pred = float('inf')
                                best_of_N_in_outage = False
                                idx_of_best = 0

                                tpr[model_result] += TPR(qth=qth, y_pred=Y_pred, y_true=y_label).numpy()
                                fpr[model_result] += FPR(qth=qth, y_pred=Y_pred, y_true=y_label).numpy()
                                precision[model_result] += Precision(qth=qth, y_pred=Y_pred, y_true=y_label).numpy()
                                for idx, y_pred in enumerate(Y_pred):
                                    P_1_counter += 1
                                    P_1[model_result] += float(y_label[idx][0])
                                    cdf_counter += 1
                                    if y_pred[0] <=0.1:
                                        #print(y_pred[0])
                                        cdf[model_result] += 1.0
                                        P_inf_counter += 1
                                        #print(P_inf_counter)
                                        P_inf[model_result] += float(y_label[idx][0])
                                        #print(P_inf[model_result])

                                    if y_pred[0] < lowest_y_pred:
                                        lowest_y_pred = y_pred[0]
                                        idx_of_best = idx
                                        best_of_N_in_outage = float(y_label[idx][0]) >=0.1

                                    if should_count:
                                        if idx == resource-1:
                                            resource_used = idx_of_best
                                            P_R_critical[model_result] += float(best_of_N_in_outage)
                                        elif y_pred[0] <=0.1:
                                            resource_used = idx
                                            should_count = False
                                            P_R_critical[model_result] += float(y_label[idx][0] >=0.1)
                                        if idx == resource-1 or y_pred[0] <=0.1:
                                            resource_used = idx
                                            should_count = False
                                            P_R[model_result] += float(y_label[idx][0] >=0.1)
                                            print(P_R[model_result])

                                P_best_N[model_result] += float(best_of_N_in_outage)
                                resources_used[model_result] += resource_used

                            # Compute final Eqth average
                            Eqth_average_outage_counters[model_result] = np.mean(Eqth_values[model_result])
                            print(Eqth_average_outage_counters[model_result])
                            # Convert all float32 values to Python floats before saving
                            Eqth_average_outage_counters = {k: float(v) for k, v in Eqth_average_outage_counters.items()}

                            P_best_N[model_result] /= number_of_tests
                            best_N_average_outage_counters[model_result].append(P_best_N[model_result])

                            P_R[model_result] /= number_of_tests
                            P_R_average_outage_counters[model_result].append(P_R[model_result])

                            tpr[model_result] /= number_of_tests
                            tpr_average_outage_counters[model_result].append(tpr[model_result])

                            fpr[model_result] /= number_of_tests
                            fpr_average_outage_counters[model_result].append(fpr[model_result])

                            precision[model_result] /= number_of_tests
                            precision_average_outage_counters[model_result].append(precision[model_result])

                            P_R_critical[model_result] /= number_of_tests
                            P_R_critical_average_outage_counters[model_result].append(P_R_critical[model_result])

                            resources_used[model_result] /= number_of_tests
                            average_resources_used[model_result].append(resources_used[model_result])

                            P_1[model_result] /= P_1_counter
                            P_1_average_outage_counters[model_result] += P_1[model_result]

                            P_inf[model_result] = 0 if P_inf_counter == 0 else P_inf[model_result] / P_inf_counter
                            P_inf_average_outage_counters[model_result].append(P_inf[model_result])

                            cdf[model_result] /= cdf_counter
                            cdf_average_outage_counters[model_result].append(cdf[model_result])

                            with open(f'simulation_results_{resource}_{model_prefix}_{scaling_method}_{qth}_nbest.txt', 'a') as convert_file:
                                convert_file.write("\nData configuration:\n")
                                convert_file.write(json.dumps(data_config, indent=4))
                                convert_file.write(f"\nEpochs: {epochs}, qth: {qth}, Number of tests: {number_of_tests}, scaling method: {scaling_method}\n")
                                convert_file.write("\n\nP_R:\n")
                                convert_file.write(json.dumps(P_R, indent=4))
                                convert_file.write("\n\nPRECISION:\n")
                                convert_file.write(json.dumps(precision, indent=4))
                                convert_file.write("\n\nTPR:\n")
                                convert_file.write(json.dumps(tpr, indent=4))
                                convert_file.write("\n\nFPR:\n")
                                convert_file.write(json.dumps(fpr, indent=4))
                                convert_file.write("\n\nP_R_critical:\n")
                                convert_file.write(json.dumps(P_R_critical, indent=4))
                                convert_file.write("\n\nP_Best_N:\n")
                                convert_file.write(json.dumps(P_best_N, indent=4))
                                convert_file.write("\n\nAverage number of sub-bands used:\n")
                                convert_file.write(json.dumps(resources_used, indent=4))
                                convert_file.write("\n=====================================\n")

                            with open(f'analytic_results_{resource}_{model_prefix}_{scaling_method}_qth_{qth}_nbest.txt', 'a') as convert_file:
                                convert_file.write("\nData configuration:\n")
                                convert_file.write(json.dumps(data_config, indent=4))
                                convert_file.write(f"\nEpochs: {epochs}, qth: {qth}, Number of tests: {number_of_tests}, scaling method: {scaling_method}\n")
                                convert_file.write("\nP_1:\n")
                                convert_file.write(json.dumps(P_1, indent=4))
                                convert_file.write("\nP_inf:\n")
                                convert_file.write(json.dumps(P_inf, indent=4))
                                convert_file.write("\ncdf:\n")
                                convert_file.write(json.dumps(cdf, indent=4))
                                convert_file.write("\nEqth:\n")
                                convert_file.write(json.dumps(Eqth_average_outage_counters, indent=4))
                                convert_file.write("\n=====================================\n")

                        indices_to_average = bubble_sort_indices(P_R_average_outage_counters[model_result])[0:max(number_of_training_routines_per_model-number_to_discard,1)]

                        precision_average_outage_counters[model_result] = mean_of_values_at_indices(precision_average_outage_counters[model_result], indices_to_average)
                        tpr_average_outage_counters[model_result] = mean_of_values_at_indices(tpr_average_outage_counters[model_result], indices_to_average)
                        fpr_average_outage_counters[model_result] = mean_of_values_at_indices(fpr_average_outage_counters[model_result], indices_to_average)

                        P_R_average_outage_counters[model_result] = mean_of_values_at_indices(P_R_average_outage_counters[model_result], indices_to_average)
                        P_R_critical_average_outage_counters[model_result] = mean_of_values_at_indices(P_R_critical_average_outage_counters[model_result], indices_to_average)
                        best_N_average_outage_counters[model_result] = mean_of_values_at_indices(best_N_average_outage_counters[model_result], indices_to_average)
                        P_inf_average_outage_counters[model_result] = mean_of_values_at_indices(P_inf_average_outage_counters[model_result], indices_to_average)
                        average_resources_used[model_result] = mean_of_values_at_indices(average_resources_used[model_result], indices_to_average)
                        P_1_average_outage_counters[model_result] = P_1_average_outage_counters[model_result] / number_of_training_routines_per_model
                        cdf_average_outage_counters[model_result] = mean_of_values_at_indices(cdf_average_outage_counters[model_result], indices_to_average)
                        with open(f'final_results.txt', 'w') as convert_file:
                            convert_file.write("P_1:\n")
                            convert_file.write(json.dumps(P_1_average_outage_counters, indent=4))
                            convert_file.write("\n\nP_R:\n")
                            convert_file.write(json.dumps(P_R_average_outage_counters, indent=4))
                            convert_file.write("\n\nP_R_critical:\n")
                            convert_file.write(json.dumps(P_R_critical_average_outage_counters, indent=4))
                            convert_file.write("\n\nprecision_average_outage_counters:\n")
                            convert_file.write(json.dumps(precision_average_outage_counters, indent=4))
                            convert_file.write("\n\ntpr_average_outage_counters:\n")
                            convert_file.write(json.dumps(tpr_average_outage_counters, indent=4))
                            convert_file.write("\n\nfpr_average_outage_counters:\n")
                            convert_file.write(json.dumps(fpr_average_outage_counters, indent=4))
                            convert_file.write("\n\nP_Best_N:\n")
                            convert_file.write(json.dumps(best_N_average_outage_counters, indent=4))
                            convert_file.write("\n\nP_inf:\n")
                            convert_file.write(json.dumps(P_inf_average_outage_counters, indent=4))
                            convert_file.write("\n\nCDF:\n")
                            convert_file.write(json.dumps(cdf_average_outage_counters, indent=4))
                            convert_file.write("\n\nSub-bands:\n")
                            convert_file.write(json.dumps(average_resources_used, indent=4))
                            convert_file.write("\n\n=====================================\n")
                            convert_file.write("*************************************")
