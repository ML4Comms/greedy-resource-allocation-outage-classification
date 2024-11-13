#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import tensorflow as tf
import numpy as np
from data_generator import OutageData
from outage_loss import InfiniteOutageCoefficientLoss, TPR, FPR, Precision
import dqn_lstm
import toy_models
from dqn_lstm import DQNLSTM

def bubble_sort_indices(arr):
    n = len(arr)
    indices = list(range(n))
    for i in range(n-1):
        for j in range(0, n-i-1):
            # Swap if the element found is greater
            # than the next element
            if arr[indices[j]] > arr[indices[j+1]]:
                indices[j], indices[j+1] = indices[j+1], indices[j]
    return indices

def mean_of_values_at_indices(arr, indices):
    # Summing values at provided indices
    total = sum(arr[i] for i in indices)
    # Compute the mean
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
number_of_training_routines_per_model = 1
number_to_discard = 0
out = 10
number_of_tests = 6
SNRs = [10.0]#FOR SNR =1 its 2000; for 8 its 6000
qth_range = [0.1]
phase_shift = 0.1
epochs = 3
epoch_size = 10
resources = [4]
rates = [0.5]
model_prefix_names = ["fin_coef_loss"]
force_retrain_models = True
temperature_value = 0 #used 10 before
# Prompt for model type once
use_model = input("Press 1 to use LSTM, any other key for DQN-LSTM: ")


for snr in SNRs:
    for qth in qth_range:
        for lstm_size in [32]:
            for resource in resources:
                for model_prefix in model_prefix_names:
                    for rate_threshold in rates:
                        model_result = f"{model_prefix}_snr-{snr}_rt-{rate_threshold}_r-{resource}_qth--{qth}_lstm-{lstm_size}_out-{out}_phase-{phase_shift}"
                        model_name = f"{model_prefix}_snr-{snr}_rt-{rate_threshold}_r-{resource}_qth--{qth}_lstm-{lstm_size}_out-{out}_phase-{phase_shift}"

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
                                        "snr": snr
                                    }
                            
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
                            
                            
                            dqn_lstm = DQNLSTM(qth,epochs=epochs,data_config=data_config,model_name=model_name,lstm_units=lstm_size)


                            for _ in range(number_of_tests):
                                X, y_label = training_generator.__getitem__(0)
                                Y_pred = model.predict(X)

                                resource_used = 0
                                
                                should_count = True
                                lowest_y_pred = float('inf')  # Set to infinity initially so any value will be lower
                                best_of_N_in_outage = False
                                idx_of_best = 0
                                # Loop over the pre-calibrated outputs
                                tpr[model_result] += TPR(qth=qth, y_pred=Y_pred, y_true=y_label).numpy()
                                fpr[model_result] += FPR(qth=qth, y_pred=Y_pred, y_true=y_label).numpy()
                                precision[model_result] += Precision(qth=qth, y_pred=Y_pred, y_true=y_label).numpy()
                                for idx, y_pred in enumerate(Y_pred):
                                
                                    # P1 calculation
                                    P_1_counter += 1
                                    P_1[model_result] += float(y_label[idx][0])
                                        
                                    # P_inf and CDF calculation
                                    cdf_counter += 1
                                    if y_pred[0] <= qth:
                                        cdf[model_result] += 1.0
                                        
                                        P_inf_counter += 1
                                        P_inf[model_result] += float(y_label[idx][0])
                                    
                                    # P_best_N calculation
                                    if y_pred[0] < lowest_y_pred:
                                        lowest_y_pred = y_pred[0]
                                        idx_of_best = idx
                                        if y_label[idx][0] >= 0.5:
                                            best_of_N_in_outage = True
                                        else:
                                            best_of_N_in_outage = False

                                    # P_R_critical and P_R
                                    if should_count:
                                        if idx == resource-1:
                                            resource_used = idx_of_best
                                            P_R_critical[model_result] += float(best_of_N_in_outage)
                                        elif y_pred[0] <= qth:
                                            resource_used = idx
                                            should_count = False
                                            if y_label[idx][0] >= 0.5:
                                                P_R_critical[model_result] += 1.0
                                        if(idx == resource-1 or y_pred[0] <= qth):
                                            resource_used = idx
                                            should_count = False
                                            if y_label[idx][0] >= 0.5:
                                                P_R[model_result] += 1.0


                                if best_of_N_in_outage:
                                    P_best_N[model_result] += 1.0

                                resources_used[model_result] += resource_used
                                # print(f"{model_result}, "f"Test {_}:", f"Used sub-band number {resource_used}")
                        
                            P_best_N[model_result] = P_best_N[model_result] / number_of_tests
                            best_N_average_outage_counters[model_result].append(P_best_N[model_result])

                            P_R[model_result] = P_R[model_result] / number_of_tests
                            P_R_average_outage_counters[model_result].append(P_R[model_result])

                            tpr[model_result] = tpr[model_result] / number_of_tests
                            tpr_average_outage_counters[model_result].append(tpr[model_result])

                            fpr[model_result] = fpr[model_result] / number_of_tests
                            fpr_average_outage_counters[model_result].append(fpr[model_result])

                            precision[model_result] = precision[model_result] / number_of_tests
                            precision_average_outage_counters[model_result].append(precision[model_result])

                            P_R_critical[model_result] = P_R_critical[model_result] / number_of_tests
                            P_R_critical_average_outage_counters[model_result].append(P_R_critical[model_result])

                            resources_used[model_result] = resources_used[model_result] / number_of_tests
                            average_resources_used[model_result].append(resources_used[model_result])

                            P_1[model_result] = P_1[model_result] / P_1_counter
                            P_1_average_outage_counters[model_result] += P_1[model_result]

                            P_inf[model_result] = (0 if P_inf_counter ==0 else P_inf[model_result] / P_inf_counter)
                            P_inf_average_outage_counters[model_result].append(P_inf[model_result])

                            cdf[model_result] = cdf[model_result] / cdf_counter
                            cdf_average_outage_counters[model_result].append(cdf[model_result])

                            with open(f'simulation_results_{resource}_{model_prefix}_nbest.txt', 'a') as convert_file:
                                convert_file.write("\nData configuration:\n")
                                convert_file.write(json.dumps(data_config, indent=4))
                                convert_file.write(f"\nEpochs: {epochs}, qth: {qth}, Number of tests: {number_of_tests}\n")
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

                            with open(f'analytic_results_{resource}_{model_prefix}_nbest.txt', 'a') as convert_file:
                                convert_file.write("\nData configuration:\n")
                                convert_file.write(json.dumps(data_config, indent=4))
                                convert_file.write(f"\nEpochs: {epochs}, qth: {qth}, Number of tests: {number_of_tests}\n")
                                convert_file.write("\nP_1:\n")
                                convert_file.write(json.dumps(P_1, indent=4))
                                convert_file.write("\nP_inf:\n")
                                convert_file.write(json.dumps(P_inf, indent=4))
                                convert_file.write("\ncdf:\n")
                                convert_file.write(json.dumps(cdf, indent=4))
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
                        P_1_average_outage_counters[model_result] = P_1_average_outage_counters[model_result] / number_of_training_routines_per_model # P1 doesn't depend on model so no need to remove anomolies
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
                            convert_file.write("\n=====================================\n\n")


# In[ ]:




