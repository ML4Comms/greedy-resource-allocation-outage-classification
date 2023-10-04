import json
import tensorflow as tf
import numpy as np
from data_generator import OutageData
from outage_loss import InfiniteOutageCoefficientLoss
import toy_models



cdf = {}
P_inf = {}
P_1 = {}
P_R = {}
P_best_N = {}
resources_used = {}
P_R_average_outage_counters = {}
best_N_average_outage_counters = {}
P_1_average_outage_counters = {}
P_inf_average_outage_counters = {}
cdf_average_outage_counters = {}
average_resources_used = {}
number_of_training_routines_per_model = 8
out = 10
number_of_tests = 6000
#qth_range =[0]
# qth_range = [0.8, 0.999] #for testing
# qth_range = [0.9] #for testing
qth_range= [0.5] #for training
phase_shift = 0.1
#qth_range = [0.00001]
epochs = 40
epoch_size = 200
resources = [4]
model_prefix_names = ["fin_coef_loss","binary_cross_entropy","mse"]
force_retrain_models = True
temperature_value = 0 #used 10 before


for qth in qth_range:
    for lstm_size in [32]:
        for resource in resources:
            for model_prefix in model_prefix_names:
                for rate_threshold in [0.2]:
                    model_result = f"{model_prefix}_rt-{rate_threshold}_r-{resource}_qth--{qth}_lstm-{lstm_size}_out-{out}_phase-{phase_shift}"
                    model_name = model_prefix
                    P_R_average_outage_counters[model_result] = 0
                    average_resources_used[model_result] = 0
                    best_N_average_outage_counters[model_result] = 0
                    P_inf_average_outage_counters[model_result] = 0
                    P_1_average_outage_counters[model_result] = 0
                    cdf_average_outage_counters[model_result] = 0
                    
                    
                    for _ in range(number_of_training_routines_per_model):
                        data_config = {
                                    "taps": 1024,
                                    "padding": 0,
                                    "input_size": 100,
                                    "output_size": out,
                                    "batch_size": resource,
                                    "epoch_size": epoch_size,
                                    "phase_shift": phase_shift,
                                    "rate_threshold": rate_threshold
                                }
                        training_generator = OutageData(**data_config,)

                        P_best_N[model_result] = 0.0
                        P_R[model_result] = 0.0
                        resources_used[model_result] = 0.0
                       
                        P_inf[model_result] = 0.0
                        cdf[model_result] = 0.0
                        P_1[model_result] = 0.0

                        P_1_counter = 0
                        P_inf_counter = 0
                        cdf_counter = 0
                        
                        
                        multi_lstm_model = toy_models.get_fitted_model(data_input=data_config, 
                                                                        model_name=model_name, 
                                                                        epochs=epochs, 
                                                                        force_retrain = force_retrain_models, 
                                                                        lstm_units=lstm_size,
                                                                        qth=qth)


                        for _ in range(number_of_tests):
                            X, y_label = training_generator.__getitem__(0)
                            Y_pred = multi_lstm_model.predict(X)

                            resource_used = 0
                            
                            should_count = True
                            lowest_y_pred = float('inf')  # Set to infinity initially so any value will be lower
                            best_of_N_in_outage = False
                            idx_of_best = 0
                            # Loop over the pre-calibrated outputs
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

                                # P_R
                                if should_count:
                                    if idx == resource-1:
                                        resource_used = idx_of_best
                                        P_R[model_result] += int(best_of_N_in_outage)
                                    elif y_pred[0] <= qth:
                                        resource_used = idx
                                        should_count = False
                                        if y_label[idx][0] >= 0.5:
                                            P_R[model_result] += 1.0


                            if best_of_N_in_outage:
                                P_best_N[model_result] += 1.0

                            resources_used[model_result] += resource_used
                            # print(f"{model_result}, "f"Test {_}:", f"Used sub-band number {resource_used}")
                       
                        P_best_N[model_result] = P_best_N[model_result] / number_of_tests
                        best_N_average_outage_counters[model_result]+=P_best_N[model_result]

                        P_R[model_result] = P_R[model_result] / number_of_tests
                        P_R_average_outage_counters[model_result]+=P_R[model_result]

                        resources_used[model_result] = resources_used[model_result] / number_of_tests
                        average_resources_used[model_result]+=resources_used[model_result]

                        P_1[model_result] = P_1[model_result] / P_1_counter
                        P_1_average_outage_counters[model_result] += P_1[model_result]

                        P_inf[model_result] = (0 if P_inf_counter ==0 else P_inf[model_result] / P_inf_counter)
                        P_inf_average_outage_counters[model_result] += P_inf[model_result]

                        cdf[model_result] = cdf[model_result] / cdf_counter
                        cdf_average_outage_counters[model_result]+=cdf[model_result]

                        with open(f'simulation_results_{resource}_{model_prefix}_nbest.txt', 'a') as convert_file:
                            convert_file.write("\nData configuration:\n")
                            convert_file.write(json.dumps(data_config, indent=4))
                            convert_file.write(f"\nEpochs: {epochs}, qth: {qth}, Number of tests: {number_of_tests}\n")
                            convert_file.write("\n\nP_R:\n")
                            convert_file.write(json.dumps(P_R, indent=4))
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

                                                       
                    P_R_average_outage_counters[model_result] = P_R_average_outage_counters[model_result] / number_of_training_routines_per_model
                    best_N_average_outage_counters[model_result] = best_N_average_outage_counters[model_result] / number_of_training_routines_per_model
                    P_inf_average_outage_counters[model_result] = P_inf_average_outage_counters[model_result] / number_of_training_routines_per_model
                    average_resources_used[model_result] = average_resources_used[model_result] / number_of_training_routines_per_model
                    P_1_average_outage_counters[model_result] = P_1_average_outage_counters[model_result] / number_of_training_routines_per_model
                    cdf_average_outage_counters[model_result] = cdf_average_outage_counters[model_result] / number_of_training_routines_per_model
                    with open(f'final_results.txt', 'w') as convert_file:
                        convert_file.write("P_1:\n")
                        convert_file.write(json.dumps(P_1_average_outage_counters, indent=4))
                        convert_file.write("\n\nP_R:\n")
                        convert_file.write(json.dumps(P_R_average_outage_counters, indent=4))
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

               