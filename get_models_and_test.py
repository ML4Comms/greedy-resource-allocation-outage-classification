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
average_outage_counters = {}
average_resources_used = {}
number_of_training_routines_per_model = 10
out = 10
number_of_tests = 5000
qth_range =[0]
# qth_range = [0.8, 0.999] #for testing
# qth_range = [0.9] #for testing
# qth_range= [0.5] #for training
phase_shift = 0.1
#qth_range = [0.00001]
epochs = 20
epoch_size = 150
resources = [4]
model_prefix_names = ["binary_cross_entropy"]
force_retrain_models = True
#batch_size = 32
temperature_value = 0 #used 10 before


for qth in qth_range:
    for lstm_size in [32]:
        for resource in resources:
            for model_prefix in model_prefix_names:
                for rate_threshold in [0.5]:
                    model_result = f"{model_prefix}_rt-{rate_threshold}_r-{resource}_qth--{qth}_lstm-{lstm_size}_out-{out}_phase-{phase_shift}"
                    model_name = model_prefix
                    average_outage_counters[model_result] = 0
                    average_resources_used[model_result] = 0
                    
                    
                    for _ in range(number_of_training_routines_per_model):
                        data_config = {
                                    "taps": 1024,
                                    "padding": 0,
                                    "input_size": 100,
                                    "output_size": out,
                                    "resources": resource,
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

                        lowest_y_pred = float('inf')  # Set to infinity initially so any value will be lower
                        best_of_N_in_outage = False

                        for _ in range(number_of_tests):
                            X, y_label = training_generator.__getitem__(0)
                            Y_pred = multi_lstm_model.predict(X)

                            resource_used = 0
                            
                            should_count = True
                            # Loop over the pre-calibrated outputs
                            for idx, y_pred in enumerate(Y_pred):
                               
                                P_1_counter += 1
                                P_1[model_result] += float(y_label[idx][0])
                                    
                                
                                cdf_counter += 1
                                if y_pred[0] <= qth:
                                    cdf[model_result] += 1.0

                                if y_pred[0] <= qth:
                                    P_inf_counter += 1
                                    P_inf[model_result] += float(y_label[idx][0])
                                
                                resource_used = idx
                                if should_count:
                                    if idx == resource-1:
                                        if y_label[idx][0] >= 0.5:
                                            P_R[model_result] += 1.0
                                            should_count = False
                                    elif y_pred[0] <= qth:
                                        if y_label[idx][0] >= 0.5:
                                            P_R[model_result] += 1.0
                                            should_count = False

                                if y_pred[0] < lowest_y_pred:
                                    lowest_y_pred = y_pred[0]

                                    if y_label[idx][0] >= 0.5:
                                        best_of_N_in_outage = True
                                    else:
                                        best_of_N_in_outage = False

                            if best_of_N_in_outage:
                                P_best_N[model_result] += 1.0

                            resources_used[model_result] += resource_used
                            # print(f"{model_result}, "f"Test {_}:", f"Used sub-band number {resource_used}")
                       
                        P_best_N[model_result] = P_best_N[model_result] / number_of_tests
                        P_R[model_result] = P_R[model_result] / number_of_tests
                        average_outage_counters[model_result]+=P_R[model_result]
                        resources_used[model_result] = resources_used[model_result] / number_of_tests
                        average_resources_used[model_result]+=resources_used[model_result]
                        P_1[model_result] = P_1[model_result] / P_1_counter
                        P_inf[model_result] = (0 if P_inf_counter ==0 else P_inf[model_result] / P_inf_counter)
                        cdf[model_result] = cdf[model_result] / cdf_counter

                        with open(f'simulation_results_{resource}_{model_prefix}_microqth.txt', 'a') as convert_file:
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

                        with open(f'analytic_results_{resource}_{model_prefix}_microqth.txt', 'a') as convert_file:
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

                                                       
                    average_outage_counters[model_result] = average_outage_counters[model_result] / number_of_training_routines_per_model
                    average_resources_used[model_result] = average_resources_used[model_result] / number_of_training_routines_per_model
                    with open(f'final_results_{resource}_{model_prefix}.txt', 'w') as convert_file:
                        convert_file.write("\n\Outage probability:\n")
                        convert_file.write(json.dumps(average_outage_counters, indent=4))
                        convert_file.write("\n\nSub-bands:\n")
                        convert_file.write(json.dumps(average_resources_used, indent=4))
                        convert_file.write("\n\n=====================================\n")
                        convert_file.write("*************************************")
                        convert_file.write("\n=====================================\n\n")

               