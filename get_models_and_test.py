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
resources_used = {}
average_outage_counters = {}
average_resources_used = {}
number_of_training_routines_per_model = 1
out = 10
number_of_tests = 6000
# qth_range =[0.25, 0.35, 0.45, 0.55]
# qth_range = [0.8, 0.999] #for testing
# qth_range = [0.1, 0.3, 0.5, 0.7, 0.9] #for testing
qth_range= [0.5] #for training
phase_shift = 0.1
#qth_range = [0.00001]
epochs = 15
epoch_size = 100
resources = [10]
model_prefix_names = ["fin_coef_loss","binary_cross_entropy"]
force_retrain_models = False
temperature_value = 0 #used 10 before


for qth in qth_range:
    for lstm_size in [32]:
        for resource in resources:
            for model_prefix in model_prefix_names:
                for rate_threshold in [1.05]:
                    model_name = f"{model_prefix}_rt-{rate_threshold}_r-{resource}_lstm-{lstm_size}_out-{out}_phase-{phase_shift}"
                    average_outage_counters[model_name] = 0
                    average_resources_used[model_name] = 0
                    
                    
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

                        P_R[model_name] = 0.0
                        resources_used[model_name] = 0.0
                       
                        P_inf[model_name] = 0.0
                        cdf[model_name] = 0.0
                        P_1[model_name] = 0.0

                        P_1_counter = 0
                        P_inf_counter = 0
                        cdf_counter = 0
                        
                        
                        multi_lstm_model = toy_models.get_fitted_model(data_input=data_config, 
                                                                        model_name=model_name, 
                                                                        epochs=epochs, 
                                                                        force_retrain = force_retrain_models, 
                                                                        lstm_units=lstm_size,
                                                                        qth=0.5)

                        for _ in range(number_of_tests):
                            X, y_label = training_generator.__getitem__(0)
                            Y_pred = multi_lstm_model.predict(X)

                            resource_used = 0
                            
                            # Loop over the pre-calibrated outputs
                            for idx, y_pred in enumerate(Y_pred):
                               
                                P_1_counter += 1
                                P_1[model_name] += float(y_label[idx][0])
                                    
                                
                                cdf_counter += 1
                                if y_pred[0] <= qth:
                                    cdf[model_name] += 1.0

                                if y_pred[0] <= qth:
                                    P_inf_counter += 1
                                    P_inf[model_name] += float(y_label[idx][0])
                                
                                
                                resource_used = idx
                                if idx == resource-1:
                                    if y_label[idx][0] >= 0.5:
                                        P_R[model_name] += 1.0
                                    break
                                elif y_pred[0] <= qth:
                                    if y_label[idx][0] >= 0.5:
                                        P_R[model_name] += 1.0
                                    break
                                else:
                                    continue

                            resources_used[model_name] += resource_used
                            # print(f"{model_name}, "f"Test {_}:", f"Used sub-band number {resource_used}")
                       

                        P_R[model_name] = P_R[model_name] / number_of_tests
                        resources_used[model_name] = resources_used[model_name] / number_of_tests
                        P_1[model_name] = P_1[model_name] / P_1_counter
                        P_inf[model_name] = (0 if P_inf_counter ==0 else P_inf[model_name] / P_inf_counter)
                        cdf[model_name] = cdf[model_name] / cdf_counter

                        with open(f'simulation_results_{resource}_{model_prefix}_microqth.txt', 'a') as convert_file:
                            convert_file.write("\nData configuration:\n")
                            convert_file.write(json.dumps(data_config, indent=4))
                            convert_file.write(f"\nEpochs: {epochs}, qth: {qth_range}, Number of tests: {number_of_tests}\n")
                            convert_file.write("\n\nP_R:\n")
                            convert_file.write(json.dumps(P_R, indent=4))
                            convert_file.write("\n\nAverage number of sub-bands used:\n")
                            convert_file.write(json.dumps(resources_used, indent=4))
                            convert_file.write("\n=====================================\n")

                        with open(f'analytic_results_{resource}_{model_prefix}_microqth.txt', 'a') as convert_file:
                            convert_file.write("\nData configuration:\n")
                            convert_file.write(json.dumps(data_config, indent=4))
                            convert_file.write(f"\nEpochs: {epochs}, qth: {qth_range}, Number of tests: {number_of_tests}\n")
                            convert_file.write("\nP_1:\n")
                            convert_file.write(json.dumps(P_1, indent=4))
                            convert_file.write("\nP_inf:\n")
                            convert_file.write(json.dumps(P_inf, indent=4))
                            convert_file.write("\ncdf:\n")
                            convert_file.write(json.dumps(cdf, indent=4))
                            convert_file.write("\n=====================================\n")

                                                       
                    average_outage_counters[model_name] = average_outage_counters[model_name] / number_of_training_routines_per_model
                    average_resources_used[model_name] = average_resources_used[model_name] / number_of_training_routines_per_model
                    with open(f'final_results_{resource}_{model_prefix}_microqth.txt', 'w') as convert_file:
                        convert_file.write("\n\Outage probability:\n")
                        convert_file.write(json.dumps(average_outage_counters, indent=4))
                        convert_file.write("\n\nSub-bands:\n")
                        convert_file.write(json.dumps(average_resources_used, indent=4))
                        convert_file.write("\n\n=====================================\n")
                        convert_file.write("*************************************")
                        convert_file.write("\n=====================================\n\n")

               