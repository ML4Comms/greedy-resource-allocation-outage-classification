import json
import tensorflow as tf
import numpy as np
from data_generator import OutageData
from outage_loss import InfiniteOutageCoefficientLoss
from outage_loss import FiniteOutageCoefficientLoss
import dqn_lstm
import outage_loss
import os


cdf = {}
P_inf = {}
P_1 = {}
P_R = {}
resources_used = {}
average_outage_counters = {}
average_resources_used = {}
number_of_training_routines_per_model = 10  #program runs 10 times, an avg of 10 runs
out = 10 #output samples
number_of_tests = 8000 #set it to 5000
qth_range =[0.5]
phase_shift = 0.1 #change it, see it if works for 0, if it doesn't work set it to very very low
resources = [3] #6,8,10,12,15
loss_names = ["fin_coef_loss","bce","mse","mae"]
num_epochs = 20
batch_size = 32
epoch_size = 150
force_retrain_models = True


for qth in qth_range:
    for resource in resources:
        for loss_name in loss_names:
            for rate_threshold in [2.0]:
                model_result = f"{loss_name}_rt-{rate_threshold}_r-{resources}_qth--{qth}_out-{out}_phase-{phase_shift}"
                model_name = loss_name
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
                    

                    P_R[model_result] = 0.0
                    resources_used[model_result] = 0.0
                       
                    P_inf[model_result] = 0.0
                    cdf[model_result] = 0.0
                    P_1[model_result] = 0.0

                    P_1_counter = 0
                    P_inf_counter = 0
                    cdf_counter = 0
                    
                    dqn_lstm = DQNLSTM(qth,batch_size=batch_size,epochs=num_epochs,data_config=data_config,loss_name=loss_name)
                      
                    for _ in range(number_of_tests):
                        X, y_label = training_generator.__getitem__(0)
                        #dqn_lstm.model.fit(X,y_label)
                        #rewards = dqn_lstm.train_model()                        

                        Y_pred = dqn_lstm.model.predict(X) 

                        resource_used = 0
                            
                        # Loop over the pre-calibrated outputs
                        for idx, y_pred in enumerate(Y_pred):
                            #print(f"Iteration: {idx}")
                            #print("resource_used:", resource_used)
                            #print("y_pred:", y_pred[0])
                            P_1_counter += 1
                            P_1[model_result] += float(y_label[idx][0])
                                    
                                
                            cdf_counter += 1
                            if y_pred[0] <= qth:
                                cdf[model_result] += 1.0

                            if y_pred[0] <= qth:
                                P_inf_counter += 1
                                P_inf[model_result] += float(y_label[idx][0])
                                
                                
                            resource_used = idx
                            if idx == resource-1:
                                if y_label[idx][0] >= 0.5: 
                                    P_R[model_result] += 1.0
                                break
                            elif y_pred[0] <= qth: 
                                if y_label[idx][0] >= 0.5: 
                                    P_R[model_result] += 1.0
                                break
                            else:
                                continue

                        resources_used[model_result] += resource_used
                        #    print(f"{model_result}, "f"Test {_}:", f"Used sub-band number {resource_used}")
                       

                    P_R[model_result] = P_R[model_result] / number_of_tests
                    average_outage_counters[model_result]+=P_R[model_result]
                    resources_used[model_result] = resources_used[model_result] / number_of_tests
                    average_resources_used[model_result]+=resources_used[model_result]
                    P_1[model_result] = P_1[model_result] / P_1_counter
                    P_inf[model_result] = (0 if P_inf_counter ==0 else P_inf[model_result] / P_inf_counter)
                    cdf[model_result] = cdf[model_result] / cdf_counter

                    with open(f'simulation_results_{resource}_{loss_name}_microqth.txt', 'a') as convert_file:
                        convert_file.write("\nData configuration:\n")
                        convert_file.write(json.dumps(data_config, indent=4))
                        convert_file.write(f"\nEpochs: {num_epochs}, qth: {qth}, Number of tests: {number_of_tests}\n")
                        convert_file.write("\n\nP_R:\n")
                        convert_file.write(json.dumps(P_R, indent=4))
                        convert_file.write("\n\nAverage number of sub-bands used:\n")
                        convert_file.write(json.dumps(resources_used, indent=4))
                        convert_file.write("\n=====================================\n")

                    with open(f'analytic_results_{resource}_{loss_name}_microqth.txt', 'a') as convert_file:
                        convert_file.write("\nData configuration:\n")
                        convert_file.write(json.dumps(data_config, indent=4))
                        convert_file.write(f"\nEpochs: {num_epochs}, qth: {qth}, Number of tests: {number_of_tests}\n")
                        convert_file.write("\nP_1:\n")
                        convert_file.write(json.dumps(P_1, indent=4))
                        convert_file.write("\nP_inf:\n")
                        convert_file.write(json.dumps(P_inf, indent=4))
                        convert_file.write("\ncdf:\n")
                        convert_file.write(json.dumps(cdf, indent=4))
                        convert_file.write("\n=====================================\n")

                                                       
                average_outage_counters[model_result] = average_outage_counters[model_result] / number_of_training_routines_per_model
                average_resources_used[model_result] = average_resources_used[model_result] / number_of_training_routines_per_model
                with open(f'final_results_{resource}_{loss_name}.txt', 'w') as convert_file:
                    convert_file.write("\n\Outage probability:\n")
                    convert_file.write(json.dumps(average_outage_counters, indent=4))
                    convert_file.write("\n\nSub-bands:\n")
                    convert_file.write(json.dumps(average_resources_used, indent=4))
                    convert_file.write("\n\n=====================================\n")
                    convert_file.write("*************************************")
                    convert_file.write("\n=====================================\n\n")               
