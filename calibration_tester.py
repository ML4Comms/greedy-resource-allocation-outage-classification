import json
import tensorflow as tf
from data_generator import OutageData
import numpy as np

import matplotlib.pyplot as plt


import code, traceback, signal


def debug(sig, frame):
    """Interrupt running process, and provide a python prompt for
    interactive debugging."""
    d={'_frame':frame}         # Allow access to frame object.
    d.update(frame.f_globals)  # Unless shadowed by global
    d.update(frame.f_locals)

    i = code.InteractiveConsole(d)
    message  = "Signal received : entering python shell.\nTraceback:\n"
    message += ''.join(traceback.format_stack(frame))
    i.interact(message)

out=10
number_of_bins = 10
qth_range = [0.1, 0.3, 0.5, 0.7, 0.9]
#qth_range = [0.3]
samples = 10000
temperature_value = 10 #used 10 before


def get_bin_index(probability, number_of_bins):
    answer = int(probability * number_of_bins)
    if answer == number_of_bins:
        answer -= 1
    return int(answer)



if __name__ == "__main__":
    for resources in [128]:
        for lstm_size in [32]:
            for model in ["binary_cross_entropy"]:
                for rt in [1.05]:
                    for qth in qth_range:
                        accuracy = np.zeros(number_of_bins)
                        confdistribution = np.zeros(number_of_bins)
                        count = np.zeros(number_of_bins)
                        accuracy_temp = np.zeros(number_of_bins)
                        confdistribution_temp = np.zeros(number_of_bins)
                        count_temp = np.zeros(number_of_bins)
                        path_name = f"models/{model}_rt-{rt}_b-{resources}_lstm-32_qth-{qth}"
                        inputs = {
                                                    "taps": 1024,
                                                    "padding": 0,
                                                    "input_size": 100,
                                                    "output_size": out,
                                                    "batch_size": resources,
                                                    "epoch_size": 1,
                                                    "phase_shift": 0.1,
                                                    "rate_threshold": rt
                                                }
                        training_generator = OutageData(**inputs,)
                        multi_lstm_model = tf.keras.models.load_model(path_name, compile = False)
                        temperature = np.loadtxt(f"{path_name}/temperature.txt")
                        for _ in range(samples):
                            print(_)
                            X, y = training_generator.__getitem__(0)
                            y_pred_vector = multi_lstm_model.predict(X)
                            y_pred_vector_with_temp = y_pred_vector / (temperature_value if temperature_value > 0 else temperature)

                            y_pred_vector = tf.nn.softmax(y_pred_vector, axis = -1)
                            output_with_temp = tf.nn.softmax(y_pred_vector_with_temp, axis = -1)
                            print("y_pred_vector", y_pred_vector)
                            print("output_with_temp", output_with_temp)

                            for idx, y_pred in enumerate(y_pred_vector):
                                success = 0
                                estimated_success_prob = y_pred[1]
                                estimated_success_prob_temp = output_with_temp[idx][1]
                                if y[idx][1] > 0.5:
                                    success = 1
                                else:
                                    pass
                                bin_index = get_bin_index(estimated_success_prob, number_of_bins)
                                bin_index_temp = get_bin_index(estimated_success_prob_temp, number_of_bins)
                                print("estimated_success_prob:", estimated_success_prob)
                                print("binned_success_probability:", float(bin_index) / number_of_bins)
                                count[bin_index] += 1
                                accuracy[bin_index] += success
                                confdistribution[bin_index:] += 1 
                                
                                count_temp[bin_index_temp] += 1
                                accuracy_temp[bin_index_temp] += success
                                confdistribution_temp[bin_index_temp:] += 1 

                        accuracy = np.divide(accuracy, count)
                        accuracy_temp = np.divide(accuracy_temp, count_temp)
                        confdistribution = np.divide(confdistribution, confdistribution[-1])
                        confdistribution_temp = np.divide(confdistribution_temp, confdistribution_temp[-1])
                        print(f"\n\nResources: {resources} Model: {model} Rate: {rt} qth: {qth}\n")

                        with open('ac_results_new.txt', 'a') as file1:
                            file1.write(f"\n\nResources: {resources} Model: {model} Rate: {rt} qth: {qth}\n")
                            np.savetxt(file1, accuracy, newline=",")

                        with open('conf_distribution_results_new.txt', 'a') as file2:
                            file2.write(f"\n\nResources: {resources} Model: {model} Rate: {rt} qth: {qth}\n")
                            np.savetxt(file2, confdistribution, newline=",")

                        with open('ac_temp_results_new.txt', 'a') as file1:
                            file1.write(f"\n\nResources: {resources} Model: {model} Rate: {rt} qth: {qth}\n")
                            np.savetxt(file1, accuracy_temp, newline=",")

                        with open('conf_temp_distribution_results_new.txt', 'a') as file2:
                            file2.write(f"\n\nResources: {resources} Model: {model} Rate: {rt} qth: {qth}\n")
                            np.savetxt(file2, confdistribution_temp, newline=",")
                            
