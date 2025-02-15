import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from data_generator import OutageData

from toy_models import get_model_and_loss
# load pima indians dataset

model_name = "fin_coef_loss"
out = 10
epochs = 80
epoch_size = 150
validation_epoch_size = epoch_size
rate_threshold = 0.5
phase_shift = 0.1
snr = 1

resource = 4

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

validation_data_config = {
            "taps": 1024,
            "padding": 0,
            "input_size": 100,
            "output_size": out,
            "batch_size": resource,
            "epoch_size": validation_epoch_size,
            "phase_shift": phase_shift,
            "rate_threshold": rate_threshold,
            "snr": snr
        }
training_generator = OutageData(**data_config,)
validate_generator = OutageData(**validation_data_config,)

model_prefix_names = ["fin_coef_loss","binary_cross_entropy"]

filename = f'training_validation_results.txt'

with open(filename, 'a') as convert_file:
    convert_file.write(f"======================================\n")

for model_prefix in model_prefix_names:
    model = get_model_and_loss(data_input=data_config, 
                    model_name=model_name,
                    qth=0.5)
    history = model.fit(training_generator, 
                        validation_data=validate_generator,
                        epochs=epochs)


    # list all data in history
    keys = history.history.keys()
    print(keys)
    recall = next((s for s in keys if s.startswith('val_recall')), None)
    loss = next((s for s in keys if s.startswith('loss')), None)
    val_loss = next((s for s in keys if s.startswith('val_loss')), None)
    precision = next((s for s in keys if s.startswith('val_precision')), None)
    FN = next((s for s in keys if s.startswith('val_false')), None)
    custom_FN_rate = next((s for s in keys if s.startswith('val_custom_FN_rate')), None)

    # summarize history for accuracy
    plt.plot(history.history[loss])
    plt.plot(history.history[val_loss])
    with open(filename, 'a') as convert_file:
        convert_file.write(f"\n\n{model_prefix}:\n")
        convert_file.write(f"{loss}:\n")
        convert_file.write(f"{json.dumps(history.history[loss], indent=4)}\n")
        convert_file.write(f"{val_loss}:\n")
        convert_file.write(f"{json.dumps(history.history[val_loss], indent=4)}\n")
    # plt.plot(history.history['val_accuracy'])
    # plt.show()
    # summarize history for loss
    # plt.plot(history.history[loss])
    # plt.plot(history.history['val_loss'])
plt.title('Training and validation loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(["fin_coef_loss", "val_fin_coef_loss", "binary_cross_entropy", "val_binary_cross_entropy"], loc='upper left')
plt.show()