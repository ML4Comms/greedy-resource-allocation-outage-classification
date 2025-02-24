import numpy as np
import tensorflow as tf
from sklearn.isotonic import IsotonicRegression
from data_generator import OutageData
from outage_loss import InfiniteOutageCoefficientLoss
import json

# Define test parameters
SNRs = [2.0]  
resources = [4]  
lstm_size = 32  
models = ["fin_coef_loss"]  
rate_thresholds = [0.5]  

qth_train_values = [0.5]  
qth_test_values = [0.001]  

num_tests = 10000  

results = {}

# Define `conditional_average()` in the main script
def conditional_average(y_pred, qth):
    y_pred_tensor = tf.convert_to_tensor(y_pred, dtype=tf.float32)

    # Masking values where y_pred <= qth
    mask_y_pred = tf.cast(y_pred_tensor <= qth, tf.float32)
    valid_y_pred = y_pred_tensor * mask_y_pred  
    count_valid_y_pred = tf.reduce_sum(mask_y_pred)  
    sum_valid_y_pred = tf.reduce_sum(valid_y_pred)

    return tf.math.divide_no_nan(sum_valid_y_pred, tf.cast(count_valid_y_pred, tf.float32))

if __name__ == "__main__":
    for snr in SNRs:
        for rate_threshold in rate_thresholds:
            for model_name in models:
                for resource in resources:
                    for qth_train in qth_train_values:  
                        model_path = f"models/{model_name}_snr-{snr}_rt-{rate_threshold}_r-{resource}_qth--{qth_train}_lstm-{lstm_size}_out-{10}_phase-{0.1}"
                        lstm_model = tf.keras.models.load_model(model_path, compile=False)

                        data_config = {
                            "taps": 1024,
                            "padding": 0,
                            "input_size": 100,
                            "output_size": 10,
                            "batch_size": resource,
                            "epoch_size": 150,
                            "phase_shift": 0.1,
                            "rate_threshold": rate_threshold,
                            "snr": snr
                        }
                        test_generator = OutageData(**data_config)

                        model_results = {}

                        for qth_test in qth_test_values:
                            loss_function = InfiniteOutageCoefficientLoss(qth=qth_train)

                            Pinf_values = []
                            Eqth_values = []

                            y_pred_all = []
                            y_true_all = []

                            for _ in range(num_tests):
                                X, y_true = test_generator.__getitem__(0)
                                y_pred = lstm_model.predict(X)

                                y_true_np = y_true.numpy() if isinstance(y_true, tf.Tensor) else y_true
                                y_pred_np = y_pred.numpy() if isinstance(y_pred, tf.Tensor) else y_pred

                                y_true_all.extend(y_true_np.flatten())
                                y_pred_all.extend(y_pred_np.flatten())

                                # Compute Pinf using M()
                                Pinf_values.append(float(loss_function.M(y_true, y_pred).numpy()))

                                # Compute Eqth using `conditional_average()` (NOT from `loss_function`)
                                Eqth_val = float(conditional_average(y_pred_np, qth_train).numpy())
                                Eqth_values.append(Eqth_val if not np.isnan(Eqth_val) else 0)

                            # Apply Isotonic Regression
                            isotonic_regressor = IsotonicRegression(out_of_bounds="clip")
                            isotonic_regressor.fit(y_pred_all, y_true_all)
                            y_pred_iso_all = isotonic_regressor.transform(y_pred_all)

                            Pinf_iso_values = []
                            Eqth_iso_values = []

                            for idx in range(len(y_true_all)):
                                y_pred_iso = y_pred_iso_all[idx]  # Get isotonic-calibrated prediction

                                #  Compute Isotonic Pinf using M() with isotonic-scaled predictions
                                y_pred_iso_tensor = tf.convert_to_tensor([[y_pred_iso]], dtype=tf.float32)
                                y_true_tensor = tf.convert_to_tensor([[y_true_all[idx]]], dtype=tf.float32)

                                Pinf_iso_values.append(float(loss_function.M(y_true_tensor, y_pred_iso_tensor).numpy()))

                            # Compute Isotonic Eqth using `conditional_average()`
                            Eqth_iso_mean = float(conditional_average(y_pred_iso_all, qth_train).numpy())
                            Eqth_iso_mean = Eqth_iso_mean if not np.isnan(Eqth_iso_mean) else 0

                            #  Store Final Results
                            model_results[qth_test] = {
                                "Regular": {
                                    "Pinf": np.mean(Pinf_values),
                                    "Eqth": np.mean(Eqth_values)
                                },
                                "Isotonic": {
                                    "Pinf": np.mean(Pinf_iso_values),
                                    "Eqth": Eqth_iso_mean
                                }
                            }

                            #  Print the Results
                            print(f"Test qth_test={qth_test:.5f} -> Pinf={np.mean(Pinf_values):.4f}, Eqth (qth_train)={np.mean(Eqth_values):.4f}")
                            print(f"Isotonic qth_test={qth_test:.5f} -> Pinf={np.mean(Pinf_iso_values):.4f}, Eqth (qth_train)={Eqth_iso_mean:.4f}")

                        results[f"{model_name}_snr-{snr}_rt-{rate_threshold}_r-{resource}_qth-{qth_train}"] = model_results

    # Save results to JSON file
    with open("pinf_eqth_results.json", "w") as file:
        json.dump(results, file, indent=4)

    print("Results saved in 'pinf_eqth_results.json'")
