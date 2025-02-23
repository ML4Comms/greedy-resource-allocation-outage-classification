import numpy as np
import tensorflow as tf
from sklearn.isotonic import IsotonicRegression
from data_generator import OutageData
from outage_loss import InfiniteOutageCoefficientLoss
import json

# Define parameters
SNRs = [2.0]
resources = [4]
lstm_size = 32
models = ["fin_coef_loss"]
rate_thresholds = [0.5]

qth_train_values = [0.5]  
qth_test_values = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5]

num_tests = 10000  # Number of test runs


results = {}

if __name__ == "__main__":
    for snr in SNRs:
        for rt in rate_thresholds:
            for model in models:
                for resource in resources:
                    for qth_train in qth_train_values:  
                        path_name = f"models/{model}_snr-{snr}_rt-{rt}_r-{resource}_qth--{qth_train}_lstm-{lstm_size}_out-{10}_phase-{0.1}"
                        dqn_lstm_model = tf.keras.models.load_model(path_name, compile=False)

                        # Create data generator
                        inputs = {
                            "taps": 1024,
                            "padding": 0,
                            "input_size": 100,
                            "output_size": 10,
                            "batch_size": resource,
                            "epoch_size": 150,
                            "phase_shift": 0.1,
                            "rate_threshold": rt,
                            "snr": snr
                        }
                        training_generator = OutageData(**inputs)

                        model_results = {}

                        for qth_test in qth_test_values:
                            # Initialize loss function with qth_test
                            loss_function = InfiniteOutageCoefficientLoss(qth=qth_test)

                            Pinf_values = []
                            Eqth_values = []

                            # Variables for isotonic scaling
                            y_pred_all = []
                            y_true_all = []

                        
                            for _ in range(num_tests):
                                X, y_true = training_generator.__getitem__(0)
                                y_pred = dqn_lstm_model.predict(X)

                                # Convert to NumPy (only if it's a tensor)
                                y_true_np = y_true.numpy() if isinstance(y_true, tf.Tensor) else y_true
                                y_pred_np = y_pred.numpy() if isinstance(y_pred, tf.Tensor) else y_pred

                                # Store predictions for isotonic regression
                                y_true_all.extend(y_true_np.flatten())
                                y_pred_all.extend(y_pred_np.flatten())

                                # Compute regular Pinf (M) and Eqth
                                Pinf = loss_function.M(y_true, y_pred).numpy()  
                                Eqth = np.mean(y_pred_np[y_pred_np <= qth_test]) if np.sum(y_pred_np <= qth_test) > 0 else 0.0  

                                Pinf_values.append(Pinf)
                                Eqth_values.append(Eqth)

                            # Train isotonic regression once after collecting all predictions
                            isotonic_regressor = IsotonicRegression(out_of_bounds="clip")
                            isotonic_regressor.fit(y_pred_all, y_true_all) 
                            
                            #  Single pass for isotonic-scaled Pinf & Eqth
                            Pinf_iso_values = []
                            Eqth_iso_values = []

                            # Apply isotonic regression to all stored predictions
                            y_pred_iso_all = isotonic_regressor.transform(y_pred_all)

                            for i in range(len(y_pred_all)):
                                y_pred_iso_tensor = tf.convert_to_tensor([[y_pred_iso_all[i]]], dtype=tf.float32)
                                y_true_tensor = tf.convert_to_tensor([[y_true_all[i]]], dtype=tf.float32)

                                # Compute isotonic Pinf & Eqth
                                Pinf_iso = loss_function.M(y_true_tensor, y_pred_iso_tensor).numpy()  
                                Eqth_iso = np.mean(y_pred_iso_all[y_pred_iso_all <= qth_test]) if np.sum(y_pred_iso_all <= qth_test) > 0 else 0.0  

                                Pinf_iso_values.append(Pinf_iso)
                                Eqth_iso_values.append(Eqth_iso)

                            model_results[qth_test] = {
                                "Regular": {
                                    "Pinf": float(np.mean(Pinf_values)),
                                    "Eqth": float(np.mean(Eqth_values))
                                },
                                "Isotonic": {
                                    "Pinf": float(np.mean(Pinf_iso_values)),
                                    "Eqth": float(np.mean(Eqth_iso_values))
                                }
                            }

                            print(f"Test qth={qth_test:.5f} -> Pinf={np.mean(Pinf_values):.4f}, Eqth={np.mean(Eqth_values):.4f}")
                            print(f"Isotonic qth={qth_test:.5f} -> Pinf={np.mean(Pinf_iso_values):.4f}, Eqth={np.mean(Eqth_iso_values):.4f}")

                        # Save model results (include qth_train in key)
                        results[f"{model}_snr-{snr}_rt-{rt}_r-{resource}_qth-{qth_train}"] = model_results

    # Save results to a file (now JSON serializable)
    with open("pinf_eqth_results.json", "w") as file:
        json.dump(results, file, indent=4)

    print("Results saved in 'pinf_eqth_results.json'")
