import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import cv2
from data_generator import OutageData
import numpy as np
import tensorflow as tf
from scipy.special import expit, logit
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from betacal import BetaCalibration


def fit_platt_scaling(y_true, y_pred):
    """Fit Platt scaling (logistic regression on logit of predictions)."""
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

    def neg_log_likelihood(params):
        A, B = params
        p = expit(A * logit(y_pred) + B)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    result = minimize(neg_log_likelihood, x0=[1.0, 0.0], method='Nelder-Mead')
    A, B = result.x
    return A, B


def fit_isotonic_scaling(y_true, y_pred):
    """Fit isotonic regression calibrator."""
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(y_pred, y_true)
    return iso_reg


def collect_calibration_data(lstm_model, generator, samples=50, batch_size=4):
    """Run the model over `samples` batches and collect (y_true, y_pred) lists."""
    all_y_true, all_y_pred = [], []
    for _ in range(samples):
        X, y = generator.__getitem__(0)
        y_pred_vector = lstm_model.predict(X)
        for idx, pred in enumerate(y_pred_vector):
            outage = int(y[idx][0] > qth_test)
            pred_prob = float(pred[0]) if hasattr(pred, '__len__') else float(pred)
            all_y_true.append(outage)
            all_y_pred.append(pred_prob)
    return all_y_true, all_y_pred


class CalibratedModel:
    """Wraps a Keras model and applies post-hoc calibration to its outputs."""

    def __init__(self, base_model, scaling_type, calibration_params):
        self.base_model = base_model
        self.scaling_type = scaling_type
        self.calibration_params = calibration_params  # (A,B) / iso_reg / beta_calibrator

    def predict(self, X):
        raw = self.base_model.predict(X)           # shape (batch,) or (batch,1)
        raw = np.array(raw).flatten()
        raw = np.clip(raw, 1e-7, 1 - 1e-7)

        calibrated = np.empty_like(raw)
        for i, p in enumerate(raw):
            if self.scaling_type == 'platt':
                A, B = self.calibration_params
                calibrated[i] = expit(A * logit(p) + B)
            elif self.scaling_type == 'beta':
                calibrated[i] = self.calibration_params.predict([p])[0]
            elif self.scaling_type == 'isotonic':
                calibrated[i] = self.calibration_params.transform([p])[0]
            else:
                calibrated[i] = p
        return calibrated

def apply_channel_effect(frame, outage):
    """
    Stub function that pretends to convolve the symbol stream of
    the video frame with the channel response. In reality, you'll
    have domain-specific logic to transform the frame or symbol
    data. For demonstration, we'll slightly adjust brightness.
    """
    if outage.numpy() == 1:
        frame_out = np.full_like(frame, (0, 0, 255))  # Red screen (BGR)
    else:
        frame_out = frame.copy()
    return frame_out


snr = 31.62
rt = 2.2
phase = 0.1
lstm_size = 32
R = 4
qth = 0.5       # training threshold 
qth_test = 0.001 # test/decision threshold — used in op_value < qth_test
out = 10

bce_name = f"bce_snr-{snr}_rt-{rt}_r-{R}_qth--{qth}_lstm-{lstm_size}_out-{out}_phase-{phase}"
#mse_name = f"mse_snr-{snr}_rt-{rt}_r-{R}_qth--{qth}_lstm-{lstm_size}_out-{out}_phase-{phase}"
olf_name = f"olf_snr-{snr}_rt-{rt}_r-{R}_qth--{qth}_lstm-{lstm_size}_out-{out}_phase-{phase}"

bce_path = f"models/{bce_name}"
#mse_path = f"models/{mse_name}"
olf_path = f"models/{olf_name}"

print("\nLoading models:")
print(" BCE:", bce_path)
#print(" MSE:", mse_path)
print(" OLF:", olf_path)


class DummyModel():
    def predict(self, X):
        return np.zeros(len(X))


model_1 = DummyModel()
model_2 = tf.keras.models.load_model(bce_path, compile=False)
#model_3 = tf.keras.models.load_model(mse_path, compile=False)
model_4 = tf.keras.models.load_model(olf_path, compile=False)

raw_models = [model_1, model_2, model_4]
model_names = ["Dummy Model", "Binary Cross Entropy", "Outage Loss Function"]


epoch_size = 150
data_config = {
    "taps": 1024,
    "padding": 0,
    "input_size": 100,
    "output_size": out,
    "batch_size": R,
    "epoch_size": epoch_size,
    "phase_shift": phase,
    "rate_threshold": rt,
    "snr": snr
}
training_generator = OutageData(**data_config)

#####################################
# Calibration fitting
calibration_samples = 100  
print("\nSelect calibration method:")
print("  1 - Platt scaling")
print("  2 - Beta scaling")
print("  3 - Isotonic regression")
while True:
    choice = input("Enter choice (1/2/3): ").strip()
    if choice in ('1', '2', '3'):
        break
    print("  Invalid choice, please enter 1, 2 or 3.")

scaling_map = {'1': 'platt', '2': 'beta', '3': 'isotonic'}
scaling_type = scaling_map[choice]
print(f"\nUsing calibration method: {scaling_type}")


def fit_calibrator(scaling_type, y_true, y_pred):
    """Fit and return the calibrator for the chosen scaling type."""
    if scaling_type == 'platt':
        A, B = fit_platt_scaling(y_true, y_pred)
        print(f"  Platt params — A: {A:.4f}, B: {B:.4f}")
        return (A, B)
    elif scaling_type == 'beta':
        beta_cal = BetaCalibration(parameters="abm")
        beta_cal.fit(np.array(y_pred), np.array(y_true))
        print("  Beta calibration fitted.")
        return beta_cal
    elif scaling_type == 'isotonic':
        iso_reg = fit_isotonic_scaling(y_true, y_pred)
        print("  Isotonic regression fitted.")
        return iso_reg


print("\nFitting calibration for BCE...")
y_true_bce, y_pred_bce = collect_calibration_data(model_2, training_generator, samples=calibration_samples, batch_size=R)
cal_model_2 = CalibratedModel(model_2, scaling_type, fit_calibrator(scaling_type, y_true_bce, y_pred_bce))

#print("\nFitting calibration for MSE...")
#y_true_mse, y_pred_mse = collect_calibration_data(model_3, training_generator, samples=calibration_samples, batch_size=R)
#cal_model_3 = CalibratedModel(model_3, scaling_type, fit_calibrator(scaling_type, y_true_mse, y_pred_mse))

print("\nFitting calibration for OLF...")
y_true_olf, y_pred_olf = collect_calibration_data(model_4, training_generator, samples=calibration_samples, batch_size=R)
cal_model_4 = CalibratedModel(model_4, scaling_type, fit_calibrator(scaling_type, y_true_olf, y_pred_olf))

# Calibrated models list (model_1/Dummy has no calibration)
#cal_models = [model_1, cal_model_2, cal_model_3, cal_model_4]
cal_models = [model_1, cal_model_2, cal_model_4]
VIDEO_PATH = "input_video.mp4"
OUTPUT_VIDEO_PATH = "output_demo.mp4" 

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error opening video file: {VIDEO_PATH}")
        return

    # Outage counters for uncalibrated and calibrated models
    outage_counts_uncal = [0] * len(raw_models)
    outage_counts_cal   = [0] * len(cal_models)
    total_frames = 0

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # AFTER
    n_models = len(raw_models)
    new_w_out = (src_w // n_models) * n_models
    new_h_out = src_h // 2 * 2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, src_fps, (new_w_out, new_h_out))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1

        # Prepare 8 output frames: 4 uncalibrated + 4 calibrated
        uncal_frames = []
        cal_frames   = []

        for i in range(len(raw_models)):
            # Get one batch from the generator
            X, Y = training_generator.__getitem__(0)

            # --- Uncalibrated prediction ---
            op_uncal = raw_models[i].predict(X)
            for r_uncal in range(R):
                if op_uncal[r_uncal] < qth_test:
                    break
            outage_flag_uncal = int(Y[r_uncal][0].numpy()) if hasattr(Y[r_uncal][0], "numpy") else int(Y[r_uncal][0])
            outage_counts_uncal[i] += outage_flag_uncal
            outage_pct_uncal = 100.0 * outage_counts_uncal[i] / total_frames

            # --- Calibrated prediction ---
            op_cal = cal_models[i].predict(X)
            for r_cal in range(R):
                if op_cal[r_cal] < qth_test:
                    break
            outage_flag_cal = int(Y[r_cal][0].numpy()) if hasattr(Y[r_cal][0], "numpy") else int(Y[r_cal][0])
            outage_counts_cal[i] += outage_flag_cal
            outage_pct_cal = 100.0 * outage_counts_cal[i] / total_frames

            # --- Build uncalibrated panel ---
            uncal_frame = apply_channel_effect(frame, Y[r_uncal][0])
            # Model name + "Uncalibrated" label
            label_text = f"{model_names[i]} (Uncalibrated)"
            label_font_scale = 1.0
            label_thickness = 2
            (lw, lh), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, label_thickness)
            cv2.rectangle(uncal_frame, (10, 8), (10 + lw + 6, 8 + lh + 8), (0, 0, 0), -1)
            cv2.putText(uncal_frame, label_text, (13, 8 + lh + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, (255, 255, 255), label_thickness, cv2.LINE_AA)
            # Outage %
            pct_text = f"Video Outages: {outage_pct_uncal:.1f}%"
            pct_font_scale = 1.0
            (ow, oh), _ = cv2.getTextSize(pct_text, cv2.FONT_HERSHEY_SIMPLEX, pct_font_scale, label_thickness)
            oy = uncal_frame.shape[0] - 20
            cv2.rectangle(uncal_frame, (10, oy - oh - 4), (10 + ow + 6, oy + 8), (0, 0, 0), -1)
            cv2.putText(uncal_frame, pct_text, (13, oy),
                        cv2.FONT_HERSHEY_SIMPLEX, pct_font_scale, (255, 255, 255), label_thickness, cv2.LINE_AA)
            uncal_frames.append(uncal_frame)

            # --- Build calibrated panel ---
            cal_frame = apply_channel_effect(frame, Y[r_cal][0])
            # Model name + "Calibrated" label
            label_text_cal = f"{model_names[i]} (Calibrated)"
            (lw2, lh2), _ = cv2.getTextSize(label_text_cal, cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, label_thickness)
            cv2.rectangle(cal_frame, (10, 8), (10 + lw2 + 6, 8 + lh2 + 8), (0, 0, 0), -1)
            cv2.putText(cal_frame, label_text_cal, (13, 8 + lh2 + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, (255, 255, 255), label_thickness, cv2.LINE_AA)
            # Outage %
            pct_text_cal = f"Video Outages: {outage_pct_cal:.1f}%"
            (ow2, oh2), _ = cv2.getTextSize(pct_text_cal, cv2.FONT_HERSHEY_SIMPLEX, pct_font_scale, label_thickness)
            oy2 = cal_frame.shape[0] - 20
            cv2.rectangle(cal_frame, (10, oy2 - oh2 - 4), (10 + ow2 + 6, oy2 + 8), (0, 0, 0), -1)
            cv2.putText(cal_frame, pct_text_cal, (13, oy2),
                        cv2.FONT_HERSHEY_SIMPLEX, pct_font_scale, (255, 255, 255), label_thickness, cv2.LINE_AA)
            cal_frames.append(cal_frame)

        h, w = uncal_frames[0].shape[:2]
        n_models = len(raw_models)
        new_h, new_w = h // 2, w // n_models

        uncal_resized = [cv2.resize(f, (new_w, new_h)) for f in uncal_frames]
        cal_resized   = [cv2.resize(f, (new_w, new_h)) for f in cal_frames]
        top_row    = cv2.hconcat(uncal_resized)
        bottom_row = cv2.hconcat(cal_resized)
        combined   = cv2.vconcat([top_row, bottom_row])

        # Added: save frame to output video
        writer.write(combined)

        cv2.imshow("4-Model Demo", combined)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()  # Added
    cv2.destroyAllWindows()

    # Final outage percentages over the whole video
    print("\nAverage outage rate over entire video duration:")
    for i, name in enumerate(model_names):
        pct_uncal = 100.0 * outage_counts_uncal[i] / total_frames if total_frames > 0 else 0.0
        pct_cal   = 100.0 * outage_counts_cal[i]   / total_frames if total_frames > 0 else 0.0
        print(f"  {name}:")
        print(f"    Uncalibrated: {pct_uncal:.2f}%")
        print(f"    Calibrated:   {pct_cal:.2f}%")


if __name__ == "__main__":
    main()