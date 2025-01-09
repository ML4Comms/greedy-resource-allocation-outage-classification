import cv2
from data_generator import OutageData
import numpy as np
import tensorflow as tf


#####################################
# 2) Stub "Channel Convolution" Function
#####################################
def apply_channel_effect(frame, outage):
    """
    Stub function that pretends to convolve the symbol stream of
    the video frame with the channel response. In reality, you'll
    have domain-specific logic to transform the frame or symbol
    data. For demonstration, we'll slightly adjust brightness.
    """
    # channel_response might be used to do some operation
    # For example, let's scale the frame by some factor derived from channel_response
    scale_factor = 1 if not outage.numpy() == 1 else 0
    frame_out = cv2.convertScaleAbs(frame, alpha=scale_factor, beta=0)
    return frame_out

#####################################
# 3) Load or Define Your Models
#####################################
# In real usage, load your trained models with tf.keras.models.load_model(...)
# or define them in code. Here we create simple stubs for demonstration.

class DummyModel():
    def predict(self, X):
        return np.zeros(len(X))

model_1 = DummyModel()
model_2 = tf.keras.models.load_model("models/bce_demo", compile = False)
model_3 = tf.keras.models.load_model("models/bce_demo", compile = False)
model_4 = tf.keras.models.load_model("models/fin_coef_demo", compile = False)

models = [model_1, model_2, model_3, model_4]
model_names = ["Dummy model", "BCE", "BCE", "FIN COEF"]

#####################################
# 4) Parameters
#####################################
R = 4                       # Number of channel resources
qth = 0.5                   # Example outage threshold
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
qth_range = [0.5]
phase_shift = 0.1
epochs = 3
epoch_size = 10
resources = [4]
rates = [1.5]
model_prefix_names = ["fin_coef_loss"]
force_retrain_models = True
data_config = {
            "taps": 1024,
            "padding": 0,
            "input_size": 100,
            "output_size": out,
            "batch_size": R,
            "epoch_size": epoch_size,
            "phase_shift": phase_shift,
            "rate_threshold": rates[0],
            "snr": SNRs[0]
        }
training_generator = OutageData(**data_config)
VIDEO_PATH = "input_video.mp4"

#####################################
# 5) Main Demo Loop
#####################################
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error opening video file: {VIDEO_PATH}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Prepare 4 output frames, one for each model
        output_frames = []

        for i, model in enumerate(models):
            # For demonstration, we'll just get one batch from the generator
            X, Y = training_generator.__getitem__(0)

            # Attempt to find a good channel among R resources
            # In real usage, you might have R different X's or you might pass the
            # same X but interpret channels differently. We'll just loop R times
            # with different "cuts" of Y for demonstration.
            chosen_channel = R  # Default fallback (the "last" resource)

            op = model.predict(X)  # shape = (1,1) if the model output is single scalar
            
            for r in range(R):
                # Let's pretend each resource has a different slice of X
                # or you might have a separate generator call. 
                # We'll just reuse X here for simplicity.
                # Predict
                op_value = op[r]

                # Check if channel is predicted not to be in outage
                if op_value > qth:
                    # This is the resource we'll use
                    # We'll define chosen_channel based on r
                    break

            # Now "transmit" the frame using the chosen channel
            out_frame = apply_channel_effect(frame, Y[r][0])

            # Put the model name as legend/text in the corner
            cv2.putText(out_frame, model_names[i],
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)

            output_frames.append(out_frame)

        # Now we have 4 frames: output_frames[0..3].
        # Let's arrange them in a 2x2 grid for display:
        # top-left:   model_1
        # top-right:  model_2
        # bottom-left: model_3
        # bottom-right: model_4

        # First ensure they are all the same size
        # (We can just resize them to half their dimension to create a 2x2 grid)
        h, w = output_frames[0].shape[:2]
        new_h, new_w = h // 2, w // 2

        resized = [cv2.resize(f, (new_w, new_h)) for f in output_frames]
        top_row = cv2.hconcat([resized[0], resized[1]])
        bottom_row = cv2.hconcat([resized[2], resized[3]])
        combined = cv2.vconcat([top_row, bottom_row])

        cv2.imshow("4-Model Demo", combined)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
