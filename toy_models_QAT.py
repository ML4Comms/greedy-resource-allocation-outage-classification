# toy_models.py

import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from data_generator import OutageData
from outage_loss import InfiniteOutageCoefficientLoss, FiniteOutageCoefficientLoss

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from betacal import BetaCalibration
from scipy.special import logit


try:
    import brevitas
    from brevitas.nn import QuantIdentity, QuantLinear, QuantReLU, QuantSigmoid
    from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat
    _BREVITAS_OK = True
except Exception:
    QuantIdentity = QuantLinear = QuantReLU = QuantSigmoid = None
    _BREVITAS_OK = False

def _warmup_quant(model, sample_X, mode: str):
    if mode != "qat_int8":
        return
    model.train()
    with torch.no_grad():
        if isinstance(sample_X, np.ndarray):
            sample_X = torch.from_numpy(sample_X).float()
        _ = model(sample_X)  # run one forward to materialize quant buffers

def _verify_qat_implementation(model, stage: str):
    """Verify QAT implementation at different stages"""
    if not any(quant_type in str(type(module)) for name, module in model.named_modules() for quant_type in ['Quant', 'quant']):
        return
    
    print(f"\n" + "="*50)
    print(f"QAT VERIFICATION - {stage}")
    print("="*50)
    
    quant_layers = []
    for name, module in model.named_modules():
        if any(quant_type in str(type(module)) for quant_type in ['Quant', 'quant']):
            quant_layers.append((name, module))
            
            # Get bit width info
            bit_info = []
            if hasattr(module, 'weight_bit_width'):
                bit_info.append(f"weights: {module.weight_bit_width}bit")
            if hasattr(module, 'act_bit_width'):
                bit_info.append(f"activations: {module.act_bit_width}bit")
            if hasattr(module, 'bias_bit_width'):
                bit_info.append(f"bias: {module.bias_bit_width}bit")
            
            bit_str = f" ({', '.join(bit_info)})" if bit_info else ""
            print(f"  {name}: {type(module).__name__}{bit_str}")

    print(f"\nSUMMARY: Found {len(quant_layers)} quantized layers")
    if len(quant_layers) == 0:
        print(" CRITICAL: No quantized layers detected!")
        print("   Possible causes:")
        print("   - Brevitas not installed properly")
        print("   - Using wrong model class")
        print("   - Quantization layers not being used")
    else:
        print(" QAT is correctly implemented with INT8 quantization")
    print("="*50 + "\n")

class DummyModel:
    def predict(self, X):
        return np.zeros(len(X))


class TemperatureModel(nn.Module):
    """FP32/FP16 model that returns probabilities (sigmoid in forward)."""
    def __init__(self, lstm_units, output_size, input_size):
        super().__init__()
        self.lstm_units = lstm_units
        self.input_size = input_size

        self.layer1 = nn.LSTM(input_size=input_size, hidden_size=lstm_units, batch_first=True)
        self.layer2 = nn.Linear(lstm_units, output_size)
        self.relu = nn.ReLU()
        #self.relu = nn.PReLU()
        self.layer3 = nn.Linear(output_size, 1)
        self.sigmoid = nn.Sigmoid()

        # He initialization
        nn.init.kaiming_uniform_(self.layer2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.layer3.weight, nonlinearity='sigmoid')

    def forward(self, inputs):
        x, _ = self.layer1(inputs)
        x = x[:, -1, :]  # Get the last time step
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        x = self.sigmoid(x)
        return x

    def predict(self, inputs):
        self.eval()
        with torch.no_grad():
            if isinstance(inputs, np.ndarray):
                inputs = torch.FloatTensor(inputs)
            outputs = self.forward(inputs)
            return outputs.numpy()


class WorkingQuantTemperatureModel(nn.Module):
    """Working INT8 QAT version with proper Brevitas configuration."""
    def __init__(self, lstm_units, output_size, input_size):
        super().__init__()
        if not _BREVITAS_OK:
            raise ImportError("Brevitas is not installed. Install brevitas to use mode='qat_int8'.")

        # Input quantization - properly configured
        self.inp_quant = QuantIdentity(
            bit_width=8, 
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )

        # Regular LSTM (FP32)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_units,
            batch_first=True
        )
        
        # LSTM output quantization
        self.lstm_out_quant = QuantIdentity(
            bit_width=8,
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )

        # First linear layer with proper quantization
        self.fc1 = QuantLinear(
            in_features=lstm_units, 
            out_features=output_size,
            bias=True,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )
        
        # ReLU activation
        self.act1 = QuantReLU(
            bit_width=8,
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )

        # Second linear layer
        self.fc2 = QuantLinear(
            in_features=output_size, 
            out_features=1,
            bias=True,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True
        )
        
        # Output activation - don't return quant tensor for final output
        self.out_act = QuantSigmoid(
            bit_width=8,
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=False
        )

    def forward(self, x):
        # Input quantization
        x = self.inp_quant(x)
        
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        
        # Quantize LSTM output
        x = self.lstm_out_quant(x)
        
        # Quantized layers
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.out_act(x)
        return x

    def predict(self, inputs):
        self.eval()
        with torch.no_grad():
            if isinstance(inputs, np.ndarray):
                inputs = torch.FloatTensor(inputs)
            outputs = self.forward(inputs)
            return outputs.numpy()


# Even simpler approach - minimal quantization
class MinimalQuantModel(nn.Module):
    """Minimal INT8 QAT - only quantize what's absolutely necessary."""
    def __init__(self, lstm_units, output_size, input_size):
        super().__init__()
        if not _BREVITAS_OK:
            raise ImportError("Brevitas is not installed. Install brevitas to use mode='qat_int8'.")

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_units,
            batch_first=True
        )

        # Only quantize the linear layers with minimal configuration
        self.fc1 = QuantLinear(
            in_features=lstm_units, 
            out_features=output_size,
            weight_bit_width=8,
            bias_bit_width=8
        )
        
        self.act1 = nn.ReLU()
        #self.act1 = nn.PReLU()

        self.fc2 = QuantLinear(
            in_features=output_size, 
            out_features=1,
            weight_bit_width=8,
            bias_bit_width=8
        )
        
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        # LSTM in FP32
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        
        # Quantized linear layers with regular activations
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.out_act(x)
        return x

    def predict(self, inputs):
        self.eval()
        with torch.no_grad():
            if isinstance(inputs, np.ndarray):
                inputs = torch.FloatTensor(inputs)
            outputs = self.forward(inputs)
            return outputs.numpy()


def compute_loss():
    pass


def get_model(data_input, model_name, qth: float, lstm_units: int = 32, mode: str = "fp32"):
    if "dummy" in model_name:
        return DummyModel()

    training_generator = OutageData(**data_input)
    sample_X, _ = training_generator.__getitem__(0)
    input_size = sample_X.shape[2]

    if mode == "qat_int8":
        # Try the minimal model first for stability
        model = MinimalQuantModel(
            lstm_units=lstm_units,
            output_size=data_input["output_size"],
            input_size=input_size,
        )
        # Initialize quantization parameters
        _warmup_quant(model, torch.FloatTensor(sample_X), mode)
        print(f"Training model (INT8 QAT - Minimal): {model_name}")
        
        # Verify QAT implementation after creation
        _verify_qat_implementation(model, "AFTER CREATION")

    elif mode == "fixed16":
        model = TemperatureModel(
            output_size=data_input["output_size"],
            lstm_units=lstm_units,
            input_size=input_size,
        )
        model = model.half()
        print(f"Training model (16-bit fixed point): {model_name}")

    elif mode == "fp32":
        model = TemperatureModel(
            output_size=data_input["output_size"],
            lstm_units=lstm_units,
            input_size=input_size,
        )
        print(f"Training model (FP32): {model_name}")

    else:
        raise ValueError("mode must be one of {'fp32','fixed16','qat_int8'}")

    return model


def get_fitted_model(
    data_input,
    model_name,
    qth: float,
    epochs: int = 2,
    force_retrain: bool = False,
    lstm_units: int = 32,
    mode: str = "fp32",
):
    training_generator = OutageData(**data_input)
    os.makedirs("models", exist_ok=True)
    path = f"models/{model_name}_{mode}.pth"   

    if force_retrain or not os.path.exists(path):
        model = get_model(data_input, model_name, qth, lstm_units, mode=mode)

        if "mse" in model_name:
            criterion = nn.MSELoss()
            outage_like = False
        elif "bce" in model_name or "binary_cross_entropy" in model_name:
            criterion = nn.BCELoss()
            outage_like = False
        elif "mae" in model_name:
            criterion = nn.L1Loss()
            outage_like = False
        elif "inf_coef_loss" in model_name:
            criterion = InfiniteOutageCoefficientLoss(qth=qth)
            outage_like = True
        elif ("fin_coef_loss" in model_name) or ("olf" in model_name):
            criterion = FiniteOutageCoefficientLoss(S=data_input["batch_size"], qth=qth)
            outage_like = True
        elif "dummy" in model_name:
            return DummyModel()
        else:
            raise ValueError("Incorrect Loss Function in model_name.")

        # Use lower learning rate for QAT for stability
        lr = 0.001 if mode == "qat_int8" else 0.01
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        best_loss = float("inf")
        patience = max(30, epochs // 4)
        patience_counter = 0
        best_model_state = None

        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = len(training_generator)

            for batch_idx in range(num_batches):
                X, y = training_generator.__getitem__(batch_idx)
                X = torch.FloatTensor(X)
                y = torch.FloatTensor(y).view(-1, 1)

                if mode == "fixed16":
                    X = X.half()
                    y = y.half()

                optimizer.zero_grad()
                preds = model(X)

                if outage_like:
                    loss = criterion(y, preds)
                else:
                    loss = criterion(preds, y)

                loss.backward()
                
                # Gradient clipping for QAT stability
                if mode == "qat_int8":
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        if best_model_state:
            # Load best model state and do final quantization warmup
            model.load_state_dict(best_model_state)
            
            # Final warmup for quantization parameters
            sample_X, _ = training_generator.__getitem__(0)
            _warmup_quant(model, torch.FloatTensor(sample_X), mode)
            
            # Verify QAT implementation after training
            if mode == "qat_int8":
                _verify_qat_implementation(model, "AFTER TRAINING")
            
            # Save the trained model
            torch.save(model.state_dict(), path)
        return model

    else:
        print(f"Loading model: {model_name}")
        model = get_model(data_input, model_name, qth, lstm_units, mode=mode)
        
        # Load state dict
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        
        # Warmup quantization parameters
        sample_X, _ = training_generator.__getitem__(0)
        _warmup_quant(model, torch.FloatTensor(sample_X), mode)
        
        # Verify QAT implementation after loading
        if mode == "qat_int8":
            _verify_qat_implementation(model, "AFTER LOADING")
        
        model.eval()
        return model