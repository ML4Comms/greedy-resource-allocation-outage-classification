import torch
import torch.nn as nn

from data_generator import OutageData


def heaviside(x):
    return torch.heaviside(x, torch.tensor(0.0, device=x.device))


def ensure_tensor(x):
    """Convert NumPy arrays or lists to torch.Tensor (float32)."""
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.tensor(x, dtype=torch.float32)


def TPR(qth, y_true, y_pred, step_function=heaviside):
    y_true = ensure_tensor(y_true)
    y_pred = ensure_tensor(y_pred)
    qth = torch.tensor(qth, dtype=torch.float32)

    result = generalise_TP(qth, y_true, y_pred, step_function) / (
        generalise_TP(qth, y_true, y_pred, step_function)
        + generalise_FN(qth, y_true, y_pred, step_function)
    )

    if torch.isnan(result):
        return torch.tensor(0.0).numpy()
    return result.detach().cpu().numpy()


def FPR(qth, y_true, y_pred, step_function=heaviside):
    y_true = ensure_tensor(y_true)
    y_pred = ensure_tensor(y_pred)
    qth = torch.tensor(qth, dtype=torch.float32)

    result = generalise_FP(qth, y_true, y_pred, step_function) / (
        generalise_FP(qth, y_true, y_pred, step_function)
        + generalise_TN(qth, y_true, y_pred, step_function)
    )

    if torch.isnan(result):
        return torch.tensor(0.0).numpy()
    return result.detach().cpu().numpy()


def Precision(qth, y_true, y_pred, step_function=heaviside):
    y_true = ensure_tensor(y_true)
    y_pred = ensure_tensor(y_pred)
    qth = torch.tensor(qth, dtype=torch.float32)

    result = generalise_TP(qth, y_true, y_pred, step_function) / (
        generalise_TP(qth, y_true, y_pred, step_function)
        + generalise_FP(qth, y_true, y_pred, step_function)
    )

    if torch.isnan(result):
        return torch.tensor(0.0).numpy()
    return result.detach().cpu().numpy()


def generalise_TN(qth, y_true, y_pred, step_function=heaviside):
    qth = ensure_tensor(qth)
    y_true = ensure_tensor(y_true)
    y_pred = ensure_tensor(y_pred)
    return torch.sum(step_function(qth - y_pred) * (1.0 - y_true))


def generalise_FN(qth, y_true, y_pred, step_function=heaviside):
    qth = ensure_tensor(qth)
    y_true = ensure_tensor(y_true)
    y_pred = ensure_tensor(y_pred)
    return torch.sum(step_function(qth - y_pred) * y_true)


def generalise_TP(qth, y_true, y_pred, step_function=heaviside):
    qth = ensure_tensor(qth)
    y_true = ensure_tensor(y_true)
    y_pred = ensure_tensor(y_pred)
    return torch.sum(step_function(y_pred - qth) * y_true)


def generalise_FP(qth, y_true, y_pred, step_function=heaviside):
    qth = ensure_tensor(qth)
    y_true = ensure_tensor(y_true)
    y_pred = ensure_tensor(y_pred)
    return torch.sum(step_function(y_pred - qth) * (1.0 - y_true))


# Loss classes
class MeanSquaredErrorTemperature(nn.Module):
    def __init__(self, reduction="mean", name=None):
        super().__init__()
        self.reduction = reduction

    def M(self, y_true, y_pred):
        y_true = ensure_tensor(y_true)
        y_pred = ensure_tensor(y_pred)
        return torch.mean((y_true - y_pred) ** 2)

    def forward(self, y_true, y_pred):
        return self.M(y_true, y_pred)


class InfiniteOutageCoefficientLoss(nn.Module):
    def __init__(self, reduction="mean", name=None, qth=0.5):
        super().__init__()
        self.qth = qth
        self.reduction = reduction

    def squashed_sigmoid(self, input, factor: float = 5.0):
        return 1.0 / (1.0 + torch.exp(-factor * input))

    def TN(self, y_true, y_pred):
        return generalise_TN(
            qth=self.qth,
            y_true=ensure_tensor(y_true),
            y_pred=ensure_tensor(y_pred),
            step_function=self.squashed_sigmoid,
        )

    def FN(self, y_true, y_pred):
        return generalise_FN(
            qth=self.qth,
            y_true=ensure_tensor(y_true),
            y_pred=ensure_tensor(y_pred),
            step_function=self.squashed_sigmoid,
        )

    def TP(self, y_true, y_pred):
        return generalise_TP(
            qth=self.qth,
            y_true=ensure_tensor(y_true),
            y_pred=ensure_tensor(y_pred),
            step_function=self.squashed_sigmoid,
        )

    def FP(self, y_true, y_pred):
        return generalise_FP(
            qth=self.qth,
            y_true=ensure_tensor(y_true),
            y_pred=ensure_tensor(y_pred),
            step_function=self.squashed_sigmoid,
        )

    def M(self, y_true, y_pred):
        y_true = ensure_tensor(y_true)
        y_pred = ensure_tensor(y_pred)
        epsilon = 1e-4

        numerator = self.FN(y_true, y_pred) * (
            self.TN(y_true, y_pred)
            + self.FN(y_true, y_pred)
            + self.TP(y_true, y_pred)
            + self.FP(y_true, y_pred)
        )

        denominator = epsilon + (self.TN(y_true, y_pred) + self.FN(y_true, y_pred)) * (
            self.FN(y_true, y_pred) + self.TP(y_true, y_pred)
        )
        return numerator / denominator

    def forward(self, y_true, y_pred):
        return self.M(y_true, y_pred)


class FiniteOutageCoefficientLoss(InfiniteOutageCoefficientLoss):
    def __init__(self, S: int, reduction="mean", name=None, qth=0.5):
        super().__init__(reduction=reduction, name=name, qth=qth)
        self.S = S

    def q(self, y_true, y_pred):
        y_true = ensure_tensor(y_true)
        y_pred = ensure_tensor(y_pred)
        epsilon = 1e-4

        numerator = self.TP(y_true, y_pred) + self.FP(y_true, y_pred)
        denominator = (
            epsilon
            + self.TP(y_true, y_pred)
            + self.FP(y_true, y_pred)
            + self.TN(y_true, y_pred)
            + self.FN(y_true, y_pred)
        )
        return numerator / denominator

    def forward(self, y_true, y_pred):
        M_val = self.M(y_true, y_pred)
        return M_val - torch.pow(self.q(y_true, y_pred), self.S - 1) * (M_val - 1.0)
