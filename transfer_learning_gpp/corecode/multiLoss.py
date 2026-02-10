import numpy as np
import torch

# ---------------------------------------------------------------------
# Device configuration
# ---------------------------------------------------------------------
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------
# Normalization statistics (mean–std normalization)
# ---------------------------------------------------------------------
SCALER = {
    "daily_means": torch.from_numpy(np.load("NORM/dyn_mean.npy")).to(DEVICE),
    "daily_stds": torch.from_numpy(np.load("NORM/dyn_std.npy")).to(DEVICE),
    "gpp_mean": torch.from_numpy(np.load("NORM/gpp_mean.npy")).to(DEVICE),
    "gpp_std": torch.from_numpy(np.load("NORM/gpp_std.npy")).to(DEVICE),
    "reco_mean": torch.from_numpy(np.load("NORM/reco_mean.npy")).to(DEVICE),
    "reco_std": torch.from_numpy(np.load("NORM/reco_std.npy")).to(DEVICE),
    "nee_mean": torch.from_numpy(np.load("NORM/nee_mean.npy")).to(DEVICE),
    "nee_std": torch.from_numpy(np.load("NORM/nee_std.npy")).to(DEVICE),
}


def _nee_from_preds(gpp_pred: torch.Tensor, reco_pred: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct normalized NEE from predicted GPP and Reco.

    Predictions for GPP and Reco are in normalized space.
    NEE is reconstructed in physical units as Reco − GPP and then re-normalized.
    """
    reco_phys = reco_pred * SCALER["reco_std"] + SCALER["reco_mean"]
    gpp_phys = gpp_pred * SCALER["gpp_std"] + SCALER["gpp_mean"]
    nee_phys = reco_phys - gpp_phys
    nee_norm = (nee_phys - SCALER["nee_mean"]) / SCALER["nee_std"]
    return nee_norm


class MultiLoss(torch.nn.Module):
    """
    Equal-weight multi-task MSE for (GPP, Reco, NEE).

    GPP and Reco are predicted directly; NEE is reconstructed as Reco − GPP.
    """

    def __init__(self):
        super().__init__()

    def forward(self, y_gpp, y_reco, y_nee, gpp_pred, reco_pred):
        nee_pred = _nee_from_preds(gpp_pred=gpp_pred, reco_pred=reco_pred)

        gpp_loss = torch.mean((y_gpp - gpp_pred) ** 2)
        reco_loss = torch.mean((y_reco - reco_pred) ** 2)
        nee_loss = torch.mean((y_nee - nee_pred) ** 2)

        loss = gpp_loss + reco_loss + nee_loss
        return loss, gpp_loss.detach(), reco_loss.detach(), nee_loss.detach()


class MultiLossKendall(torch.nn.Module):
    """
    Uncertainty-weighted multi-task loss (Kendall et al. (2018)) for (GPP, Reco, NEE).

    Task weights are learned as log-variances; each task is weighted by exp(-log_var).
    """

    def __init__(self):
        super().__init__()
        self.log_vars = torch.nn.Parameter(torch.zeros(3))

    def forward(self, y_gpp, y_reco, y_nee, gpp_pred, reco_pred):
        nee_pred = _nee_from_preds(gpp_pred=gpp_pred, reco_pred=reco_pred)

        gpp_loss = torch.mean((y_gpp - gpp_pred) ** 2)
        reco_loss = torch.mean((y_reco - reco_pred) ** 2)
        nee_loss = torch.mean((y_nee - nee_pred) ** 2)

        precision = torch.exp(-self.log_vars)  # inverse variances

        loss = (
            precision[0] * gpp_loss + self.log_vars[0]
            + precision[1] * reco_loss + self.log_vars[1]
            + precision[2] * nee_loss + self.log_vars[2]
        )

        return (
            loss,
            gpp_loss.detach(),
            reco_loss.detach(),
            nee_loss.detach(),
            precision.detach().cpu().tolist(),
        )


class MultiLossSoftmax(torch.nn.Module):
    """
    Softmax-weighted multi-task MSE for (GPP, Reco, NEE).

    Weights are learned unconstrained parameters mapped to a simplex via softmax.
    """

    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.ones(3))

    def forward(self, y_gpp, y_reco, y_nee, gpp_pred, reco_pred):
        nee_pred = _nee_from_preds(gpp_pred=gpp_pred, reco_pred=reco_pred)

        gpp_loss = torch.mean((y_gpp - gpp_pred) ** 2)
        reco_loss = torch.mean((y_reco - reco_pred) ** 2)
        nee_loss = torch.mean((y_nee - nee_pred) ** 2)

        w = torch.nn.functional.softmax(self.weights, dim=0)
        loss = w[0] * gpp_loss + w[1] * reco_loss + w[2] * nee_loss

        return loss, gpp_loss.detach(), reco_loss.detach(), nee_loss.detach(), w.detach().cpu().tolist()
