import math
import torch
import torch.nn as nn


class RDLossIntra(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, z_likelihood, y_likelihood, x_hat, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["z_bpp_loss"] = torch.log(z_likelihood).sum() / (-math.log(2) * num_pixels)
        out["y_bpp_loss"] = torch.log(y_likelihood).sum() / (-math.log(2) * num_pixels)
        out["bpp_loss"] = out["z_bpp_loss"] + out["y_bpp_loss"]
        out["mse_loss"] = self.mse(x_hat, target)
        out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]

        return out

    def loss_without_z(self, y_likelihood, x_hat, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["y_bpp_loss"] = torch.log(y_likelihood).sum() / (-math.log(2) * num_pixels)
        out["bpp_loss"] = out["y_bpp_loss"]
        out["mse_loss"] = self.mse(x_hat, target)
        out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]

        return out
