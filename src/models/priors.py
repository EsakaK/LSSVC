# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# pylint: disable=E0611,E0401
from src.IntraModules.rd_loss_intra import RDLossIntra
from src.IntraModules.utils import update_registered_buffers
from src.entropy_models.img_entropy_models import EntropyBottleneck, GaussianConditional
from src.utils.stream_helper import encode_i, decode_i, get_downsampled_shape, filesize
from src.IntraModules.layers import (
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    MaskedConv2d,
    conv3x3,
    subpel_conv3x3,
)
from .dmc_net_extend import DMCExtend


# pylint: enable=E0611,E0401


class CompressionModel(nn.Module):
    """Base class for constructing an auto-encoder with at least one entropy
    bottleneck module.

    Args:
        entropy_bottleneck_channels (int): Number of channels of the entropy
            bottleneck
    """

    def __init__(self, entropy_bottleneck_channels, init_weights=True):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(entropy_bottleneck_channels)

        if init_weights:
            self._initialize_weights()

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, *args):
        raise NotImplementedError()

    def parameters(self, *args, **kwargs):
        """Returns an iterator over the model parameters."""
        for m in self.children():
            if isinstance(m, EntropyBottleneck):
                continue
            for p in m.parameters():
                yield p

    def aux_parameters(self):
        """
        Returns an iterator over the entropy bottleneck(s) parameters for
        the auxiliary loss.
        """
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            for p in m.parameters():
                yield p

    def update(self, force=False):
        """Updates the entropy bottleneck(s) CDF values.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            force (bool): overwrite previous values (default: False)

        """
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            m.update(force=force)


class IntraNoAR(CompressionModel):
    def __init__(self, N, **kwargs):
        super().__init__(entropy_bottleneck_channels=N, **kwargs)

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
        )

        self.h_a = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2),
        )

        self.g_s = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )

        self.gaussian_conditional = GaussianConditional()
        self.N = int(N)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def load_state_dict(self, state_dict):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_s.0.conv1.weight"].size(0)
        net = cls(N)
        # remove gaussian_conditional.scale_table from the state_dict
        if 'gaussian_conditional.scale_table' in state_dict:
            state_dict.pop('gaussian_conditional.scale_table', None)
        net.load_state_dict(state_dict)
        return net

    def update(self, force=False):
        self.gaussian_conditional.update()
        super().update(force=force)

    def get_rec_only(self, x):
        with torch.no_grad():
            y = self.g_a(x)
            z = self.h_a(y)
            z_hat, _ = self.entropy_bottleneck(z)
            gaussian_params = self.h_s(z_hat)
            _, means_hat = gaussian_params.chunk(2, 1)

            y_hat = torch.round(y - means_hat) + means_hat
            x_hat = self.g_s(y_hat)
            return {
                "x_hat": x_hat,
                "y_hat": y_hat
            }

    def bits_rdo(self, y, z, x_padded, lmbda,
                 max_iter=3000, iter_to_exit=50, iter_to_reduce=25):
        y = Variable(y.clone().detach(), requires_grad=True)
        z = Variable(z.clone().detach(), requires_grad=True)

        criterion = RDLossIntra(lmbda)
        best_loss = 1e10
        best_y = y.clone().detach()
        best_z = z.clone().detach()
        iter_without_better_loss = 0
        iter_to_reduce_counter = 0
        update_threshold_y = 0.25
        update_step_y = 0.8
        update_threshold_z = 0.25
        update_step_z = 0.1
        reduced = 0

        for index in range(max_iter):
            z_hat, z_likelihoods = self.entropy_bottleneck(z)
            gaussian_params = self.h_s(z_hat)
            scales_hat, means_hat = gaussian_params.chunk(2, 1)
            y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)

            x_hat = self.g_s(y_hat)
            out_criterion = criterion(z_likelihoods, y_likelihoods, x_hat, x_padded)

            if out_criterion["loss"] < best_loss:
                iter_without_better_loss = 0
                iter_to_reduce_counter = 0
                best_loss = out_criterion["loss"]
                best_y = y.clone().detach()
                best_z = z.clone().detach()
                # print(f"{index}: {best_loss} q_step: {q_step}")
            else:
                # print(f"{index}: not best loss: {out_criterion['loss']}")
                iter_without_better_loss += 1
                iter_to_reduce_counter += 1
            out_criterion["loss"].backward()

            if reduced == 0 and iter_to_reduce_counter > iter_to_reduce:
                update_threshold_y = 0.5
                update_step_y = 0.2
                update_threshold_z = 0.5
                update_step_z = 0.05
                reduced = 1
                y = Variable(best_y.clone().detach(), requires_grad=True)
                z = Variable(best_z.clone().detach(), requires_grad=True)
                iter_to_reduce_counter = 0
                continue
            elif reduced == 1 and iter_to_reduce_counter > iter_to_reduce:
                update_threshold_y = 0.75
                update_step_y = 0.1
                update_threshold_z = 0.75
                update_step_z = 0.05
                reduced = 2
                y = Variable(best_y.clone().detach(), requires_grad=True)
                z = Variable(best_z.clone().detach(), requires_grad=True)
                iter_to_reduce_counter = 0
                continue

            y_grad = y.grad
            y_grad_abs = torch.abs(y_grad)
            y_grad_abs_max = torch.max(y_grad_abs)
            if y_grad_abs_max > 0:
                y_updates = torch.where(
                    y_grad_abs > y_grad_abs_max * update_threshold_y,
                    y_grad / y_grad_abs_max * update_step_y,
                    torch.zeros_like(y_grad))
                y = y - y_updates
            # else:
            #     print(f"too small y grad of {y_grad_abs_max}")
            y = Variable(y.clone().detach(), requires_grad=True)

            z_grad = z.grad
            z_grad_abs = torch.abs(z_grad)
            z_grad_abs_max = torch.max(z_grad_abs)
            if z_grad_abs_max > 0:
                z_updates = torch.where(
                    z_grad_abs > z_grad_abs_max * update_threshold_z,
                    z_grad / z_grad_abs_max * update_step_z,
                    torch.zeros_like(z_grad))
                z = z - z_updates
            # else:
            #     print(f"too small z grad of {z_grad_abs_max}")
            z = Variable(z.clone().detach(), requires_grad=True)

            if iter_without_better_loss >= iter_to_exit:
                break

        return best_y, best_z, best_loss

    def global_rdo(self, y, z, x_padded, rdo_opt):
        best_loss = 1e10
        best_y = y.clone().detach()
        best_z = z.clone().detach()
        self.entropy_bottleneck.set_RDO(True)
        self.gaussian_conditional.set_RDO(True)
        curr_y, curr_z, curr_loss = self.bits_rdo(
            best_y, best_z, x_padded, rdo_opt['lmbda'],
            iter_to_exit=rdo_opt['iter_to_exit'], iter_to_reduce=rdo_opt['iter_to_reduce'])
        if curr_loss < best_loss:
            best_loss = curr_loss
            best_y = curr_y.clone().detach()
            best_z = curr_z.clone().detach()

        self.entropy_bottleneck.set_RDO(False)
        self.gaussian_conditional.set_RDO(False)
        return best_y, best_z

    ##########################################################################
    def get_y_z(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        return y, z

    def get_recon_only(self, x):
        y, z = self.get_y_z(x)
        result = self.get_recon_wo_stream(y, z)
        return result

    def get_recon_wo_stream(self, y, z):
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)
        return {'x_hat': x_hat, 'y_hat': y_hat}

    ###########################################################################

    def get_y_hat_recon(self, y, z):
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)
        bit = (torch.log(y_likelihoods).sum() + torch.log(z_likelihoods).sum()) / (-math.log(2))

        return {
            'x_hat': x_hat,
            'y_hat': y_hat,
            'bit': bit
        }

    def get_layer_information(self, x):
        # inter-layer training
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)
        bits = (torch.log(y_likelihoods).sum() + torch.log(z_likelihoods).sum()) / (-math.log(2))

        pixel_num = x.shape[0] * x.shape[2] * x.shape[3]
        bpp = bits / pixel_num
        mse = torch.mean((x - x_hat).pow(2))
        return {
            'bits': bits,
            'mse': mse,
            'bpp': bpp,
            'x_hat': x_hat,
            'y_hat': y_hat
        }

    def encode_decode(self, x, output_path=None, pic_width=None, pic_height=None,
                      rdo=False, rdo_opt=None):
        # pic_width and pic_height may be different from x's size. X here is after padding
        # x_hat has the same size with x
        y, z = self.get_y_z(x)
        if output_path is None:
            x_hat, y_hat, bit = self.get_y_hat_recon(y, z).values()
            bit = bit.item()
        else:
            assert pic_height is not None
            assert pic_width is not None
            compressed = self.compress(y=y, z=z)
            y_string = compressed['strings'][0][0]
            z_string = compressed['strings'][1][0]
            encode_i(pic_height, pic_width, y_string, z_string, output_path)
            bit = filesize(output_path) * 8

            height, width, y_string, z_string = decode_i(output_path)
            shape = get_downsampled_shape(height, width, 64)
            decompressed = self.decompress([[y_string], [z_string]], shape)
            x_hat = decompressed['x_hat']
            y_hat = decompressed['y_hat']

        result = {
            'bit': bit,
            'x_hat': x_hat,
            'y_hat': y_hat
        }
        return result

    def compress(self, x=None, y=None, z=None):
        if x is None:
            assert y is not None and z is not None
        else:
            assert y is None and z is None

        if x is not None:
            y = self.g_a(x)
            z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )

        x_hat = self.g_s(y_hat)
        return {"x_hat": x_hat, "y_hat": y_hat}


class Cheng2020Anchor(CompressionModel):
    """Anchor model variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Uses residual blocks with small convolutions (3x3 and 1x1), and sub-pixel
    convolutions for up-sampling.

    Args:
        N (int): Number of channels
    """

    def __init__(self, N=192, leaky_relu_slope=0.01, **kwargs):
        super().__init__(entropy_bottleneck_channels=N, **kwargs)

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
        )

        self.h_a = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True, negative_slope=leaky_relu_slope),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True, negative_slope=leaky_relu_slope),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True, negative_slope=leaky_relu_slope),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True, negative_slope=leaky_relu_slope),
            conv3x3(N, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True, negative_slope=leaky_relu_slope),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True, negative_slope=leaky_relu_slope),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True, negative_slope=leaky_relu_slope),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True, negative_slope=leaky_relu_slope),
            conv3x3(N * 3 // 2, N * 2),
        )

        self.g_s = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, 3, 2),
        )

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(N * 12 // 3, N * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(N * 10 // 3, N * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(N * 8 // 3, N * 6 // 3, 1),
        )

        self.context_prediction = MaskedConv2d(
            N, 2 * N, kernel_size=5, padding=2, stride=1
        )

        self.gaussian_conditional = GaussianConditional()
        self.N = int(N)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional._quantize(  # pylint: disable=protected-access
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def get_rec_only(self, x):
        with torch.no_grad():
            y = self.g_a(x)
            y_hat = torch.round(y)
            x_hat = self.g_s(y_hat)

            return {
                "x_hat": x_hat,
                "y_hat": y_hat
            }

    @classmethod
    def from_state_dict(cls, state_dict, leaky_relu_slope=0.01):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_s.0.conv1.weight"].size(0)
        # print(f"slope: {leaky_relu_slope}")
        net = cls(N, leaky_relu_slope)
        net.load_state_dict(state_dict)
        return net

    def get_y_z(self, x, rdo=False, rdo_opt=None):
        if rdo:
            warnings.warn("RDO is not supported.")
        y = self.g_a(x)
        z = self.h_a(y)
        return y, z

    def get_recon_wo_stream(self, y, z):
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional._quantize(  # pylint: disable=protected-access
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)

        bit = (torch.log(y_likelihoods).sum() + torch.log(z_likelihoods).sum()) / (-math.log(2))
        return x_hat, bit

    def encode_decode(self, x, output_path=None, pic_width=None, pic_height=None,
                      rdo=False, rdo_opt=None):
        # pic_width and pic_height may be different from x's size. X here is after padding
        # x_hat has the same size with x
        y, z = self.get_y_z(x, rdo, rdo_opt)
        if output_path is None:
            x_hat, bit = self.get_recon_wo_stream(y, z)
            bit = bit.item()
        else:
            assert pic_height is not None
            assert pic_width is not None
            compressed = self.compress(y=y, z=z)
            y_string = compressed['strings'][0][0]
            z_string = compressed['strings'][1][0]
            encode_i(pic_height, pic_width, y_string, z_string, output_path)
            bit = filesize(output_path) * 8

            height, width, y_string, z_string = decode_i(output_path)
            shape = get_downsampled_shape(height, width, 64)
            decompressed = self.decompress([[y_string], [z_string]], shape)
            x_hat = decompressed['x_hat']

        result = {
            'bit': bit,
            'x_hat': x_hat,
        }
        return result

    def compress(self, x=None, y=None, z=None):
        from ..entropy_models.MLCodec_rans import BufferedRansEncoder
        if x is None:
            assert y is not None and z is not None
        else:
            assert y is None and z is None

        if x is not None:
            y = self.g_a(x)
            z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        # pylint: disable=protected-access
        cdf = self.gaussian_conditional._quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional._cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional._offset.reshape(-1).int().tolist()
        # pylint: enable=protected-access

        y_strings = []
        for i in range(y.size(0)):
            encoder = BufferedRansEncoder()
            # Warning, this is slow...
            # TODO: profile the calls to the bindings...
            symbols_list = []
            indexes_list = []
            for h in range(y_height):
                for w in range(y_width):
                    y_crop = y_hat[i:i + 1, :, h:h + kernel_size, w:w + kernel_size]
                    ctx_p = F.conv2d(
                        y_crop,
                        self.context_prediction.weight,
                        bias=self.context_prediction.bias,
                    )

                    # 1x1 conv for the entropy parameters prediction network, so
                    # we only keep the elements in the "center"
                    p = params[i:i + 1, :, h:h + 1, w:w + 1]
                    gaussian_params = self.entropy_parameters(
                        torch.cat((p, ctx_p), dim=1)
                    )
                    scales_hat, means_hat = gaussian_params.chunk(2, 1)

                    indexes = self.gaussian_conditional.build_indexes(scales_hat)
                    y_q = torch.round(y_crop - means_hat)
                    y_hat[i, :, h + padding, w + padding] = (y_q + means_hat)[
                                                            i, :, padding, padding
                                                            ]

                    symbols_list.extend(y_q[i, :, padding, padding].int().tolist())
                    indexes_list.extend(indexes[i, :].squeeze().int().tolist())

            encoder.encode_with_indexes(
                symbols_list, indexes_list, cdf, cdf_lengths, offsets
            )

            string = encoder.flush()
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        from ..entropy_models.MLCodec_rans import RansDecoder
        assert isinstance(strings, list) and len(strings) == 2

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "InterModules (the entropy coder is run sequentially on CPU)."
            )

        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (z_hat.size(0), self.N, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )
        decoder = RansDecoder()

        # pylint: disable=protected-access
        cdf = self.gaussian_conditional._quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional._cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional._offset.reshape(-1).int().tolist()

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        for i, y_string in enumerate(strings[0]):
            decoder.set_stream(y_string)

            for h in range(y_height):
                for w in range(y_width):
                    # only perform the 5x5 convolution on a cropped tensor
                    # centered in (h, w)
                    y_crop = y_hat[i:i + 1, :, h:h + kernel_size, w:w + kernel_size]
                    ctx_p = F.conv2d(
                        y_crop,
                        self.context_prediction.weight,
                        bias=self.context_prediction.bias,
                    )
                    # 1x1 conv for the entropy parameters prediction network, so
                    # we only keep the elements in the "center"
                    p = params[i:i + 1, :, h:h + 1, w:w + 1]
                    gaussian_params = self.entropy_parameters(
                        torch.cat((p, ctx_p), dim=1)
                    )
                    scales_hat, means_hat = gaussian_params.chunk(2, 1)

                    indexes = self.gaussian_conditional.build_indexes(scales_hat)

                    rv = decoder.decode_stream(
                        indexes[i, :].squeeze().int().tolist(),
                        cdf,
                        cdf_lengths,
                        offsets,
                    )
                    rv = torch.Tensor(rv).reshape(1, -1, 1, 1)

                    rv = self.gaussian_conditional._dequantize(rv, means_hat)

                    y_hat[i, :, h + padding: h + padding + 1, w + padding: w + padding + 1] = rv
        y_hat = y_hat[:, :, padding:-padding, padding:-padding]
        # pylint: enable=protected-access

        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

    def update(self, force=False):
        self.gaussian_conditional.update()
        super().update(force=force)

    def load_state_dict(self, state_dict):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        # remove gaussian_conditional.scale_table from the state_dict
        if 'gaussian_conditional.scale_table' in state_dict:
            state_dict.pop('gaussian_conditional.scale_table', None)
        super().load_state_dict(state_dict)


model_architectures = {
    "cheng2020-anchor": Cheng2020Anchor,
    "IntraNoAR": IntraNoAR,
}
