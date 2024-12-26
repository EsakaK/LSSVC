import math

import torch
import torch.nn as nn
from src.models.priors import IntraNoAR
from src.IntraModules.utils import update_registered_buffers
from src.entropy_models.img_entropy_models import EntropyBottleneck, GaussianConditional
from src.utils.stream_helper import encode_i, decode_i, get_downsampled_shape, filesize
from src.IntraModules.layers import (
    conv3x3,
    subpel_conv3x3,
    MultiScaleTextureExtractor,
    MultiScaleTextureFusion,
    ReconGeneration,
    PriorFusion,
    TextureResampler,
    LayerPriorResampler,
    ResEncoder,
    ResDecoder
)


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


class IntraSS(CompressionModel):
    def __init__(self, channel_BL=192, channel_N=64, channel_M=96, base_layer_model_path=None, **kwargs):
        super().__init__(entropy_bottleneck_channels=channel_N, **kwargs)

        self.base_layer_model = IntraNoAR(channel_BL)

        self.texture_resampler = TextureResampler()
        self.layer_prior_resampler = LayerPriorResampler(channel_M, channel_BL)
        self.texture_extractor = MultiScaleTextureExtractor()
        self.context_fusion_net = MultiScaleTextureFusion()

        self.g_a = ResEncoder()

        self.h_a = nn.Sequential(
            nn.Conv2d(channel_M, channel_N, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_N, channel_N, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_N, channel_N, 3, stride=2, padding=1),
        )

        self.h_s = nn.Sequential(
            subpel_conv3x3(channel_N, channel_M, r=2),
            nn.LeakyReLU(),
            subpel_conv3x3(channel_M, channel_M * 3 // 2, r=2),
            nn.LeakyReLU(),
            nn.Conv2d(channel_M * 3 // 2, channel_M * 2, 3, stride=1, padding=1)
        )

        self.g_s = ResDecoder()

        self.recon_net = ReconGeneration()
        self.prior_fusion_net = PriorFusion()

        self.gaussian_conditional = GaussianConditional()
        self.N = int(channel_N)
        self.M = int(channel_M)
        self.shape_hr = (256, 256)
        self.scale_factor = 2.0
        self.pad_size = (0, 0, 0, 0)

        if base_layer_model_path is not None:
            self.load_bl_pretrain(base_layer_model_path)
            print(f"Loaded base_layer weights from: {base_layer_model_path}")

    def multi_scale_context_mining(self, x_bl):
        texture = self.texture_resampler(x_bl, self.shape_hr)
        texture1, texture2, texture3 = self.texture_extractor(texture)
        return self.context_fusion_net(texture1, texture2, texture3)

    def get_depadded_feature(self, feature, p=1):
        if feature is None:
            return None
        pad_a, pad_b, pad_c, pad_d = self.pad_size
        pad_size = (int(pad_a / p), int(pad_b / p), int(pad_c / p), int(pad_d / p),)
        feature = torch.nn.functional.pad(
            feature,
            pad_size,
            mode="constant",
            value=0,
        )
        return feature

    def forward(self, x_bl, x_el, train_with_recon=False):
        # BL forward
        # x_bl should not be compressed
        result = self.base_layer_model.get_layer_information(x_bl)
        bit_bl = result['bits']
        y_hat_bl = result['y_hat']
        x_hat_bl = result['x_hat']
        x_bl_for_ctx_mining = x_hat_bl
        # depadded process
        x_bl_for_ctx_mining = self.get_depadded_feature(x_bl_for_ctx_mining)  # texture resample
        y_hat_bl = self.get_depadded_feature(y_hat_bl, p=16)
        # EL forward
        context1, context2, context3 = self.multi_scale_context_mining(x_bl_for_ctx_mining)

        y = self.g_a(x_el, context1, context2, context3)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        hyper_prior = self.h_s(z_hat)
        layer_prior = self.layer_prior_resampler(y_hat_bl, self.shape_hr)
        params = self.prior_fusion_net(hyper_prior, layer_prior, context3)

        scales_hat, means_hat = params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        res_hat = self.g_s(y_hat, context2, context3)
        feature, x_hat = self.recon_net(res_hat, context1)

        bit_el = (torch.log(y_likelihoods).sum() + torch.log(z_likelihoods).sum()) / (-math.log(2))

        result = {
            'bit_bl': bit_bl.item(),
            'bit_el': bit_el.item(),
            'x_hat_bl': x_hat_bl,
            'x_hat_el': x_hat,
            'feature_el': feature
        }
        return result

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
    def from_state_dict(cls, state_dict, base_layer_model_path=None):
        """Return a new model instance from `state_dict`."""
        result_dict = {}
        for key, weight in state_dict.items():
            result_key = key
            if key[:7] == "module.":
                result_key = key[7:]
            result_dict[result_key] = weight
        if base_layer_model_path is not None:
            print(f'reload baselayer from {base_layer_model_path}')
            base_layer_load_checkpoint = torch.load(base_layer_model_path, map_location=torch.device('cpu'))
            if "state_dict" in base_layer_load_checkpoint:
                base_layer_load_checkpoint = base_layer_load_checkpoint['state_dict']
            for key in base_layer_load_checkpoint:
                result_dict['base_layer_model.' + key] = base_layer_load_checkpoint[key]
        # channel_N=192, channel_C=128, channel_BL=192, channel_res=64
        N_bl = result_dict["base_layer_model.g_s.0.conv1.weight"].size(0)
        net = cls(N_bl)
        # remove gaussian_conditional.scale_table from the state_dict
        if 'gaussian_conditional.scale_table' in result_dict:
            result_dict.pop('gaussian_conditional.scale_table', None)
        net.load_state_dict(result_dict)

        return net

    def load_bl_pretrain(self, bl_model_path):
        load_checkpoint = torch.load(bl_model_path, map_location=torch.device('cpu'))
        if "state_dict" in load_checkpoint:
            load_checkpoint = load_checkpoint['state_dict']
        self.base_layer_model.load_state_dict(load_checkpoint)

    def control_bl_is_train(self, is_train=False):
        self.base_layer_model.requires_grad_(is_train)
        if is_train:
            self.base_layer_model.train()
        else:
            self.base_layer_model.eval()

    def set_scale_information(self, scale, shape_hr, pad_size):
        self.scale_factor = scale
        self.shape_hr = shape_hr
        self.pad_size = pad_size

    def update(self, force=False):
        self.gaussian_conditional.update()
        self.base_layer_model.update()
        super().update(force=force)

    def get_y_z_ctx(self, x_bl, x_el):
        context1, context2, context3 = self.multi_scale_context_mining(x_bl)
        y = self.g_a(x_el, context1, context2, context3)
        z = self.h_a(y)
        return y, z, (context1, context2, context3)

    def encode_decode(self, x_bl, x_el, bin_path_bl, bin_path_el,
                      pic_height_bl, pic_width_bl,
                      pic_height_el, pic_width_el):
        if bin_path_bl is None:
            return self.forward(x_bl, x_el)
        # -------------------------------Encode----------------------------
        # BL encode
        y_bl, z_bl = self.base_layer_model.get_y_z(x_bl)
        compressed = self.base_layer_model.compress(None, y_bl, z_bl)

        y_string = compressed['strings'][0][0]
        z_string = compressed['strings'][1][0]
        encode_i(pic_height_bl, pic_width_bl, y_string, z_string, bin_path_bl)
        bit_bl = filesize(bin_path_bl) * 8
        x_hat_bl, y_hat_bl, _ = self.base_layer_model.get_y_hat_recon(y_bl, z_bl).values()

        # Depadded
        x_hat_bl_depadded = self.get_depadded_feature(x_hat_bl)
        y_hat_bl_depadded = self.get_depadded_feature(y_hat_bl, p=16)

        # EL encode
        y_el, z_el, ctx = self.get_y_z_ctx(x_hat_bl_depadded, x_el)
        assert pic_height_el is not None
        assert pic_width_el is not None
        compressed = self.compress(y=y_el, z=z_el, ctx3=ctx[2], y_hat_bl=y_hat_bl_depadded)

        y_string = compressed['strings'][0][0]
        z_string = compressed['strings'][1][0]
        encode_i(pic_height_el, pic_width_el, y_string, z_string, bin_path_el)
        bit_el = filesize(bin_path_el) * 8

        # -------------------------------Decode----------------------------
        # BL decode
        pic_height_bl, pic_width_bl, y_string, z_string = decode_i(bin_path_bl)
        shape = get_downsampled_shape(pic_height_bl, pic_width_bl, 64)
        decompressed = self.base_layer_model.decompress([[y_string], [z_string]], shape)

        x_hat_bl = decompressed['x_hat']
        y_hat_bl = decompressed['y_hat']
        # EL decode
        x_hat_bl_depadded = self.get_depadded_feature(x_hat_bl)
        y_hat_bl_depadded = self.get_depadded_feature(y_hat_bl, p=16)

        pic_height_el, pic_width_el, y_string, z_string = decode_i(bin_path_el)
        shape = get_downsampled_shape(pic_height_el, pic_width_el, 64)
        DPB_layer = {'x_hat_bl': x_hat_bl_depadded, 'y_hat_bl': y_hat_bl_depadded}
        decompressed = self.decompress([[y_string], [z_string]], DPB_layer, shape)
        x_hat_el = decompressed['x_hat']
        feature = decompressed['feature']

        result = {
            'bit_bl': bit_bl,
            'bit_el': bit_el,
            'x_hat_bl': x_hat_bl,
            'x_hat_el': x_hat_el,
            'feature_el': feature
        }
        return result

    def compress(self, y=None, z=None, ctx3=None, y_hat_bl=None):
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        hyper_prior = self.h_s(z_hat)
        layer_prior = self.layer_prior_resampler(y_hat_bl, self.shape_hr)
        params = self.prior_fusion_net(hyper_prior, layer_prior, ctx3)
        scales_hat, means_hat = params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, DPB_layer, shape):
        x_hat_bl = DPB_layer['x_hat_bl']
        y_hat_bl = DPB_layer['y_hat_bl']
        ctx1, ctx2, ctx3 = self.multi_scale_context_mining(x_hat_bl)
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)

        hyper_prior = self.h_s(z_hat)
        layer_prior = self.layer_prior_resampler(y_hat_bl, self.shape_hr)

        # print(torch.sum(ctx3 - x))

        params = self.prior_fusion_net(hyper_prior, layer_prior, ctx3)
        scales_hat, means_hat = params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)

        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        res_hat = self.g_s(y_hat, ctx2, ctx3)
        feature, x_hat = self.recon_net(res_hat, ctx1)
        return {'x_hat': x_hat, 'feature': feature}
