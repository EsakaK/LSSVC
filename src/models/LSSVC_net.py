import math

import torch
from pytorch_msssim import MS_SSIM

from src.InterModules.video_net_component import flow_warp, bilineardownsacling, ME_Spynet_DCVC
from src.entropy_models.video_entropy_models import BitEstimator, GaussianEncoder
from src.models.dmc_net_extend import DMCExtend
from src.InterModules.lssvc_modules import *


class LSSVC(nn.Module):
    def __init__(self, bl_model_path=None, mv_pretrain_path=None, win_size=11):
        super().__init__()
        self.version = 'Final'

        channel_N = 64
        channel_mv = 64

        self.channel_N = channel_N
        self.channel_mv = channel_mv

        # load BL
        self.base_layer_model = DMCExtend()
        # adaptor
        self.feature_adaptor_EL_I = nn.Conv2d(3, g_ch_1x, 3, stride=1, padding=1)
        self.feature_adaptor_EL_first_P = nn.Conv2d(channel_N, g_ch_1x, 3, stride=1, padding=1)
        self.feature_adaptor_EL = nn.Conv2d(g_ch_1x, g_ch_1x, 3, stride=1, padding=1)
        # upsampler
        self.mv_resampler = MvResampler()
        self.texture_resampler = TextureResampler()
        self.layer_prior_resampler = LayerPriorResampler()
        # context extractor
        self.feature_extractor = FeatureExtractor()
        self.texture_extractor = TextureExtractor()
        self.context_fusion_net = MultiScaleContextFusion()
        self.weight_map_generator = HybridWeightGenerator()
        # prior modules
        self.prior_fusion_net = PriorFusion()
        self.y_spatial_prior_adaptor_1 = nn.Conv2d(g_ch_16x * 3, g_ch_16x * 3, 1)
        self.y_spatial_prior_adaptor_2 = nn.Conv2d(g_ch_16x * 3, g_ch_16x * 3, 1)
        self.y_spatial_prior_adaptor_3 = nn.Conv2d(g_ch_16x * 3, g_ch_16x * 3, 1)

        self.y_spatial_prior = nn.Sequential(
            DepthConvBlock(g_ch_16x * 3, g_ch_16x * 3, inplace=False),
            DepthConvBlock(g_ch_16x * 3, g_ch_16x * 3, inplace=False),
            DepthConvBlock(g_ch_16x * 3, g_ch_16x * 2, inplace=False),
        )

        self.masks = {}

        # autoencoder
        self.res_encoder = ResEncoder()

        self.res_prior_encoder = nn.Sequential(
            nn.Conv2d(g_ch_16x, g_ch_16x, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(g_ch_16x, g_ch_16x, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(g_ch_16x, g_ch_16x, 3, stride=2, padding=1),
        )

        self.res_prior_decoder = nn.Sequential(
            nn.Conv2d(g_ch_16x, g_ch_16x, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            subpel_conv1x1(g_ch_16x, g_ch_16x, 2),
            nn.LeakyReLU(),
            nn.Conv2d(g_ch_16x, g_ch_16x, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            subpel_conv1x1(g_ch_16x, g_ch_16x, 2),
            nn.LeakyReLU(),
            nn.Conv2d(g_ch_16x, g_ch_16x, 3, stride=1, padding=1),
        )

        self.temporal_prior_encoder = nn.Sequential(
            nn.Conv2d(g_ch_4x, g_ch_8x, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(g_ch_8x, g_ch_16x, 3, stride=2, padding=1),
        )
        self.res_decoder = ResDecoder()
        self.recon_generation_net = ReconGeneration()

        ###############################
        # flow part
        self.optic_flow = ME_Spynet_DCVC(me_model_dir=None)
        self.align = OffsetDiversity(inplace=False)
        self.mv_ctx_transform = MVContextTransformer()
        self.mv_encoder = MVResEncoder()

        self.mv_prior_encoder = nn.Sequential(
            nn.Conv2d(channel_mv, channel_mv, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_mv, channel_mv, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_mv, channel_mv, 3, stride=2, padding=1),
        )

        self.mv_prior_decoder = nn.Sequential(
            subpel_conv3x3(channel_mv, channel_mv, r=2),
            nn.LeakyReLU(),
            subpel_conv3x3(channel_mv, channel_mv * 3 // 2, r=2),
            nn.LeakyReLU(),
            nn.Conv2d(channel_mv * 3 // 2, channel_mv * 2, 3, stride=1, padding=1)
        )

        self.mv_decoder = MVResDecoder()

        self.mv_ctx_prior_encoder = nn.Sequential(
            nn.Conv2d(2, channel_mv, 3, stride=2, padding=1),
            GDN(channel_mv),
            nn.Conv2d(channel_mv, channel_mv, 3, stride=2, padding=1),
            GDN(channel_mv),
            nn.Conv2d(channel_mv, channel_mv, 3, stride=2, padding=1),
            GDN(channel_mv),
            nn.Conv2d(channel_mv, channel_mv, 3, stride=2, padding=1),
        )

        self.mv_prior_fusion = nn.Sequential(
            nn.Conv2d(channel_mv * 9 // 3, channel_mv * 8 // 3, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_mv * 8 // 3, channel_mv * 7 // 3, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_mv * 7 // 3, channel_mv * 6 // 3, 3, stride=1, padding=1),
        )
        ###############
        # entropy part
        self.bit_estimator_z = BitEstimator(g_ch_16x)
        self.bit_estimator_z_mv = BitEstimator(channel_mv)
        self.gaussian_encoder = GaussianEncoder()
        self.ms_ssim_loss = MS_SSIM(data_range=1.0, win_size=win_size)

        # decoded buffer
        self.previous_frame_recon_bl = None
        self.previous_frame_feature_bl = None
        self.previous_frame_recon_el = None
        self.previous_frame_feature_el = None
        self.shape_hr = (256, 256)
        self.scale_factor = 2.0
        self.pad_size = (0, 0, 0, 0)

    def load_dict(self, pretrained_dict, strict=True):
        result_dict = {}
        for key, weight in pretrained_dict.items():
            result_key = key
            if key[:7] == "module.":
                result_key = key[7:]
            result_dict[result_key] = weight

        self.load_state_dict(result_dict, strict=strict)

    def update_ms_ssim_win_size(self, win_size):
        self.ms_ssim_loss = MS_SSIM(data_range=1.0, win_size=win_size)

    @staticmethod
    def get_y_bits_probs(y, sigma):
        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.laplace.Laplace(mu, sigma)
        probs = gaussian.cdf(y + 0.5) - gaussian.cdf(y - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50))
        return total_bits, probs

    @staticmethod
    def get_z_bits_probs(z, bit_estimator):
        prob = bit_estimator(z + 0.5) - bit_estimator(z - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))
        return total_bits, prob

    def encode_decode_extend(self, *args, **kwargs):
        pass

    def encode_decode(self, x_bl, x_el, dpb, output_path_bl=None, output_path_el=None,
                      pic_width=None, pic_height=None, pic_width_bl=None, pic_height_bl=None):
        # pic_width and pic_height may be different from x's size. x here is after padding
        # x_hat has the same size with x
        if output_path_el is not None:
            return self.encode_decode_extend(x_bl, x_el, dpb,
                                             output_path_bl, output_path_el,
                                             pic_width, pic_height, pic_width_bl, pic_height_bl)
        return None

    def quant(self, x, force_detach=False):
        if self.training or force_detach:
            n = torch.round(x) - x
            n = n.clone().detach()
            return x + n
        else:
            return torch.round(x)

    def multi_scale_feature_extractor(self, ref_el, feature):
        if feature is None:
            f = self.feature_adaptor_EL_I(ref_el)
        elif feature.size()[1] == 64:
            f = self.feature_adaptor_EL_first_P(feature)
        else:
            f = self.feature_adaptor_EL(feature)
        return self.feature_extractor(f)

    def multi_scale_texture_extractor(self, texture):
        return self.texture_extractor(texture)

    def motion_resampling(self, mv_bl):
        """
        Motion Resampling
        version:
        1.5: bilinear resample
        """
        mv = self.mv_resampler(mv_bl, self.shape_hr, self.scale_factor)
        return mv

    def texture_resampling(self, texture_bl):
        """
        Texture Resampling part
        version:
        1.5: bilinear upsample
        """
        texture_el = self.texture_resampler(texture_bl, self.shape_hr)
        return texture_el

    def layer_prior_resampling(self, y_hat_bl):
        layer_prior = self.layer_prior_resampler(y_hat_bl, (self.shape_hr[0] // 16, (self.shape_hr[1] // 16)))
        return layer_prior

    def motion_compensation(self, ref, feature_el, mv):
        # all P frame
        mv1 = mv
        warpframe = flow_warp(ref, mv1)
        mv2 = bilineardownsacling(mv1) / 2
        mv3 = bilineardownsacling(mv2) / 2
        # EL feature extract
        ref_feature1, ref_feature2, ref_feature3 = self.multi_scale_feature_extractor(ref, feature_el)
        context1_init = flow_warp(ref_feature1, mv)
        context1 = self.align(ref_feature1, torch.cat(
            (context1_init, warpframe, mv), dim=1), mv)
        context2 = flow_warp(ref_feature2, mv2)
        context3 = flow_warp(ref_feature3, mv3)
        context1, context2, context3 = self.context_fusion_net(context1, context2, context3)

        return (context1, context2, context3), warpframe, mv1

    def hybrid_temporal_layer_context_fusion(self, texture_bl, mv, ref, feature_el):
        temporal_ctx, warp_frame, mv1 = self.motion_compensation(ref, feature_el, mv)
        if texture_bl is not None:
            texture = self.texture_resampling(texture_bl)
            spatial_ctx = self.texture_extractor(texture)
            map_temp, map_spat = self.weight_map_generator(temporal_ctx, spatial_ctx)

            context1 = temporal_ctx[0] * map_temp[0] + spatial_ctx[0] * map_spat[0]
            context2 = temporal_ctx[1] * map_temp[1] + spatial_ctx[1] * map_spat[1]
            context3 = temporal_ctx[2] * map_temp[2] + spatial_ctx[2] * map_spat[2]
        else:
            context1, context2, context3 = temporal_ctx
        context1, context2, context3 = self.context_fusion_net(context1, context2, context3)
        return context1, context2, context3, warp_frame, mv

    def hybrid_temporal_layer_prior_parameters(self, hyper_prior, temporal_prior, layer_prior_bl):
        # layer_prior resample
        layer_prior = self.layer_prior_resampling(layer_prior_bl)
        return self.prior_fusion_net(hyper_prior, temporal_prior, layer_prior)

    def set_scale_information(self, scale, shape_hr, pad_size):
        self.scale_factor = scale
        self.shape_hr = shape_hr
        self.pad_size = pad_size

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

    @staticmethod
    def separate_prior(params):
        return params.chunk(2, 1)

    def process_with_mask(self, y, scales, means, mask):
        scales_hat = scales * mask
        means_hat = means * mask

        y_res = (y - means_hat) * mask
        y_q = self.quant(y_res)
        y_hat = y_q + means_hat

        return y_res, y_q, y_hat, scales_hat

    def get_mask_four_parts(self, height, width, dtype, device):
        curr_mask_str = f"{width}x{height}"
        if curr_mask_str not in self.masks:
            micro_mask_0 = torch.tensor(((1, 0), (0, 0)), dtype=dtype, device=device)
            mask_0 = micro_mask_0.repeat((height + 1) // 2, (width + 1) // 2)
            mask_0 = mask_0[:height, :width]
            mask_0 = torch.unsqueeze(mask_0, 0)
            mask_0 = torch.unsqueeze(mask_0, 0)

            micro_mask_1 = torch.tensor(((0, 1), (0, 0)), dtype=dtype, device=device)
            mask_1 = micro_mask_1.repeat((height + 1) // 2, (width + 1) // 2)
            mask_1 = mask_1[:height, :width]
            mask_1 = torch.unsqueeze(mask_1, 0)
            mask_1 = torch.unsqueeze(mask_1, 0)

            micro_mask_2 = torch.tensor(((0, 0), (1, 0)), dtype=dtype, device=device)
            mask_2 = micro_mask_2.repeat((height + 1) // 2, (width + 1) // 2)
            mask_2 = mask_2[:height, :width]
            mask_2 = torch.unsqueeze(mask_2, 0)
            mask_2 = torch.unsqueeze(mask_2, 0)

            micro_mask_3 = torch.tensor(((0, 0), (0, 1)), dtype=dtype, device=device)
            mask_3 = micro_mask_3.repeat((height + 1) // 2, (width + 1) // 2)
            mask_3 = mask_3[:height, :width]
            mask_3 = torch.unsqueeze(mask_3, 0)
            mask_3 = torch.unsqueeze(mask_3, 0)
            self.masks[curr_mask_str] = [mask_0, mask_1, mask_2, mask_3]
        return self.masks[curr_mask_str]

    @staticmethod
    def combine_four_parts(x_0_0, x_0_1, x_0_2, x_0_3,
                           x_1_0, x_1_1, x_1_2, x_1_3,
                           x_2_0, x_2_1, x_2_2, x_2_3,
                           x_3_0, x_3_1, x_3_2, x_3_3):
        x_0 = x_0_0 + x_0_1 + x_0_2 + x_0_3
        x_1 = x_1_0 + x_1_1 + x_1_2 + x_1_3
        x_2 = x_2_0 + x_2_1 + x_2_2 + x_2_3
        x_3 = x_3_0 + x_3_1 + x_3_2 + x_3_3
        return torch.cat((x_0, x_1, x_2, x_3), dim=1)

    def forward_four_part_prior(self, y, common_params,
                                y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                y_spatial_prior_adaptor_3, y_spatial_prior, write=False):
        '''
        y_0 means split in channel, the 0/4 quater
        y_1 means split in channel, the 1/4 quater
        y_2 means split in channel, the 2/4 quater
        y_3 means split in channel, the 3/4 quater
        y_?_0, means multiply with mask_0
        y_?_1, means multiply with mask_1
        y_?_2, means multiply with mask_2
        y_?_3, means multiply with mask_3
        '''
        scales, means = self.separate_prior(common_params)
        dtype = y.dtype
        device = y.device
        _, _, H, W = y.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(H, W, dtype, device)
        y_0, y_1, y_2, y_3 = y.chunk(4, 1)

        scales_0, scales_1, scales_2, scales_3 = scales.chunk(4, 1)
        means_0, means_1, means_2, means_3 = means.chunk(4, 1)

        y_res_0_0, y_q_0_0, y_hat_0_0, s_hat_0_0 = \
            self.process_with_mask(y_0, scales_0, means_0, mask_0)
        y_res_1_1, y_q_1_1, y_hat_1_1, s_hat_1_1 = \
            self.process_with_mask(y_1, scales_1, means_1, mask_1)
        y_res_2_2, y_q_2_2, y_hat_2_2, s_hat_2_2 = \
            self.process_with_mask(y_2, scales_2, means_2, mask_2)
        y_res_3_3, y_q_3_3, y_hat_3_3, s_hat_3_3 = \
            self.process_with_mask(y_3, scales_3, means_3, mask_3)
        y_hat_curr_step = torch.cat((y_hat_0_0, y_hat_1_1, y_hat_2_2, y_hat_3_3), dim=1)

        y_hat_so_far = y_hat_curr_step
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales_0, scales_1, scales_2, scales_3, means_0, means_1, means_2, means_3 = \
            y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(8, 1)

        y_res_0_3, y_q_0_3, y_hat_0_3, s_hat_0_3 = \
            self.process_with_mask(y_0, scales_0, means_0, mask_3)
        y_res_1_2, y_q_1_2, y_hat_1_2, s_hat_1_2 = \
            self.process_with_mask(y_1, scales_1, means_1, mask_2)
        y_res_2_1, y_q_2_1, y_hat_2_1, s_hat_2_1 = \
            self.process_with_mask(y_2, scales_2, means_2, mask_1)
        y_res_3_0, y_q_3_0, y_hat_3_0, s_hat_3_0 = \
            self.process_with_mask(y_3, scales_3, means_3, mask_0)
        y_hat_curr_step = torch.cat((y_hat_0_3, y_hat_1_2, y_hat_2_1, y_hat_3_0), dim=1)

        y_hat_so_far = y_hat_so_far + y_hat_curr_step
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales_0, scales_1, scales_2, scales_3, means_0, means_1, means_2, means_3 = \
            y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(8, 1)

        y_res_0_2, y_q_0_2, y_hat_0_2, s_hat_0_2 = \
            self.process_with_mask(y_0, scales_0, means_0, mask_2)
        y_res_1_3, y_q_1_3, y_hat_1_3, s_hat_1_3 = \
            self.process_with_mask(y_1, scales_1, means_1, mask_3)
        y_res_2_0, y_q_2_0, y_hat_2_0, s_hat_2_0 = \
            self.process_with_mask(y_2, scales_2, means_2, mask_0)
        y_res_3_1, y_q_3_1, y_hat_3_1, s_hat_3_1 = \
            self.process_with_mask(y_3, scales_3, means_3, mask_1)
        y_hat_curr_step = torch.cat((y_hat_0_2, y_hat_1_3, y_hat_2_0, y_hat_3_1), dim=1)

        y_hat_so_far = y_hat_so_far + y_hat_curr_step
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales_0, scales_1, scales_2, scales_3, means_0, means_1, means_2, means_3 = \
            y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(8, 1)

        y_res_0_1, y_q_0_1, y_hat_0_1, s_hat_0_1 = \
            self.process_with_mask(y_0, scales_0, means_0, mask_1)
        y_res_1_0, y_q_1_0, y_hat_1_0, s_hat_1_0 = \
            self.process_with_mask(y_1, scales_1, means_1, mask_0)
        y_res_2_3, y_q_2_3, y_hat_2_3, s_hat_2_3 = \
            self.process_with_mask(y_2, scales_2, means_2, mask_3)
        y_res_3_2, y_q_3_2, y_hat_3_2, s_hat_3_2 = \
            self.process_with_mask(y_3, scales_3, means_3, mask_2)

        y_res = self.combine_four_parts(y_res_0_0, y_res_0_1, y_res_0_2, y_res_0_3,
                                        y_res_1_0, y_res_1_1, y_res_1_2, y_res_1_3,
                                        y_res_2_0, y_res_2_1, y_res_2_2, y_res_2_3,
                                        y_res_3_0, y_res_3_1, y_res_3_2, y_res_3_3)
        y_q = self.combine_four_parts(y_q_0_0, y_q_0_1, y_q_0_2, y_q_0_3,
                                      y_q_1_0, y_q_1_1, y_q_1_2, y_q_1_3,
                                      y_q_2_0, y_q_2_1, y_q_2_2, y_q_2_3,
                                      y_q_3_0, y_q_3_1, y_q_3_2, y_q_3_3)
        y_hat = self.combine_four_parts(y_hat_0_0, y_hat_0_1, y_hat_0_2, y_hat_0_3,
                                        y_hat_1_0, y_hat_1_1, y_hat_1_2, y_hat_1_3,
                                        y_hat_2_0, y_hat_2_1, y_hat_2_2, y_hat_2_3,
                                        y_hat_3_0, y_hat_3_1, y_hat_3_2, y_hat_3_3)
        scales_hat = self.combine_four_parts(s_hat_0_0, s_hat_0_1, s_hat_0_2, s_hat_0_3,
                                             s_hat_1_0, s_hat_1_1, s_hat_1_2, s_hat_1_3,
                                             s_hat_2_0, s_hat_2_1, s_hat_2_2, s_hat_2_3,
                                             s_hat_3_0, s_hat_3_1, s_hat_3_2, s_hat_3_3)

        if write:
            y_q_w_0 = y_q_0_0 + y_q_1_1 + y_q_2_2 + y_q_3_3
            y_q_w_1 = y_q_0_3 + y_q_1_2 + y_q_2_1 + y_q_3_0
            y_q_w_2 = y_q_0_2 + y_q_1_3 + y_q_2_0 + y_q_3_1
            y_q_w_3 = y_q_0_1 + y_q_1_0 + y_q_2_3 + y_q_3_2
            scales_w_0 = s_hat_0_0 + s_hat_1_1 + s_hat_2_2 + s_hat_3_3
            scales_w_1 = s_hat_0_3 + s_hat_1_2 + s_hat_2_1 + s_hat_3_0
            scales_w_2 = s_hat_0_2 + s_hat_1_3 + s_hat_2_0 + s_hat_3_1
            scales_w_3 = s_hat_0_1 + s_hat_1_0 + s_hat_2_3 + s_hat_3_2
            return y_q_w_0, y_q_w_1, y_q_w_2, y_q_w_3, \
                scales_w_0, scales_w_1, scales_w_2, scales_w_3, y_hat
        return y_res, y_q, y_hat, scales_hat
