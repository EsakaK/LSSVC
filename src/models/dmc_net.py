import torch
import math
import torch.nn as nn
from pytorch_msssim import MS_SSIM

from src.InterModules.video_net_component import ME_Spynet, GDN, flow_warp, ResBlock, bilineardownsacling
from src.entropy_models.video_entropy_models import BitEstimator, GaussianEncoder
from src.IntraModules.layers import subpel_conv3x3


class FeatureExtractor(nn.Module):
    def __init__(self, channel=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, channel, 3, stride=1, padding=1)
        self.res_block1 = ResBlock(channel)
        self.conv2 = nn.Conv2d(channel, channel, 3, stride=2, padding=1)
        self.res_block2 = ResBlock(channel)
        self.conv3 = nn.Conv2d(channel, channel, 3, stride=2, padding=1)
        self.res_block3 = ResBlock(channel)

    def forward(self, feature):
        layer1 = self.conv1(feature)
        layer1 = self.res_block1(layer1)

        layer2 = self.conv2(layer1)
        layer2 = self.res_block2(layer2)

        layer3 = self.conv3(layer2)
        layer3 = self.res_block3(layer3)

        return layer1, layer2, layer3


class MultiScaleContextFusion(nn.Module):
    def __init__(self, channel_in=64, channel_out=64):
        super().__init__()
        self.conv3_up = subpel_conv3x3(channel_in, channel_out, 2)
        self.res_block3_up = ResBlock(channel_out)
        self.conv3_out = nn.Conv2d(channel_out, channel_out, 3, padding=1)
        self.res_block3_out = ResBlock(channel_out)
        self.conv2_up = subpel_conv3x3(channel_out * 2, channel_out, 2)
        self.res_block2_up = ResBlock(channel_out)
        self.conv2_out = nn.Conv2d(channel_out * 2, channel_out, 3, padding=1)
        self.res_block2_out = ResBlock(channel_out)
        self.conv1_out = nn.Conv2d(channel_out * 2, channel_out, 3, padding=1)
        self.res_block1_out = ResBlock(channel_out)

    def forward(self, context1, context2, context3):
        context3_up = self.conv3_up(context3)
        context3_up = self.res_block3_up(context3_up)
        context3_out = self.conv3_out(context3)
        context3_out = self.res_block3_out(context3_out)
        context2_up = self.conv2_up(torch.cat((context3_up, context2), dim=1))
        context2_up = self.res_block2_up(context2_up)
        context2_out = self.conv2_out(torch.cat((context3_up, context2), dim=1))
        context2_out = self.res_block2_out(context2_out)
        context1_out = self.conv1_out(torch.cat((context2_up, context1), dim=1))
        context1_out = self.res_block1_out(context1_out)
        context1 = context1 + context1_out
        context2 = context2 + context2_out
        context3 = context3 + context3_out
        return context1, context2, context3


class ResEncoder(nn.Module):
    def __init__(self, channel_N=64, channel_M=96):
        super().__init__()
        self.conv1 = nn.Conv2d(channel_N + 3, channel_N, 3, stride=2, padding=1)
        self.gdn1 = GDN(channel_N)
        self.res1 = ResBlock(channel_N * 2, bottleneck=True, slope=0.1,
                             start_from_relu=False, end_with_relu=True)
        self.conv2 = nn.Conv2d(channel_N * 2, channel_N, 3, stride=2, padding=1)
        self.gdn2 = GDN(channel_N)
        self.res2 = ResBlock(channel_N * 2, bottleneck=True, slope=0.1,
                             start_from_relu=False, end_with_relu=True)
        self.conv3 = nn.Conv2d(channel_N * 2, channel_N, 3, stride=2, padding=1)
        self.gdn3 = GDN(channel_N)
        self.conv4 = nn.Conv2d(channel_N, channel_M, 3, stride=2, padding=1)

    def forward(self, x, context1, context2, context3):
        feature = self.conv1(torch.cat([x, context1], dim=1))
        feature = self.gdn1(feature)
        feature = self.res1(torch.cat([feature, context2], dim=1))
        feature = self.conv2(feature)
        feature = self.gdn2(feature)
        feature = self.res2(torch.cat([feature, context3], dim=1))
        feature = self.conv3(feature)
        feature = self.gdn3(feature)
        feature = self.conv4(feature)
        return feature


class ResDecoder(nn.Module):
    def __init__(self, channel_N=64, channel_M=96):
        super().__init__()
        self.up1 = subpel_conv3x3(channel_M, channel_N, 2)
        self.gdn1 = GDN(channel_N, inverse=True)
        self.up2 = subpel_conv3x3(channel_N, channel_N, 2)
        self.gdn2 = GDN(channel_N, inverse=True)
        self.res1 = ResBlock(channel_N * 2, bottleneck=True, slope=0.1,
                             start_from_relu=False, end_with_relu=True)
        self.up3 = subpel_conv3x3(channel_N * 2, channel_N, 2)
        self.gdn3 = GDN(channel_N, inverse=True)
        self.res2 = ResBlock(channel_N * 2, bottleneck=True, slope=0.1,
                             start_from_relu=False, end_with_relu=True)
        self.up4 = subpel_conv3x3(channel_N * 2, 32, 2)

    def forward(self, x, context2, context3):
        feature = self.up1(x)
        feature = self.gdn1(feature)
        feature = self.up2(feature)
        feature = self.gdn2(feature)
        feature = self.res1(torch.cat([feature, context3], dim=1))
        feature = self.up3(feature)
        feature = self.gdn3(feature)
        feature = self.res2(torch.cat([feature, context2], dim=1))
        feature = self.up4(feature)
        return feature


class TemporalPriorEncoder(nn.Module):
    def __init__(self, channel_N=64, channel_M=96):
        super().__init__()
        self.conv1 = nn.Conv2d(channel_N, channel_N, 3, stride=2, padding=1)
        self.gdn1 = GDN(channel_N)
        self.conv2 = nn.Conv2d(channel_N * 2, channel_M, 3, stride=2, padding=1)
        self.gdn2 = GDN(channel_M)
        self.conv3 = nn.Conv2d(channel_M + channel_N, channel_M * 3 // 2, 3, stride=2, padding=1)
        self.gdn3 = GDN(channel_M * 3 // 2)
        self.conv4 = nn.Conv2d(channel_M * 3 // 2, channel_M * 2, 3, stride=2, padding=1)

    def forward(self, context1, context2, context3):
        feature = self.conv1(context1)
        feature = self.gdn1(feature)
        feature = self.conv2(torch.cat([feature, context2], dim=1))
        feature = self.gdn2(feature)
        feature = self.conv3(torch.cat([feature, context3], dim=1))
        feature = self.gdn3(feature)
        feature = self.conv4(feature)
        return feature


class ReconGeneration(nn.Module):
    def __init__(self, ctx_channel=64, res_channel=32, channel=64):
        super().__init__()
        self.feature_conv = nn.Sequential(
            nn.Conv2d(ctx_channel + res_channel, channel, 3, stride=1, padding=1),
            ResBlock(channel),
            ResBlock(channel),
        )
        self.recon_conv = nn.Conv2d(channel, 3, 3, stride=1, padding=1)

    def forward(self, ctx, res):
        feature = self.feature_conv(torch.cat((ctx, res), dim=1))
        recon = self.recon_conv(feature)
        return feature, recon


class DMC(nn.Module):
    def __init__(self, me_pretrain_dir=None, mv_enc_dec_pretrain_path=None, win_size=11):
        super().__init__()
        self.DMC_version = '1.8'

        channel_mv = 128
        channel_N = 64
        channel_M = 96

        self.channel_mv = channel_mv
        self.channel_N = channel_N
        self.channel_M = channel_M

        self.optic_flow = ME_Spynet()

        self.mv_encoder = nn.Sequential(
            nn.Conv2d(2, channel_mv, 3, stride=2, padding=1),
            GDN(channel_mv),
            ResBlock(channel_mv, start_from_relu=False),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(channel_mv, channel_mv, 3, stride=2, padding=1),
            GDN(channel_mv),
            ResBlock(channel_mv, start_from_relu=False),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(channel_mv, channel_mv, 3, stride=2, padding=1),
            GDN(channel_mv),
            ResBlock(channel_mv, start_from_relu=False),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(channel_mv, channel_mv, 3, stride=2, padding=1),
        )

        self.mv_prior_encoder = nn.Sequential(
            nn.Conv2d(channel_mv, channel_N, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_N, channel_N, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_N, channel_N, 3, stride=2, padding=1),
        )

        self.mv_prior_decoder = nn.Sequential(
            nn.ConvTranspose2d(channel_N, channel_mv, 3,
                               stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(channel_mv, channel_mv * 3 // 2, 3,
                               stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(channel_mv * 3 // 2, channel_mv * 2, 3, stride=1, padding=1)
        )

        self.mv_decoder = nn.Sequential(
            nn.ConvTranspose2d(channel_mv, channel_mv, 3,
                               stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            ResBlock(channel_mv, start_from_relu=False),
            GDN(channel_mv, inverse=True),
            nn.ConvTranspose2d(channel_mv, channel_mv, 3,
                               stride=2, padding=1, output_padding=1),
            GDN(channel_mv, inverse=True),
            nn.ConvTranspose2d(channel_mv, channel_mv, 3,
                               stride=2, padding=1, output_padding=1),
            GDN(channel_mv, inverse=True),
            nn.ConvTranspose2d(channel_mv, 2, 3, stride=2, padding=1, output_padding=1),
        )

        self.feature_adaptor_I = nn.Conv2d(3, channel_N, 3, stride=1, padding=1)
        self.feature_adaptor_P = nn.Conv2d(channel_N, channel_N, 1)
        self.feature_extractor = FeatureExtractor()
        self.context_fusion_net = MultiScaleContextFusion()

        self.res_encoder = ResEncoder()

        self.res_prior_encoder = nn.Sequential(
            nn.Conv2d(channel_M, channel_N, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_N, channel_N, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_N, channel_N, 3, stride=2, padding=1),
        )

        self.res_prior_decoder = nn.Sequential(
            nn.ConvTranspose2d(channel_N, channel_M, 3,
                               stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(channel_M, channel_M * 3 // 2, 3,
                               stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(channel_M * 3 // 2, channel_M * 2, 3, stride=1, padding=1)
        )

        self.temporal_prior_encoder = TemporalPriorEncoder()

        self.res_entropy_parameter = nn.Sequential(
            nn.Conv2d(channel_M * 12 // 3, channel_M * 10 // 3, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_M * 10 // 3, channel_M * 8 // 3, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_M * 8 // 3, channel_M * 6 // 3, 3, stride=1, padding=1),
        )

        self.res_decoder = ResDecoder()

        self.recon_generation_net = ReconGeneration()

        self.bit_estimator_z = BitEstimator(channel_N)
        self.bit_estimator_z_mv = BitEstimator(channel_N)
        self.gaussian_encoder = GaussianEncoder()
        self.ms_ssim_loss = MS_SSIM(data_range=1.0, win_size=win_size)

        self._initialize_weights()

        self.inter_module_names = ['mv_encoder', 'mv_decoder',
                                   'mv_prior_encoder', 'mv_prior_decoder',
                                   'bit_estimator_z_mv', 'optic_flow']

        if mv_enc_dec_pretrain_path is not None:
            self.load_mv_enc_dec_pretrain(mv_enc_dec_pretrain_path)
            print(f"loaded mv_enc_dec weights from: {mv_enc_dec_pretrain_path}")
        # elif me_pretrain_path is not None:
        #     self.load_me_pretrain(me_pretrain_path)
        #     print(f"loaded optic flow weights from: {me_pretrain_path}")

        self.previous_frame_recon = None
        self.previous_frame_feature = None

    def control_is_prediction_parameter(self, is_prediction=True, is_train=True):
        for name, value in self.named_parameters():
            include_inter = False
            for inter_name in self.inter_module_names:
                if inter_name in name:
                    include_inter = True
            if include_inter == is_prediction:
                value.requires_grad = is_train

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                torch.nn.init.xavier_normal_(m.weight, math.sqrt(2))
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.01)

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

    def load_me_pretrain(self, path):
        ckpt = torch.load(path, map_location=torch.device('cpu'))
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        s = {}
        for k in ckpt:
            if k.startswith('optic_flow.'):
                s[k[len('optic_flow.'):]] = ckpt[k]
        self.optic_flow.load_state_dict(s)

    def load_mv_enc_dec_pretrain(self, path):
        ckpt = torch.load(path, map_location=torch.device('cpu'))
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        ckpt_bit_estimator_z_mv = {}
        ckpt_mv_encoder = {}
        ckpt_mv_decoder = {}
        ckpt_mv_prior_encoder = {}
        ckpt_mv_prior_decoder = {}
        ckpt_optic_flow = {}
        for k in ckpt:
            if k.startswith('bit_estimator_z_mv.'):
                ckpt_bit_estimator_z_mv[k[len('bit_estimator_z_mv.'):]] = ckpt[k]
            elif k.startswith('mv_encoder.'):
                ckpt_mv_encoder[k[len('mv_encoder.'):]] = ckpt[k]
            elif k.startswith('mv_decoder.'):
                ckpt_mv_decoder[k[len('mv_decoder.'):]] = ckpt[k]
            elif k.startswith('mv_prior_encoder.'):
                ckpt_mv_prior_encoder[k[len('mv_prior_encoder.'):]] = ckpt[k]
            elif k.startswith('mv_prior_decoder.'):
                ckpt_mv_prior_decoder[k[len('mv_prior_decoder.'):]] = ckpt[k]
            elif k.startswith('optic_flow.'):
                ckpt_optic_flow[k[len('optic_flow.'):]] = ckpt[k]
        self.bit_estimator_z_mv.load_state_dict(ckpt_bit_estimator_z_mv)
        self.mv_encoder.load_state_dict(ckpt_mv_encoder)
        self.mv_decoder.load_state_dict(ckpt_mv_decoder)
        self.mv_prior_encoder.load_state_dict(ckpt_mv_prior_encoder)
        self.mv_prior_decoder.load_state_dict(ckpt_mv_prior_decoder)
        self.optic_flow.load_state_dict(ckpt_optic_flow)

    def multi_scale_feature_extractor(self, ref, feature):
        if feature is None:
            feature = self.feature_adaptor_I(ref)
        else:
            feature = self.feature_adaptor_P(feature)
        return self.feature_extractor(feature)

    def motion_compensation(self, ref, feature, mv):
        warpframe = flow_warp(ref, mv)
        mv2 = bilineardownsacling(mv) / 2
        mv3 = bilineardownsacling(mv2) / 2
        ref_feature1, ref_feature2, ref_feature3 = self.multi_scale_feature_extractor(ref, feature)
        context1 = flow_warp(ref_feature1, mv)
        context2 = flow_warp(ref_feature2, mv2)
        context3 = flow_warp(ref_feature3, mv3)
        context1, context2, context3 = self.context_fusion_net(context1, context2, context3)
        return context1, context2, context3, warpframe, ref_feature1, ref_feature2, ref_feature3

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

    def encode_decode(self, x, ref_frame, ref_feature, output_path=None,
                      pic_width=None, pic_height=None,
                      rdo=False, rdo_opt=None, profile_decoding=False):
        # pic_width and pic_height may be different from x's size. x here is after padding
        # x_hat has the same size with x
        if rdo or output_path is not None:
            return self.encode_decode_extend(x, ref_frame, ref_feature, output_path,
                                             pic_width, pic_height, rdo, rdo_opt,
                                             profile_decoding)
        encoded = self.forward_one_frame(x, ref_frame, ref_feature)
        result = {
            'dpb': {
                "ref_frame": encoded['recon_image'],
                "ref_feature": encoded['feature']
            },
            "bit": encoded['bit'].item(),
            "decoding_time": 0,
        }
        return result

    def quant(self, x, force_detach=False):
        if self.training or force_detach:
            n = torch.round(x) - x
            n = n.clone().detach()
            return x + n
        else:
            return torch.round(x)

    @staticmethod
    def add_noise(x):
        noise = torch.nn.init.uniform_(torch.zeros_like(x), -0.5, 0.5)
        return x + noise

    def get_inter_layer_information(self, x, ref_frame, ref_feature):
        est_mv = self.optic_flow(x, ref_frame)
        mv_y = self.mv_encoder(est_mv)
        mv_z = self.mv_prior_encoder(mv_y)
        mv_z_hat = self.quant(mv_z)
        mv_params = self.mv_prior_decoder(mv_z_hat)
        mv_scales_hat, mv_means_hat = mv_params.chunk(2, 1)

        mv_y_res = mv_y - mv_means_hat
        mv_y_q = self.quant(mv_y_res)
        mv_y_hat = mv_y_q + mv_means_hat

        mv_hat = self.mv_decoder(mv_y_hat)
        context1, context2, context3, _, _, _, _ = self.motion_compensation(
            ref_frame, ref_feature, mv_hat)

        y = self.res_encoder(x, context1, context2, context3)
        z = self.res_prior_encoder(y)
        z_hat = self.quant(z)
        hierarchical_params = self.res_prior_decoder(z_hat)
        temporal_params = self.temporal_prior_encoder(context1, context2, context3)

        params = torch.cat((temporal_params, hierarchical_params), dim=1)
        gaussian_params = self.res_entropy_parameter(params)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        y_res = y - means_hat
        y_q = self.quant(y_res)
        y_hat = y_q + means_hat

        recon_image_feature = self.res_decoder(y_hat, context2, context3)
        feature, recon_image = self.recon_generation_net(recon_image_feature, context1)

        mse_loss = torch.mean((recon_image - x).pow(2))

        if self.training:
            y_for_bit = self.add_noise(y_res)
            mv_y_for_bit = self.add_noise(mv_y_res)
            z_for_bit = self.add_noise(z)
            mv_z_for_bit = self.add_noise(mv_z)
        else:
            y_for_bit = y_q
            mv_y_for_bit = mv_y_q
            z_for_bit = z_hat
            mv_z_for_bit = mv_z_hat
        total_bits_y, _ = self.get_y_bits_probs(y_for_bit, scales_hat)
        total_bits_mv_y, _ = self.get_y_bits_probs(mv_y_for_bit, mv_scales_hat)
        total_bits_z, _ = self.get_z_bits_probs(z_for_bit, self.bit_estimator_z)
        total_bits_mv_z, _ = self.get_z_bits_probs(mv_z_for_bit, self.bit_estimator_z_mv)

        im_shape = x.size()
        pixel_num = im_shape[0] * im_shape[2] * im_shape[3]
        total_bits = total_bits_y + total_bits_z + total_bits_mv_y + total_bits_mv_z
        bpp = total_bits / pixel_num

        return {"bpp": bpp,
                "bits": total_bits,
                "mse_loss": mse_loss,
                "recon_image": recon_image,
                "feature": feature,
                "y_hat": y_hat,
                "mv_hat": mv_hat,
                "temporal_params": temporal_params,
                "y": y,
                "z": z,
                "mv_y": mv_y,
                "mv_z": mv_z
                }
