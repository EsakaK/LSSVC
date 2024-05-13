import time
import torch
from .LSSVC_net import LSSVC
from src.entropy_models.video_entropy_models import EntropyCoder
from src.utils.stream_helper import get_downsampled_shape, encode_p, decode_p, filesize


class LSSVC_extend(LSSVC):
    """
    This class needs to be modified if you want to write bitstreams or performing RDO
    """

    def __init__(self):
        super().__init__()
        self.entropy_coder = None

    def update(self, force=False):
        self.entropy_coder = EntropyCoder()
        self.bit_estimator_z_mv.update(force=force, entropy_coder=self.entropy_coder)
        self.bit_estimator_z.update(force=force, entropy_coder=self.entropy_coder)
        self.gaussian_encoder.update(force=force, entropy_coder=self.entropy_coder)
        self.base_layer_model.update(force=force)

    def compress(self, x, dpb):
        ref_frame = dpb['ref_frame_el']
        ref_feature = dpb['ref_feature_el']
        texture_bl = dpb['texture']
        y_hat_bl = dpb['y_hat_bl']
        mv_hat_bl = dpb['mv_hat_bl']
        # depadded
        texture = self.get_depadded_feature(texture_bl)  # texture resample
        mv_bl_hat = self.get_depadded_feature(mv_hat_bl)  # motion resample
        y_bl_hat = self.get_depadded_feature(y_hat_bl, p=16)
        # flow part
        mv_upsample = self.motion_resampling(mv_bl_hat)
        mv_ctx_prior = self.mv_ctx_prior_encoder(mv_upsample)
        mv_ctx = self.mv_ctx_transform(mv_upsample)

        mv = self.optic_flow(x, ref_frame)
        mv_y = self.mv_encoder(mv, mv_ctx)
        mv_z = self.mv_prior_encoder(mv_y)
        mv_z_hat = self.quant(mv_z)  # mv hyper encode
        mv_hyper_prior = self.mv_prior_decoder(mv_z_hat)
        mv_params = self.mv_prior_fusion(torch.cat([mv_hyper_prior, mv_ctx_prior], dim=1))
        mv_scales_hat, mv_means_hat = mv_params.chunk(2, 1)

        mv_y_res = mv_y - mv_means_hat
        mv_y_q = self.quant(mv_y_res)  # mv res encode
        mv_y_hat = mv_y_q + mv_means_hat

        mv_hat = self.mv_decoder(mv_y_hat, mv_ctx)
        # res part
        context1, context2, context3, warp_frame, _ = self.hybrid_temporal_layer_context_fusion(texture, mv_hat, ref_frame, ref_feature)
        y = self.res_encoder(x, context1, context2, context3)
        z = self.res_prior_encoder(y)
        z_hat = self.quant(z)
        hierarchical_params = self.res_prior_decoder(z_hat)
        temporal_params_el = self.temporal_prior_encoder(context3)
        params = self.hybrid_temporal_layer_prior_parameters(hierarchical_params, temporal_params_el, y_bl_hat)
        y_q_w_0, y_q_w_1, y_q_w_2, y_q_w_3, scales_w_0, scales_w_1, scales_w_2, scales_w_3, y_hat = self.compress_four_part_prior(y, params,
                                                                                                                                  self.y_spatial_prior_adaptor_1,
                                                                                                                                  self.y_spatial_prior_adaptor_2,
                                                                                                                                  self.y_spatial_prior_adaptor_3,
                                                                                                                                  self.y_spatial_prior)
        # entropy coding
        self.entropy_coder.reset_encoder()
        _ = self.bit_estimator_z_mv.encode(mv_z_hat)
        _ = self.gaussian_encoder.encode(mv_y_q, mv_scales_hat)
        _ = self.bit_estimator_z.encode(z_hat)
        _ = self.gaussian_encoder.encode(y_q_w_0, scales_w_0)
        _ = self.gaussian_encoder.encode(y_q_w_1, scales_w_1)
        _ = self.gaussian_encoder.encode(y_q_w_2, scales_w_2)
        _ = self.gaussian_encoder.encode(y_q_w_3, scales_w_3)
        string = self.entropy_coder.flush_encoder()

        recon_image_feature = self.res_decoder(y_hat, context2, context3)
        feature, recon_image = self.recon_generation_net(recon_image_feature, context1)
        return {
            "string": string,
            "dpb": {
                "ref_frame_el": recon_image,
                "ref_feature_el": feature,
                "warp_frame": warp_frame,
                "mv_hat": mv_hat
            }
        }

    def decompress(self, string, height, width, dpb):
        ref_frame = dpb['ref_frame_el']
        ref_feature = dpb['ref_feature_el']
        texture_bl = dpb['texture']
        y_hat_bl = dpb['y_hat_bl']
        mv_hat_bl = dpb['mv_hat_bl']

        # Mv Inter-Layer-Processing
        # depadded
        texture = self.get_depadded_feature(texture_bl)  # texture resample
        mv_bl_hat = self.get_depadded_feature(mv_hat_bl)  # motion resample
        y_bl_hat = self.get_depadded_feature(y_hat_bl, p=16)
        mv_upsample = self.motion_resampling(mv_bl_hat)
        mv_ctx_prior = self.mv_ctx_prior_encoder(mv_upsample)
        mv_ctx = self.mv_ctx_transform(mv_upsample)
        # encode mv
        self.entropy_coder.set_stream(string)
        device = next(self.parameters()).device
        mv_z_size = get_downsampled_shape(height, width, 64)
        mv_z_hat = self.bit_estimator_z_mv.decode_stream(mv_z_size)
        mv_z_hat = mv_z_hat.to(device)

        mv_hyper_prior = self.mv_prior_decoder(mv_z_hat)
        mv_params = self.mv_prior_fusion(torch.cat([mv_hyper_prior, mv_ctx_prior], dim=1))
        mv_scales_hat, mv_means_hat = mv_params.chunk(2, 1)
        mv_y_q = self.gaussian_encoder.decode_stream(mv_scales_hat)
        mv_y_q = mv_y_q.to(device)
        mv_y_hat = mv_y_q + mv_means_hat
        mv_hat = self.mv_decoder(mv_y_hat, mv_ctx)

        context1, context2, context3, _, _ = self.hybrid_temporal_layer_context_fusion(texture, mv_hat, ref_frame, ref_feature)

        z_size = get_downsampled_shape(height, width, 64)
        z_hat = self.bit_estimator_z.decode_stream(z_size)
        z_hat = z_hat.to(device)

        hierarchical_params = self.res_prior_decoder(z_hat)
        temporal_params = self.temporal_prior_encoder(context3)
        params = self.hybrid_temporal_layer_prior_parameters(hierarchical_params, temporal_params, y_bl_hat)
        # ctx decode
        y_hat = self.decompress_four_part_prior(params,
                                                self.y_spatial_prior_adaptor_1,
                                                self.y_spatial_prior_adaptor_2,
                                                self.y_spatial_prior_adaptor_3,
                                                self.y_spatial_prior)

        recon_image_feature = self.res_decoder(y_hat, context2, context3)
        feature, recon_image = self.recon_generation_net(recon_image_feature, context1)

        return {
            "dpb": {
                "ref_frame_el": recon_image,
                "ref_feature_el": feature
            }
        }

    def encode_decode_extend(self, x_bl, x_el, dpb,
                             output_path_bl=None, output_path_el=None,
                             pic_width=None, pic_height=None, pic_width_bl=None, pic_height_bl=None):
        # encode-decode base layer
        bl_encode_result = self.base_layer_model.encode_decode_extend(x_bl, dpb, output_path_bl, pic_width_bl, pic_height_bl)
        layer_dpb = bl_encode_result['dpb']
        bit_bl = bl_encode_result['bit']
        encoding_time = bl_encode_result['encoding_time']
        decoding_time = bl_encode_result['decoding_time']
        # encode-decode enhance layer
        dpb['texture'] = layer_dpb['ref_feature_bl']
        dpb['y_hat_bl'] = layer_dpb['y_hat_bl']
        dpb['mv_hat_bl'] = layer_dpb['mv_hat_bl']

        device = x_el.device
        torch.cuda.synchronize(device=device)

        t0 = time.time()
        encoded = self.compress(x_el, dpb)
        encode_p(encoded['string'], output_path_el)
        bits = filesize(output_path_el) * 8
        torch.cuda.synchronize(device=device)
        t1 = time.time()

        string = decode_p(output_path_el)
        decoded = self.decompress(string, pic_height, pic_width, dpb)
        torch.cuda.synchronize(device=device)
        t2 = time.time()

        dpb = {
            'ref_frame_bl': layer_dpb['ref_frame_bl'],
            'ref_feature_bl': layer_dpb['ref_feature_bl'],
            'ref_frame_el': decoded['dpb']['ref_frame_el'],
            'ref_feature_el': decoded['dpb']['ref_feature_el']
        }
        result = {
            "dpb": dpb,
            "bit_bl": bit_bl,
            "bit_el": bits,
            "encoding_time_EL": (t1 - t0),
            "decoding_time_EL": (t2 - t1),
            "encoding_time_BL": encoding_time,
            "decoding_time_BL": decoding_time,
            "mv_hat": encoded['dpb']['mv_hat'],
            "warp_frame": encoded['dpb']['warp_frame']
        }

        return result

    def compress_four_part_prior(self, y, common_params,
                                 y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                 y_spatial_prior_adaptor_3, y_spatial_prior):
        return self.forward_four_part_prior(y, common_params,
                                            y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                            y_spatial_prior_adaptor_3, y_spatial_prior, write=True)

    def decompress_four_part_prior(self, common_params,
                                   y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                                   y_spatial_prior_adaptor_3, y_spatial_prior):
        scales, means = self.separate_prior(common_params)
        dtype = means.dtype
        device = means.device
        _, _, H, W = means.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_four_parts(H, W, dtype, device)

        scales_0, scales_1, scales_2, scales_3 = scales.chunk(4, 1)
        means_0, means_1, means_2, means_3 = means.chunk(4, 1)

        scales_r = scales_0 * mask_0 + scales_1 * mask_1 + scales_2 * mask_2 + scales_3 * mask_3
        y_q_r = self.gaussian_encoder.decode_stream(scales_r)
        y_q_r = y_q_r.to(device)
        y_hat_0_0 = (y_q_r + means_0) * mask_0
        y_hat_1_1 = (y_q_r + means_1) * mask_1
        y_hat_2_2 = (y_q_r + means_2) * mask_2
        y_hat_3_3 = (y_q_r + means_3) * mask_3
        y_hat_curr_step = torch.cat((y_hat_0_0, y_hat_1_1, y_hat_2_2, y_hat_3_3), dim=1)
        y_hat_so_far = y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales_0, scales_1, scales_2, scales_3, means_0, means_1, means_2, means_3 = \
            y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(8, 1)
        scales_r = scales_0 * mask_3 + scales_1 * mask_2 + scales_2 * mask_1 + scales_3 * mask_0
        y_q_r = self.gaussian_encoder.decode_stream(scales_r)
        y_q_r = y_q_r.to(device)
        y_hat_0_3 = (y_q_r + means_0) * mask_3
        y_hat_1_2 = (y_q_r + means_1) * mask_2
        y_hat_2_1 = (y_q_r + means_2) * mask_1
        y_hat_3_0 = (y_q_r + means_3) * mask_0
        y_hat_curr_step = torch.cat((y_hat_0_3, y_hat_1_2, y_hat_2_1, y_hat_3_0), dim=1)
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales_0, scales_1, scales_2, scales_3, means_0, means_1, means_2, means_3 = \
            y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(8, 1)
        scales_r = scales_0 * mask_2 + scales_1 * mask_3 + scales_2 * mask_0 + scales_3 * mask_1
        y_q_r = self.gaussian_encoder.decode_stream(scales_r)
        y_q_r = y_q_r.to(device)
        y_hat_0_2 = (y_q_r + means_0) * mask_2
        y_hat_1_3 = (y_q_r + means_1) * mask_3
        y_hat_2_0 = (y_q_r + means_2) * mask_0
        y_hat_3_1 = (y_q_r + means_3) * mask_1
        y_hat_curr_step = torch.cat((y_hat_0_2, y_hat_1_3, y_hat_2_0, y_hat_3_1), dim=1)
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales_0, scales_1, scales_2, scales_3, means_0, means_1, means_2, means_3 = \
            y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(8, 1)
        scales_r = scales_0 * mask_1 + scales_1 * mask_0 + scales_2 * mask_3 + scales_3 * mask_2
        y_q_r = self.gaussian_encoder.decode_stream(scales_r)
        y_q_r = y_q_r.to(device)
        y_hat_0_1 = (y_q_r + means_0) * mask_1
        y_hat_1_0 = (y_q_r + means_1) * mask_0
        y_hat_2_3 = (y_q_r + means_2) * mask_3
        y_hat_3_2 = (y_q_r + means_3) * mask_2
        y_hat_curr_step = torch.cat((y_hat_0_1, y_hat_1_0, y_hat_2_3, y_hat_3_2), dim=1)
        y_hat_so_far = y_hat_so_far + y_hat_curr_step

        y_hat = y_hat_so_far

        return y_hat
