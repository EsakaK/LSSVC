import time
import warnings
import torch
from torch.autograd import Variable
from .dmc_net import DMC
from src.entropy_models.video_entropy_models import EntropyCoder
from src.utils.stream_helper import get_downsampled_shape, encode_p, decode_p, filesize


class DMCExtend(DMC):
    '''
    This class needs to be modified if you want to write bitstreams or performing RDO
    '''

    def __init__(self):
        super().__init__()

        self.entropy_coder = None
        self.decoding_profiling = {
            "frames": 0,
            "overall": 0,
            "entropy_dec_mv_z": 0,
            "mv_y_prior_dec": 0,
            "entropy_dec_mv_y": 0,
            "mv_dec": 0,
            "motion_compensation_ctx_refine": 0,
            "entropy_dec_z": 0,
            "y_h_prior_dec": 0,
            "y_t_prior": 0,
            "y_prior": 0,
            "entropy_dec_y": 0,
            "res_dec": 0,
            "rec_generation": 0,
        }

    def reset_decoding_profiling(self):
        for k in self.decoding_profiling:
            self.decoding_profiling[k] = 0

    def get_average_decoding_profiling(self):
        result = {}
        for k in self.decoding_profiling:
            if k == "frames":
                result[k] = self.decoding_profiling[k]
                continue
            result[k] = self.decoding_profiling[k] / self.decoding_profiling["frames"]
        return result

    def update(self, force=False):
        self.entropy_coder = EntropyCoder()
        self.bit_estimator_z_mv.update(force=force, entropy_coder=self.entropy_coder)
        self.bit_estimator_z.update(force=force, entropy_coder=self.entropy_coder)
        self.gaussian_encoder.update(force=force, entropy_coder=self.entropy_coder)

    def compress(self, x, dpb):
        ref_frame = dpb['ref_frame_bl']
        ref_feature = dpb['ref_feature_bl']
        est_mv = self.optic_flow(x, ref_frame)
        mv_y = self.mv_encoder(est_mv)
        mv_z = self.mv_prior_encoder(mv_y)
        mv_z_hat = self.quant(mv_z)  # mv hyper encode
        mv_params = self.mv_prior_decoder(mv_z_hat)
        mv_scales_hat, mv_means_hat = mv_params.chunk(2, 1)

        mv_y_res = mv_y - mv_means_hat
        mv_y_q = self.quant(mv_y_res)  # mv encode
        mv_y_hat = mv_y_q + mv_means_hat

        mv_hat = self.mv_decoder(mv_y_hat)
        context1, context2, context3, _, _, _, _ = self.motion_compensation(
            ref_frame, ref_feature, mv_hat)
        y = self.res_encoder(x, context1, context2, context3)
        z = self.res_prior_encoder(y)
        z_hat = self.quant(z)  # res hyper encoder
        hierarchical_params = self.res_prior_decoder(z_hat)
        temporal_params = self.temporal_prior_encoder(context1, context2, context3)

        params = torch.cat((temporal_params, hierarchical_params), dim=1)
        gaussian_params = self.res_entropy_parameter(params)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        y_res = y - means_hat
        y_q = self.quant(y_res)  # res encode
        y_hat = y_q + means_hat

        # arithmetic coding
        self.entropy_coder.reset_encoder()
        _ = self.bit_estimator_z_mv.encode(mv_z_hat)
        _ = self.gaussian_encoder.encode(mv_y_q, mv_scales_hat)
        _ = self.bit_estimator_z.encode(z_hat)
        _ = self.gaussian_encoder.encode(y_q, scales_hat)
        string = self.entropy_coder.flush_encoder()

        recon_image_feature = self.res_decoder(y_hat, context2, context3)
        feature, recon_image = self.recon_generation_net(recon_image_feature, context1)
        return {
            "string": string,
            "dpb": {
                "ref_frame_bl": recon_image,
                "ref_feature_bl": feature,
                "y_hat_bl": y_hat,
                "mv_hat_bl": mv_hat
            }
        }

    def decompress(self, string, height, width, dpb):
        ref_frame = dpb['ref_frame_bl']
        ref_feature = dpb['ref_feature_bl']
        self.entropy_coder.set_stream(string)
        device = next(self.parameters()).device
        mv_z_size = get_downsampled_shape(height, width, 64)
        mv_z_hat = self.bit_estimator_z_mv.decode_stream(mv_z_size)
        mv_z_hat = mv_z_hat.to(device)
        mv_params = self.mv_prior_decoder(mv_z_hat)
        mv_scales_hat, mv_means_hat = mv_params.chunk(2, 1)
        mv_y_q = self.gaussian_encoder.decode_stream(mv_scales_hat)
        mv_y_q = mv_y_q.to(device)
        mv_y_hat = mv_y_q + mv_means_hat

        mv_hat = self.mv_decoder(mv_y_hat)
        context1, context2, context3, _, _, _, _ = self.motion_compensation(
            ref_frame, ref_feature, mv_hat)

        z_size = get_downsampled_shape(height, width, 64)
        z_hat = self.bit_estimator_z.decode_stream(z_size)
        z_hat = z_hat.to(device)
        hierarchical_params = self.res_prior_decoder(z_hat)
        temporal_params = self.temporal_prior_encoder(context1, context2, context3)
        params = torch.cat((temporal_params, hierarchical_params), dim=1)
        gaussian_params = self.res_entropy_parameter(params)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_q = self.gaussian_encoder.decode_stream(scales_hat)
        y_q = y_q.to(device)
        y_hat = y_q + means_hat

        recon_image_feature = self.res_decoder(y_hat, context2, context3)
        feature, recon_image = self.recon_generation_net(recon_image_feature, context1)
        recon_image = recon_image.clamp(0, 1)

        return {
            "dpb": {
                "ref_frame_bl": recon_image,
                "ref_feature_bl": feature,
                "y_hat_bl": y_hat,
                "mv_hat_bl": mv_hat
            }
        }

    def encode_decode_extend(self, x, dpb, output_path=None, pic_width=None, pic_height=None):
        device = x.device
        torch.cuda.synchronize(device=device)

        t0 = time.time()
        encoded = self.compress(x, dpb)
        encode_p(encoded['string'], output_path)
        bits = filesize(output_path) * 8
        torch.cuda.synchronize(device=device)
        t1 = time.time()

        string = decode_p(output_path)
        decoded = self.decompress(string, pic_height, pic_width, dpb)
        torch.cuda.synchronize(device=device)
        t2 = time.time()

        dpb = decoded['dpb']
        result = {
            "dpb": dpb,
            "bit": bits,
            "encoding_time": t1 - t0,
            "decoding_time": t2 - t1,
        }

        return result
