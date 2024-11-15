import argparse
import os
import concurrent.futures
import json
import multiprocessing
import time
import warnings

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from src.models.IntraSS import IntraSS
from src.models.LSSVC_net_extend import LSSVC_extend
from src.utils.common import str2bool, filter_dict, round_to_even
from src.utils.visualization import flow_to_image
from src.utils.video_reader import YUVReader
from src.utils.functional import ycbcr420_to_rgb, rgb_to_ycbcr420
from src.utils.common import get_interlayer_padding, inverse_padding_size
from tqdm import tqdm
from src.utils.metric import calc_msssim
from pytorch_msssim import ms_ssim
from src.utils import imresize

warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")

ratio_factor_dict = {
    'x1': 1.0,
    'x1_5': 1.5,
    'x2': 2.0,
    'x3': 3.0,
    'x4': 4.0
}


def parse_args():
    parser = argparse.ArgumentParser(description="Example testing script")

    parser.add_argument('--i_frame_model_name', type=str, default="IntraNoAR")
    parser.add_argument('--i_frame_model_path', type=str, nargs="+")

    parser.add_argument("--force_intra", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--force_frame_num", type=int, default=-1)
    parser.add_argument("--force_intra_period", type=int, default=-1)
    parser.add_argument("--intra_rdo", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--intra_lmbda', type=float, nargs="+")
    parser.add_argument("--intra_rdo_iter_to_exit", type=int, default=60)
    parser.add_argument("--intra_rdo_iter_to_reduce", type=int, default=20)
    parser.add_argument('--model_path', type=str, nargs="+")
    parser.add_argument("--inter_mv_rdo", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--inter_feature_rdo", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--inter_lmbda', type=float, nargs="+")
    parser.add_argument("--inter_mv_rdo_iter_to_exit", type=int, default=60)
    parser.add_argument("--inter_mv_rdo_iter_to_reduce", type=int, default=20)
    parser.add_argument("--inter_feature_rdo_iter_to_exit", type=int, default=60)
    parser.add_argument("--inter_feature_rdo_iter_to_reduce", type=int, default=20)

    parser.add_argument('--test_config', type=str, required=True)
    parser.add_argument("--worker", "-w", type=int, default=1, help="worker number")
    parser.add_argument("--cuda", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--cuda_device", default=None,
                        help="the cuda device used, e.g., 0; 0,1; 1,2,3; etc.")
    parser.add_argument('--write_stream', type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--stream_path', type=str, default="out_bin")
    parser.add_argument('--save_decoded_frame', type=str2bool, default=False)
    parser.add_argument('--save_decoded_mv', type=str2bool, default=False)
    parser.add_argument('--save_warp_frame', type=str2bool, default=False)
    parser.add_argument('--save_decoded_context', type=str2bool, default=False)
    parser.add_argument('--decoded_frame_path', type=str, default='decoded_frames')
    parser.add_argument('--decoded_mv_path', type=str, default='decoded_mv')
    parser.add_argument('--warp_frame_path', type=str, default='warp_frame')
    parser.add_argument('--decoded_context_path', type=str, default='decoded_context')
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--decoding_profiling', type=str2bool, default=False)
    parser.add_argument('--verbose', type=int, default=0)
    # Add
    parser.add_argument('--model_name', type=str, default="LSSVC_net")

    args = parser.parse_args()
    return args


def save_torch_image(img, save_path):
    img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8)
    # write_img_time = time.time()
    Image.fromarray(img).save(save_path)
    # write_img_time2 = time.time()
    # print(f'write_img_time:{write_img_time2-write_img_time}')


def save_torch_mv(mv, save_path):
    mv = mv.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    # print(mv.shape)
    mv = flow_to_image(mv)
    mv = mv.astype(np.uint8)
    Image.fromarray(mv).save(save_path)


def np_image_to_tensor(img):
    image = torch.from_numpy(img).type(torch.FloatTensor)
    image = image.unsqueeze(0)
    return image


def mse2PSNR(mse, data_range=1):
    if mse > 1e-10:
        psnr = 10 * np.log10(data_range * data_range / mse)
    else:
        psnr = 999.9
    return psnr


def PSNR(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    psnr = 10 * torch.log10(1.0 / mse)
    return psnr.item()


def run_test(video_net, i_frame_net, args_dict, device):
    frame_num = args_dict['frame_num']
    gop_size = args_dict['gop_size']
    write_stream = 'write_stream' in args_dict and args_dict['write_stream']
    save_decoded_frame = 'save_decoded_frame' in args_dict and args_dict['save_decoded_frame']
    save_decoded_mv = 'save_decoded_mv' in args_dict and args_dict['save_decoded_mv']
    save_warp_frame = 'save_warp_frame' in args_dict and args_dict['save_warp_frame']
    ratio = args_dict['ratio']
    scale_factor = ratio_factor_dict[ratio]

    yuv_path_EL = args_dict['yuv_path_el']
    height_EL, width_EL = args_dict['x1']['height'], args_dict['x1']['width']
    print(f"Testing on sequence {os.path.basename(args_dict['video_path'])}")
    frame_types = []
    BL_YUV_psnr = []
    BL_Y_psnr = []
    BL_U_psnr = []
    BL_V_psnr = []
    EL_YUV_psnr = []
    EL_Y_psnr = []
    EL_U_psnr = []
    EL_V_psnr = []
    BL_rgb_psnr = []
    EL_rgb_psnr = []
    BL_SSIM = []
    EL_SSIM = []
    BL_rgb_SSIM = []
    EL_rgb_SSIM = []
    BL_bits = []
    EL_bits = []

    start_time = time.time()
    p_frame_number = 0
    i_frame_number = 0
    overall_encoding_time_BL = 0
    overall_decoding_time_BL = 0
    overall_encoding_time_EL = 0
    overall_decoding_time_EL = 0

    bin_folder_BL = os.path.join(args_dict['bin_folder'], ratio, 'BL') if write_stream else None
    bin_folder_EL = os.path.join(args_dict['bin_folder'], ratio, 'EL') if write_stream else None
    if bin_folder_BL:
        os.makedirs(bin_folder_BL, exist_ok=True)
        os.makedirs(bin_folder_EL, exist_ok=True)
    decode_frame_folder_BL = os.path.join(args_dict['decoded_frame_folder'], ratio, 'BL') if save_decoded_frame else None
    decode_frame_folder_EL = os.path.join(args_dict['decoded_frame_folder'], ratio, 'EL') if save_decoded_frame else None
    if decode_frame_folder_BL:
        os.makedirs(decode_frame_folder_BL, exist_ok=True)
        os.makedirs(decode_frame_folder_EL, exist_ok=True)

    # size match
    padding_result = get_interlayer_padding(H_HR=height_EL, W_HR=width_EL, ratio=scale_factor)
    p_size_EL = padding_result['P_HR']
    p_size_BL = padding_result['P_LR']
    height_BL_padded, width_BL_padded = padding_result['LR_padded_size']
    height_EL_padded, width_EL_padded = padding_result['HR_padded_size']
    height_BL, width_BL = padding_result['LR_size']
    height_EL, width_EL = padding_result['HR_size']
    frame_pixel_num_BL = height_BL * width_BL
    frame_pixel_num_EL = height_EL * width_EL
    yuv_reader_EL = YUVReader(yuv_path_EL, width_EL, height_EL)
    with torch.no_grad():
        for frame_idx in range(frame_num):
            # read one YUV frame
            y_EL, uv_EL = yuv_reader_EL.read_one_frame()
            rgb_EL = np_image_to_tensor(ycbcr420_to_rgb(y_EL, uv_EL))
            y_EL = y_EL[0, :, :]
            u_EL = uv_EL[0, :, :]
            v_EL = uv_EL[1, :, :]
            # forward network
            rgb_EL = rgb_EL.to(device)
            x_EL_padded = F.pad(
                rgb_EL,
                p_size_EL,
                mode="constant",
                value=0,
            )
            # resize
            x_BL_padded = imresize(x_EL_padded, sizes=(height_BL_padded, width_BL_padded), kernel='cubic').clamp_(0, 1)
            rgb_BL = F.pad(x_BL_padded, inverse_padding_size(p_size_BL))
            y_BL, uv_BL = rgb_to_ycbcr420(rgb_BL.squeeze(0).cpu().numpy())
            y_BL = y_BL[0, :, :]
            u_BL = uv_BL[0, :, :]
            v_BL = uv_BL[1, :, :]
            pic_height_EL_padded = x_EL_padded.shape[2]
            pic_width_EL_padded = x_EL_padded.shape[3]
            pic_height_BL_padded = x_BL_padded.shape[2]
            pic_width_BL_padded = x_BL_padded.shape[3]
            assert pic_height_BL_padded == height_BL_padded and pic_width_BL_padded == width_BL_padded \
                   and pic_height_EL_padded == height_EL_padded and pic_width_EL_padded == width_EL_padded

            i_frame_net.set_scale_information(scale_factor, (height_EL_padded, width_EL_padded), (0, 0, 0, 0))
            video_net.set_scale_information(scale_factor, (height_EL_padded, width_EL_padded), (0, 0, 0, 0))
            bin_path_BL = os.path.join(args_dict['bin_folder'], ratio, 'BL', f"{frame_idx}.bin") \
                if write_stream else None
            bin_path_EL = os.path.join(args_dict['bin_folder'], ratio, 'EL', f"{frame_idx}.bin") \
                if write_stream else None

            if frame_idx % gop_size == 0:
                result = i_frame_net.encode_decode(x_BL_padded, x_EL_padded, bin_path_BL, bin_path_EL,
                                                   pic_height_bl=pic_height_BL_padded, pic_width_bl=pic_width_BL_padded,
                                                   pic_height_el=pic_height_EL_padded, pic_width_el=pic_width_EL_padded)
                DPB = {'ref_frame_bl': result['x_hat_bl'], 'ref_frame_el': result['x_hat_el'], 'ref_feature_bl': None, 'ref_feature_el': result['feature_el']}
                BL_bits.append(result['bit_bl'])
                EL_bits.append(result['bit_el'])
                frame_types.append(0)
                i_frame_number += 1

            else:
                result = video_net.encode_decode(x_BL_padded, x_EL_padded, DPB,
                                                 bin_path_BL, bin_path_EL,
                                                 pic_width=pic_width_EL_padded,
                                                 pic_height=pic_height_EL_padded,
                                                 pic_width_bl=pic_width_BL_padded,
                                                 pic_height_bl=pic_height_BL_padded,
                                                 )
                DPB = result['dpb']
                mv_EL = result['mv_hat']
                warp_frame_EL = result['warp_frame']
                frame_types.append(1)
                BL_bits.append(result['bit_bl'])
                EL_bits.append(result['bit_el'])
                p_frame_number += 1
                overall_encoding_time_BL += result['encoding_time_BL']
                overall_decoding_time_BL += result['decoding_time_BL']
                overall_encoding_time_EL += result['encoding_time_EL']
                overall_decoding_time_EL += result['decoding_time_EL']

            ref_frame_BL = DPB['ref_frame_bl'].clamp_(0, 1)
            ref_frame_EL = DPB['ref_frame_el'].clamp_(0, 1)
            x_hat_BL = F.pad(ref_frame_BL, inverse_padding_size(p_size_BL))
            x_hat_EL = F.pad(ref_frame_EL, inverse_padding_size(p_size_EL))
            BL_rgb_psnr.append(PSNR(rgb_BL, x_hat_BL))
            EL_rgb_psnr.append(PSNR(rgb_EL, x_hat_EL))
            win_size = 11
            if height_BL <= 160:
                win_size = 7
            BL_rgb_SSIM.append(ms_ssim(rgb_BL, x_hat_BL, win_size=win_size, data_range=1).item())
            EL_rgb_SSIM.append(ms_ssim(rgb_EL, x_hat_EL, win_size=win_size, data_range=1).item())
            x_hat_BL = x_hat_BL.squeeze(0).cpu().numpy()
            x_hat_EL = x_hat_EL.squeeze(0).cpu().numpy()
            y_rec_BL, uv_rec_BL = rgb_to_ycbcr420(x_hat_BL)
            y_rec_EL, uv_rec_EL = rgb_to_ycbcr420(x_hat_EL)
            if frame_idx % gop_size > 0:
                warp_frame_EL = warp_frame_EL.clamp_(0, 1)
                warp_frame_EL = F.pad(warp_frame_EL, inverse_padding_size(p_size_EL))
                warp_psnr = PSNR(warp_frame_EL, rgb_EL)
                print("warp psnr:", warp_psnr)
            # Y, U, V MSE
            y_rec_BL = y_rec_BL[0, :, :]
            u_rec_BL = uv_rec_BL[0, :, :]
            v_rec_BL = uv_rec_BL[1, :, :]
            y_rec_EL = y_rec_EL[0, :, :]
            u_rec_EL = uv_rec_EL[0, :, :]
            v_rec_EL = uv_rec_EL[1, :, :]
            y_mse_BL = np.mean(np.square(y_rec_BL - y_BL))
            u_mse_BL = np.mean(np.square(u_rec_BL - u_BL))
            v_mse_BL = np.mean(np.square(v_rec_BL - v_BL))
            y_mse_EL = np.mean(np.square(y_rec_EL - y_EL))
            u_mse_EL = np.mean(np.square(u_rec_EL - u_EL))
            v_mse_EL = np.mean(np.square(v_rec_EL - v_EL))

            # Y, U, V PSNR
            y_psnr_BL = mse2PSNR(y_mse_BL, data_range=1)
            u_psnr_BL = mse2PSNR(u_mse_BL, data_range=1)
            v_psnr_BL = mse2PSNR(v_mse_BL, data_range=1)
            y_psnr_EL = mse2PSNR(y_mse_EL, data_range=1)
            u_psnr_EL = mse2PSNR(u_mse_EL, data_range=1)
            v_psnr_EL = mse2PSNR(v_mse_EL, data_range=1)
            yuv_psnr_BL = (6 * y_psnr_BL + u_psnr_BL + v_psnr_BL) / 8
            yuv_psnr_EL = (6 * y_psnr_EL + u_psnr_EL + v_psnr_EL) / 8
            BL_YUV_psnr.append(yuv_psnr_BL)
            BL_Y_psnr.append(y_psnr_BL)
            BL_U_psnr.append(u_psnr_BL)
            BL_V_psnr.append(v_psnr_BL)
            EL_YUV_psnr.append(yuv_psnr_EL)
            EL_Y_psnr.append(y_psnr_EL)
            EL_U_psnr.append(u_psnr_EL)
            EL_V_psnr.append(v_psnr_EL)

            # Y, U, V MS-SSIM
            BL_Y_ssim = calc_msssim(y_BL, y_rec_BL, data_range=1)
            BL_U_ssim = calc_msssim(u_BL, u_rec_BL, data_range=1)
            BL_V_ssim = calc_msssim(v_BL, v_rec_BL, data_range=1)
            EL_Y_ssim = calc_msssim(y_EL, y_rec_EL, data_range=1)
            EL_U_ssim = calc_msssim(u_EL, u_rec_EL, data_range=1)
            EL_V_ssim = calc_msssim(v_EL, v_rec_EL, data_range=1)
            BL_ssim = (6 * BL_Y_ssim + BL_U_ssim + BL_V_ssim) / 8
            EL_ssim = (6 * EL_Y_ssim + EL_U_ssim + EL_V_ssim) / 8
            BL_SSIM.append(BL_ssim)
            EL_SSIM.append(EL_ssim)

            if save_decoded_frame:
                save_path_BL = os.path.join(args_dict['decoded_frame_folder'], ratio, 'BL', f'{frame_idx}.png')
                save_path_EL = os.path.join(args_dict['decoded_frame_folder'], ratio, 'EL', f'{frame_idx}.png')
                save_torch_image(x_hat_BL, save_path_BL)
                save_torch_image(x_hat_EL, save_path_EL)
            if save_decoded_mv:
                if frame_idx % gop_size > 0:
                    save_path = os.path.join(args_dict['decoded_mv_folder'], ratio, f'{frame_idx}.png')
                    save_torch_mv(mv_EL, save_path)
            if save_warp_frame:
                if frame_idx % gop_size > 0:
                    save_path = os.path.join(args_dict['warp_frame_folder'], ratio, f'{frame_idx}.png')
                    save_torch_image(warp_frame_EL, save_path)

    test_time = time.time() - start_time

    cur_ave_i_frame_bit_BL = 0
    cur_ave_i_frame_psnr_BL = 0
    cur_ave_i_frame_rgb_psnr_BL = 0
    cur_ave_i_frame_Y_psnr_BL = 0
    cur_ave_i_frame_U_psnr_BL = 0
    cur_ave_i_frame_V_psnr_BL = 0
    cur_ave_i_frame_msssim_BL = 0
    cur_ave_i_frame_rgb_msssim_BL = 0

    cur_ave_i_frame_bit_EL = 0
    cur_ave_i_frame_psnr_EL = 0
    cur_ave_i_frame_rgb_psnr_EL = 0
    cur_ave_i_frame_Y_psnr_EL = 0
    cur_ave_i_frame_U_psnr_EL = 0
    cur_ave_i_frame_V_psnr_EL = 0
    cur_ave_i_frame_msssim_EL = 0
    cur_ave_i_frame_rgb_msssim_EL = 0

    cur_ave_p_frame_bit_BL = 0
    cur_ave_p_frame_psnr_BL = 0
    cur_ave_p_frame_rgb_psnr_BL = 0
    cur_ave_p_frame_Y_psnr_BL = 0
    cur_ave_p_frame_U_psnr_BL = 0
    cur_ave_p_frame_V_psnr_BL = 0
    cur_ave_p_frame_msssim_BL = 0
    cur_ave_p_frame_rgb_msssim_BL = 0

    cur_ave_p_frame_bit_EL = 0
    cur_ave_p_frame_psnr_EL = 0
    cur_ave_p_frame_rgb_psnr_EL = 0
    cur_ave_p_frame_Y_psnr_EL = 0
    cur_ave_p_frame_U_psnr_EL = 0
    cur_ave_p_frame_V_psnr_EL = 0
    cur_ave_p_frame_msssim_EL = 0
    cur_ave_p_frame_rgb_msssim_EL = 0

    cur_i_frame_num = 0
    cur_p_frame_num = 0
    for idx in range(frame_num):
        if frame_types[idx] == 0:
            cur_ave_i_frame_bit_BL += BL_bits[idx]
            cur_ave_i_frame_psnr_BL += BL_YUV_psnr[idx]  # accumulated in MSE
            cur_ave_i_frame_rgb_psnr_BL += BL_rgb_psnr[idx]
            cur_ave_i_frame_Y_psnr_BL += BL_Y_psnr[idx]
            cur_ave_i_frame_U_psnr_BL += BL_U_psnr[idx]
            cur_ave_i_frame_V_psnr_BL += BL_V_psnr[idx]
            cur_ave_i_frame_msssim_BL += BL_SSIM[idx]
            cur_ave_i_frame_rgb_msssim_BL += BL_rgb_SSIM[idx]

            cur_ave_i_frame_bit_EL += EL_bits[idx]
            cur_ave_i_frame_psnr_EL += EL_YUV_psnr[idx]
            cur_ave_i_frame_rgb_psnr_EL += EL_rgb_psnr[idx]
            cur_ave_i_frame_Y_psnr_EL += EL_Y_psnr[idx]
            cur_ave_i_frame_U_psnr_EL += EL_U_psnr[idx]
            cur_ave_i_frame_V_psnr_EL += EL_V_psnr[idx]
            cur_ave_i_frame_msssim_EL += EL_SSIM[idx]
            cur_ave_i_frame_rgb_msssim_EL += EL_rgb_SSIM[idx]

            cur_i_frame_num += 1
        else:
            cur_ave_p_frame_bit_BL += BL_bits[idx]
            cur_ave_p_frame_psnr_BL += BL_YUV_psnr[idx]
            cur_ave_p_frame_rgb_psnr_BL += BL_rgb_psnr[idx]
            cur_ave_p_frame_Y_psnr_BL += BL_Y_psnr[idx]
            cur_ave_p_frame_U_psnr_BL += BL_U_psnr[idx]
            cur_ave_p_frame_V_psnr_BL += BL_V_psnr[idx]
            cur_ave_p_frame_msssim_BL += BL_SSIM[idx]
            cur_ave_p_frame_rgb_msssim_BL += BL_rgb_SSIM[idx]

            cur_ave_p_frame_bit_EL += EL_bits[idx]
            cur_ave_p_frame_psnr_EL += EL_YUV_psnr[idx]
            cur_ave_p_frame_rgb_psnr_EL += EL_rgb_psnr[idx]
            cur_ave_p_frame_Y_psnr_EL += EL_Y_psnr[idx]
            cur_ave_p_frame_U_psnr_EL += EL_U_psnr[idx]
            cur_ave_p_frame_V_psnr_EL += EL_V_psnr[idx]
            cur_ave_p_frame_msssim_EL += EL_SSIM[idx]
            cur_ave_p_frame_rgb_msssim_EL += EL_rgb_SSIM[idx]
            cur_p_frame_num += 1
    ################################################################################################
    # BL log
    log_result_BL = {}
    log_result_BL['frame_pixel_num'] = frame_pixel_num_BL
    log_result_BL['i_frame_num'] = cur_i_frame_num
    log_result_BL['p_frame_num'] = cur_p_frame_num
    log_result_BL['ave_i_frame_bpp'] = cur_ave_i_frame_bit_BL / cur_i_frame_num / frame_pixel_num_BL
    log_result_BL['ave_i_frame_psnr'] = cur_ave_i_frame_psnr_BL / cur_i_frame_num  # MSE TO PSNR
    log_result_BL['ave_i_frame_rgb_psnr'] = cur_ave_i_frame_rgb_psnr_BL / cur_i_frame_num
    log_result_BL['ave_i_frame_YUV_psnr'] = [cur_ave_i_frame_Y_psnr_BL / cur_i_frame_num, cur_ave_i_frame_U_psnr_BL / cur_i_frame_num,
                                             cur_ave_i_frame_V_psnr_BL / cur_i_frame_num]
    log_result_BL['ave_i_frame_msssim'] = cur_ave_i_frame_msssim_BL / cur_i_frame_num
    log_result_BL['ave_i_frame_rgb_msssim'] = cur_ave_i_frame_rgb_msssim_BL / cur_i_frame_num
    log_result_BL['frame_bpp'] = list(np.array(BL_bits) / frame_pixel_num_BL)
    log_result_BL['frame_type'] = frame_types
    log_result_BL['test_time'] = test_time
    log_result_BL['encoding_time'] = overall_encoding_time_BL / cur_p_frame_num
    log_result_BL['decoding_time'] = overall_decoding_time_BL / cur_p_frame_num
    if cur_p_frame_num > 0:
        total_p_pixel_num = cur_p_frame_num * frame_pixel_num_BL
        log_result_BL['ave_p_frame_bpp'] = cur_ave_p_frame_bit_BL / total_p_pixel_num
        log_result_BL['ave_p_frame_psnr'] = cur_ave_p_frame_psnr_BL / cur_p_frame_num
        log_result_BL['ave_p_frame_rgb_psnr'] = cur_ave_p_frame_rgb_psnr_BL / cur_p_frame_num
        log_result_BL['ave_p_frame_YUV_psnr'] = [cur_ave_p_frame_Y_psnr_BL / cur_p_frame_num, cur_ave_p_frame_U_psnr_BL / cur_p_frame_num,
                                                 cur_ave_p_frame_V_psnr_BL / cur_p_frame_num]
        log_result_BL['ave_p_frame_msssim'] = cur_ave_p_frame_msssim_BL / cur_p_frame_num
        log_result_BL['ave_p_frame_rgb_msssim'] = cur_ave_p_frame_rgb_msssim_BL / cur_p_frame_num
    else:
        log_result_BL['ave_p_frame_bpp'] = 0
        log_result_BL['ave_p_frame_psnr'] = 0
        log_result_BL['ave_p_frame_rgb_psnr'] = 0
        log_result_BL['ave_p_frame_YUV_psnr'] = [0, 0, 0]
        log_result_BL['ave_p_frame_msssim'] = 0
        log_result_BL['ave_p_frame_rgb_msssim'] = 0
    log_result_BL['ave_all_frame_bpp'] = (cur_ave_i_frame_bit_BL + cur_ave_p_frame_bit_BL) / \
                                         (frame_num * frame_pixel_num_BL)
    log_result_BL['ave_all_frame_psnr'] = (cur_ave_i_frame_psnr_BL + cur_ave_p_frame_psnr_BL) / frame_num
    log_result_BL['ave_all_frame_rgb_psnr'] = (cur_ave_i_frame_rgb_psnr_BL + cur_ave_p_frame_rgb_psnr_BL) / frame_num
    log_result_BL['ave_all_frame_YUV_psnr'] = [
        (cur_ave_i_frame_Y_psnr_BL + cur_ave_p_frame_Y_psnr_BL) / frame_num,
        (cur_ave_i_frame_U_psnr_BL + cur_ave_p_frame_U_psnr_BL) / frame_num,
        (cur_ave_i_frame_V_psnr_BL + cur_ave_p_frame_V_psnr_BL) / frame_num
    ]
    log_result_BL['ave_all_frame_msssim'] = (cur_ave_i_frame_msssim_BL + cur_ave_p_frame_msssim_BL) / \
                                            frame_num
    log_result_BL['ave_all_frame_rgb_msssim'] = (cur_ave_i_frame_rgb_msssim_BL + cur_ave_p_frame_rgb_msssim_BL) / \
                                                frame_num
    ################################################################################################
    # EL log
    log_result_EL = {}
    log_result_EL['frame_pixel_num'] = frame_pixel_num_EL
    log_result_EL['i_frame_num'] = cur_i_frame_num
    log_result_EL['p_frame_num'] = cur_p_frame_num
    log_result_EL['ave_i_frame_bpp'] = cur_ave_i_frame_bit_EL / cur_i_frame_num / frame_pixel_num_EL
    log_result_EL['ave_i_frame_psnr'] = cur_ave_i_frame_psnr_EL / cur_i_frame_num  # MSE TO PSNR
    log_result_EL['ave_i_frame_rgb_psnr'] = cur_ave_i_frame_rgb_psnr_EL / cur_i_frame_num
    log_result_EL['ave_i_frame_YUV_psnr'] = [cur_ave_i_frame_Y_psnr_EL / cur_i_frame_num, cur_ave_i_frame_U_psnr_EL / cur_i_frame_num,
                                             cur_ave_i_frame_V_psnr_EL / cur_i_frame_num]
    log_result_EL['ave_i_frame_msssim'] = cur_ave_i_frame_msssim_EL / cur_i_frame_num
    log_result_EL['ave_i_frame_rgb_msssim'] = cur_ave_i_frame_rgb_msssim_EL / cur_i_frame_num
    log_result_EL['frame_bpp'] = list(np.array(EL_bits) / frame_pixel_num_EL)
    log_result_EL['frame_type'] = frame_types
    log_result_EL['test_time'] = test_time
    log_result_EL['encoding_time'] = overall_encoding_time_EL / cur_p_frame_num
    log_result_EL['decoding_time'] = overall_decoding_time_EL / cur_p_frame_num
    if cur_p_frame_num > 0:
        total_p_pixel_num = cur_p_frame_num * frame_pixel_num_EL
        log_result_EL['ave_p_frame_bpp'] = cur_ave_p_frame_bit_EL / total_p_pixel_num
        log_result_EL['ave_p_frame_psnr'] = cur_ave_p_frame_psnr_EL / cur_p_frame_num  # MSE TO PSNR
        log_result_EL['ave_p_frame_rgb_psnr'] = cur_ave_p_frame_rgb_psnr_EL / cur_p_frame_num
        log_result_EL['ave_p_frame_YUV_psnr'] = [cur_ave_p_frame_Y_psnr_EL / cur_p_frame_num, cur_ave_p_frame_U_psnr_EL / cur_p_frame_num,
                                                 cur_ave_p_frame_V_psnr_EL / cur_p_frame_num]
        log_result_EL['ave_p_frame_msssim'] = cur_ave_p_frame_msssim_EL / cur_p_frame_num
        log_result_EL['ave_p_frame_rgb_msssim'] = cur_ave_p_frame_rgb_msssim_EL / cur_p_frame_num
    else:
        log_result_EL['ave_p_frame_bpp'] = 0
        log_result_EL['ave_p_frame_psnr'] = 0
        log_result_EL['ave_p_frame_rgb_psnr'] = 0
        log_result_EL['ave_p_frame_YUV_psnr'] = [0, 0, 0]
        log_result_EL['ave_p_frame_msssim'] = 0
        log_result_EL['ave_p_frame_rgb_msssim'] = 0
    log_result_EL['ave_all_frame_bpp'] = (cur_ave_i_frame_bit_EL + cur_ave_p_frame_bit_EL) / \
                                         (frame_num * frame_pixel_num_EL)
    log_result_EL['ave_all_frame_psnr'] = (cur_ave_i_frame_psnr_EL + cur_ave_p_frame_psnr_EL) / frame_num
    log_result_EL['ave_all_frame_rgb_psnr'] = (cur_ave_i_frame_rgb_psnr_EL + cur_ave_p_frame_rgb_psnr_EL) / frame_num
    log_result_EL['ave_all_frame_YUV_psnr'] = [
        (cur_ave_i_frame_Y_psnr_EL + cur_ave_p_frame_Y_psnr_EL) / frame_num,
        (cur_ave_i_frame_U_psnr_EL + cur_ave_p_frame_U_psnr_EL) / frame_num,
        (cur_ave_i_frame_V_psnr_EL + cur_ave_p_frame_V_psnr_EL) / frame_num
    ]
    log_result_EL['ave_all_frame_msssim'] = (cur_ave_i_frame_msssim_EL + cur_ave_p_frame_msssim_EL) / \
                                            frame_num
    log_result_EL['ave_all_frame_rgb_msssim'] = (cur_ave_i_frame_rgb_msssim_EL + cur_ave_p_frame_rgb_msssim_EL) / \
                                                frame_num

    ################################################################################################
    # FULL log
    log_result_FL = {}
    log_result_FL['frame_pixel_num'] = frame_pixel_num_EL
    log_result_FL['i_frame_num'] = cur_i_frame_num
    log_result_FL['p_frame_num'] = cur_p_frame_num
    log_result_FL['ave_i_frame_bpp'] = (cur_ave_i_frame_bit_BL + cur_ave_i_frame_bit_EL) / cur_i_frame_num / frame_pixel_num_EL
    log_result_FL['ave_i_frame_psnr'] = cur_ave_i_frame_psnr_EL / cur_i_frame_num  # MSE TO PSNR
    log_result_FL['ave_i_frame_rgb_psnr'] = cur_ave_i_frame_rgb_psnr_EL / cur_i_frame_num
    log_result_FL['ave_i_frame_msssim'] = cur_ave_i_frame_msssim_EL / cur_i_frame_num
    log_result_FL['ave_i_frame_rgb_msssim'] = cur_ave_i_frame_rgb_msssim_EL / cur_i_frame_num
    log_result_FL['frame_type'] = frame_types
    log_result_FL['test_time'] = test_time
    log_result_FL['encoding_time'] = (overall_encoding_time_BL + overall_encoding_time_EL) / cur_p_frame_num
    log_result_FL['decoding_time'] = (overall_decoding_time_BL + overall_decoding_time_EL) / cur_p_frame_num
    if cur_p_frame_num > 0:
        total_p_pixel_num = cur_p_frame_num * frame_pixel_num_EL
        log_result_FL['ave_p_frame_bpp'] = (cur_ave_p_frame_bit_EL + cur_ave_p_frame_bit_BL) / total_p_pixel_num
        log_result_FL['ave_p_frame_psnr'] = cur_ave_p_frame_psnr_EL / cur_p_frame_num  # MSE TO PSNR
        log_result_FL['ave_p_frame_rgb_psnr'] = cur_ave_p_frame_rgb_psnr_EL / cur_p_frame_num
        log_result_FL['ave_p_frame_msssim'] = cur_ave_p_frame_msssim_EL / cur_p_frame_num
        log_result_FL['ave_p_frame_rgb_msssim'] = cur_ave_p_frame_rgb_msssim_EL / cur_p_frame_num
    else:
        log_result_FL['ave_p_frame_bpp'] = 0
        log_result_FL['ave_p_frame_psnr'] = 0
        log_result_FL['ave_p_frame_rgb_psnr'] = 0
        log_result_FL['ave_p_frame_msssim'] = 0
        log_result_FL['ave_p_frame_rgb_msssim'] = 0
    log_result_FL['ave_all_frame_bpp'] = (cur_ave_i_frame_bit_EL + cur_ave_p_frame_bit_EL + cur_ave_i_frame_bit_BL + cur_ave_p_frame_bit_BL) / \
                                         (frame_num * frame_pixel_num_EL)
    log_result_FL['ave_all_frame_psnr'] = (cur_ave_i_frame_psnr_EL + cur_ave_p_frame_psnr_EL) / frame_num
    log_result_FL['ave_all_frame_rgb_psnr'] = (cur_ave_i_frame_rgb_psnr_EL + cur_ave_p_frame_rgb_psnr_EL) / frame_num
    log_result_FL['ave_all_frame_msssim'] = (cur_ave_i_frame_msssim_EL + cur_ave_p_frame_msssim_EL) / frame_num
    log_result_FL['ave_all_frame_rgb_msssim'] = (cur_ave_i_frame_rgb_msssim_EL + cur_ave_p_frame_rgb_msssim_EL) / frame_num

    return log_result_BL, log_result_EL, log_result_FL


def encode_one(args_dict, device):
    i_frame_load_checkpoint = torch.load(args_dict['i_frame_model_path'],
                                         map_location=torch.device('cpu'))
    if "state_dict" in i_frame_load_checkpoint:
        i_frame_load_checkpoint = i_frame_load_checkpoint['state_dict']
    i_frame_net = IntraSS.from_state_dict(
        i_frame_load_checkpoint).eval()
    i_frame_net = i_frame_net.to(device)
    i_frame_net.eval()

    if args_dict['force_intra']:
        video_net = None
    else:
        video_net = LSSVC_extend()
        load_checkpoint = torch.load(args_dict['video_model_path'], map_location=torch.device('cpu'))
        if "state_dict" in load_checkpoint:
            load_checkpoint = load_checkpoint['state_dict']
        video_net.load_dict(load_checkpoint)
        video_net = video_net.to(device)
        video_net.eval()

    if args_dict['write_stream']:
        if video_net is not None:
            video_net.update(force=True)
        i_frame_net.update(force=True)

    sub_dir_name = args_dict['video_path']

    gop_size = args_dict['gop']
    frame_num = args_dict['frame_num']
    ratio = args_dict['ratio']

    # to do, need to add something related to the InterModules or model index to the bin_folder
    bin_folder = os.path.join(args_dict['stream_path'], sub_dir_name, str(args_dict['model_idx']))
    if args_dict['write_stream'] and not os.path.exists(bin_folder):
        os.makedirs(bin_folder)

    if args_dict['save_decoded_frame']:
        decoded_frame_folder = os.path.join(args_dict['decoded_frame_path'], sub_dir_name,
                                            str(args_dict['model_idx']))
        os.makedirs(decoded_frame_folder, exist_ok=True)
    else:
        decoded_frame_folder = None

    if args_dict['save_decoded_mv']:
        decoded_mv_folder = os.path.join(args_dict['decoded_mv_path'], sub_dir_name,
                                         str(args_dict['model_idx']))
        os.makedirs(decoded_mv_folder, exist_ok=True)
    else:
        decoded_mv_folder = None

    if args_dict['save_warp_frame']:
        warp_frame_folder = os.path.join(args_dict['warp_frame_path'], sub_dir_name,
                                         str(args_dict['model_idx']))
        os.makedirs(warp_frame_folder, exist_ok=True)
    else:
        warp_frame_folder = None

    if args_dict['save_decoded_context']:
        decoded_context_folder = os.path.join(args_dict['decoded_context_path'], sub_dir_name,
                                              str(args_dict['model_idx']))
        os.makedirs(decoded_context_folder, exist_ok=True)
    else:
        decoded_context_folder = None

    args_dict['yuv_path_el'] = os.path.join(args_dict['dataset_path'], sub_dir_name, 'x1.yuv')
    args_dict['yuv_path_bl'] = os.path.join(args_dict['dataset_path'], sub_dir_name, ratio + '.yuv')
    args_dict['gop_size'] = gop_size
    args_dict['frame_num'] = frame_num
    args_dict['bin_folder'] = bin_folder
    args_dict['decoded_frame_folder'] = decoded_frame_folder
    args_dict['decoded_mv_folder'] = decoded_mv_folder
    args_dict['warp_frame_folder'] = warp_frame_folder
    args_dict['decoded_context_folder'] = decoded_context_folder

    result_BL, result_EL, result_fl = run_test(video_net, i_frame_net, args_dict, device=device)

    result_BL['name'] = f"{os.path.basename(args_dict['video_model_path'])}_{sub_dir_name}"
    result_BL['ds_name'] = args_dict['ds_name']
    result_BL['video_path'] = args_dict['video_path']
    result_BL['ratio'] = args_dict['ratio']

    result_EL['name'] = f"{os.path.basename(args_dict['video_model_path'])}_{sub_dir_name}"
    result_EL['ds_name'] = args_dict['ds_name']
    result_EL['video_path'] = args_dict['video_path']
    result_BL['ratio'] = args_dict['ratio']

    result_fl['name'] = f"{os.path.basename(args_dict['video_model_path'])}_{sub_dir_name}"
    result_fl['ds_name'] = args_dict['ds_name']
    result_fl['video_path'] = args_dict['video_path']
    result_BL['ratio'] = args_dict['ratio']

    return result_BL, result_EL, result_fl


def worker(use_cuda, args):
    torch.backends.cudnn.benchmark = False
    if 'use_deterministic_algorithms' in dir(torch):
        torch.use_deterministic_algorithms(True)
    else:
        torch.set_deterministic(True)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    np.random.seed(seed=0)
    gpu_num = 0
    if use_cuda:
        gpu_num = torch.cuda.device_count()

    process_name = multiprocessing.current_process().name
    process_idx = int(process_name[process_name.rfind('-') + 1:])
    gpu_id = -1
    if gpu_num > 0:
        gpu_id = process_idx % gpu_num
    if gpu_id >= 0:
        device = f"cuda:{gpu_id}"
    else:
        device = "cpu"

    result_BL, result_EL, result_fl = encode_one(args, device)
    result_BL['model_idx'] = args['model_idx']
    result_EL['model_idx'] = args['model_idx']
    result_fl['model_idx'] = args['model_idx']
    return result_BL, result_EL, result_fl


def main():
    """
    分配任务obj
    每个obj指定要测试的序列、使用的模型等信息
    """
    begin_time = time.time()

    torch.backends.cudnn.enabled = True
    args = parse_args()

    if args.cuda_device is not None and args.cuda_device != '':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

    worker_num = args.worker
    assert worker_num >= 1

    with open(args.test_config) as f:
        config = json.load(f)

    multiprocessing.set_start_method("spawn")
    threadpool_executor = concurrent.futures.ProcessPoolExecutor(max_workers=worker_num)
    objs = []

    if args.force_intra:
        args.model_path = args.i_frame_model_path
    count_frames = 0
    count_sequences = 0
    ratio_list = ["x2"]
    for ds_name in config:
        if config[ds_name]['test'] == 0:
            continue
        for ratio in ratio_list:
            for seq_name_EL in config[ds_name]['sequences']:
                count_sequences += 1
                for model_idx in range(len(args.model_path)):  # pylint: disable=C0200
                    cur_dict = {}
                    cur_dict['ratio'] = ratio
                    cur_dict['x1'] = config[ds_name]['x1']
                    cur_dict[ratio] = config[ds_name][ratio]
                    cur_dict['model_idx'] = model_idx
                    cur_dict['i_frame_model_path'] = args.i_frame_model_path[model_idx]
                    cur_dict['i_frame_model_name'] = args.i_frame_model_name
                    cur_dict['video_model_path'] = args.model_path[model_idx]
                    cur_dict['video_model_name'] = args.model_name
                    cur_dict['force_intra'] = args.force_intra
                    cur_dict['video_path'] = seq_name_EL
                    cur_dict['gop'] = config[ds_name]['sequences'][seq_name_EL]['gop']
                    if args.force_intra:
                        cur_dict['gop'] = 1
                    if args.force_intra_period > 0:
                        cur_dict['gop'] = args.force_intra_period
                    cur_dict['frame_num'] = config[ds_name]['sequences'][seq_name_EL]['frames']
                    if args.force_frame_num > 0:
                        cur_dict['frame_num'] = args.force_frame_num
                    cur_dict['dataset_path'] = config[ds_name]['base_path']
                    cur_dict['write_stream'] = args.write_stream
                    cur_dict['stream_path'] = args.stream_path
                    cur_dict['save_decoded_frame'] = args.save_decoded_frame
                    cur_dict['save_decoded_mv'] = args.save_decoded_mv
                    cur_dict['save_warp_frame'] = args.save_warp_frame
                    cur_dict['save_decoded_context'] = args.save_decoded_context
                    cur_dict['decoded_frame_path'] = f'{args.decoded_frame_path}_' \
                                                     f'{args.i_frame_model_name}_LSSVC'
                    cur_dict['decoded_mv_path'] = f'{args.decoded_mv_path}_' \
                                                  f'{args.i_frame_model_name}_LSSVC'
                    cur_dict['warp_frame_path'] = f'{args.warp_frame_path}_' \
                                                  f'{args.i_frame_model_name}_LSSVC'
                    cur_dict['decoded_context_path'] = f'{args.decoded_context_path}_' \
                                                       f'{args.i_frame_model_name}_LSSVC'
                    cur_dict['ds_name'] = ds_name

                    count_frames += cur_dict['frame_num']

                    obj = threadpool_executor.submit(
                        worker,
                        args.cuda,
                        cur_dict)
                    objs.append(obj)

    results = []
    for obj in tqdm(objs):
        result = obj.result()
        results.append(result)

    # write to JSON
    os.makedirs(args.output_path, exist_ok=True)
    for ratio in ratio_list:
        log_result_BL = {}
        log_result_EL = {}
        log_result_fl = {}
        for ds_name in config:
            if config[ds_name]['test'] == 0:
                continue
            log_result_BL[ds_name] = {}
            log_result_EL[ds_name] = {}
            log_result_fl[ds_name] = {}
            for seq in config[ds_name]['sequences']:
                log_result_BL[ds_name][seq] = {}
                log_result_EL[ds_name][seq] = {}
                log_result_fl[ds_name][seq] = {}
                for model in args.model_path:
                    ckpt = os.path.basename(model)
                    for res in results:
                        res_BL, res_EL, res_fl = res
                        if res_BL['name'].startswith(ckpt) and ds_name == res_BL['ds_name'] \
                                and seq == res_BL['video_path'] and res_BL['ratio'] == ratio:
                            log_result_BL[ds_name][seq][ckpt] = filter_dict(res_BL)
                            log_result_EL[ds_name][seq][ckpt] = filter_dict(res_EL)
                            log_result_fl[ds_name][seq][ckpt] = filter_dict(res_fl)
        out_json_dir_BL = args.output_path + f'/{ratio}_BL.json'
        out_json_dir_EL = args.output_path + f'/{ratio}_EL.json'
        out_json_dir_fl = args.output_path + f'/{ratio}_FL.json'
        with open(out_json_dir_BL, 'w') as fp:
            json.dump(log_result_BL, fp, indent=2)
        with open(out_json_dir_EL, 'w') as fp:
            json.dump(log_result_EL, fp, indent=2)
        with open(out_json_dir_fl, 'w') as fp:
            json.dump(log_result_fl, fp, indent=2)

    total_minutes = (time.time() - begin_time) / 60

    count_models = len(args.model_path)
    count_frames = count_frames // count_models
    print('Test finished')
    print(f'Tested {count_models} models on {count_frames} frames from {count_sequences} sequences')
    print(f'Total elapsed time: {total_minutes:.1f} min')


if __name__ == "__main__":
    main()
