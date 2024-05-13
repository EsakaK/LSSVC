import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_padding_size(height, width, p=64):
    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    padding_left = 0
    padding_right = int(new_w - width - padding_left)
    padding_top = 0
    padding_bottom = int(new_h - height - padding_top)
    return padding_left, padding_right, padding_top, padding_bottom


def filter_dict(result):
    # keys = ['i_frame_num', 'p_frame_num',
    #         'ave_i_frame_bpp', 'ave_i_frame_psnr', 'ave_i_frame_msssim',
    #         'ave_p_frame_bpp', 'ave_p_frame_psnr', 'ave_p_frame_msssim',
    #         'frame_bpp', 'frame_psnr', 'frame_msssim', 'frame_type',
    #         'test_time','encoding_time','decoding_time']
    keys = ['i_frame_num', 'p_frame_num',
            'ave_i_frame_bpp', 'ave_i_frame_psnr', 'ave_i_frame_rgb_psnr', 'ave_i_frame_msssim', 'ave_i_frame_rgb_msssim', 'ave_i_frame_YUV_psnr',
            'ave_p_frame_bpp', 'ave_p_frame_psnr', 'ave_p_frame_rgb_psnr', 'ave_p_frame_msssim', 'ave_p_frame_rgb_msssim', 'ave_p_frame_YUV_psnr',
            'ave_all_frame_bpp', 'ave_all_frame_psnr', 'ave_all_frame_rgb_psnr', 'ave_all_frame_msssim', 'ave_all_frame_rgb_msssim', 'ave_all_frame_YUV_psnr',
            'encoding_time', 'decoding_time']
    res = {k: v for k, v in result.items() if k in keys}
    return res


def round_to_even(x):
    tmp = int(x)
    if tmp % 2 != 0:
        return tmp + 1
    else:
        return tmp


def get_interlayer_padding(H_HR, W_HR, ratio):
    i = 0
    while True:
        p = 64 + 32 * i
        tmp_H = (H_HR + p - 1) // p * p
        if tmp_H % 64 == 0 and tmp_H % (64 * ratio) == 0:
            new_H_HR = tmp_H
            break
        i += 1
    i = 0
    while True:
        p = 64 + 32 * i
        tmp_W = (W_HR + p - 1) // p * p
        if tmp_W % 64 == 0 and tmp_W % (64 * ratio) == 0:
            new_W_HR = tmp_W
            break
        i += 1
    padding_left_EL = 0
    padding_right_EL = new_W_HR - W_HR - padding_left_EL
    padding_top_EL = 0
    padding_bottom_EL = new_H_HR - H_HR - padding_top_EL

    H_LR = round_to_even(H_HR / ratio)
    W_LR = round_to_even(W_HR / ratio)

    new_H_LR = int(new_H_HR / ratio)
    new_W_LR = int(new_W_HR / ratio)

    padding_left_BL = 0
    padding_right_BL = new_W_LR - W_LR
    padding_top_BL = 0
    padding_bottom_BL = new_H_LR - H_LR

    padding_LR = (padding_left_BL, padding_right_BL, padding_top_BL, padding_bottom_BL)
    padding_HR = (padding_left_EL, padding_right_EL, padding_top_EL, padding_bottom_EL)

    return {'P_LR': padding_LR, 'P_HR': padding_HR, 'LR_padded_size': (new_H_LR, new_W_LR),
            'HR_padded_size': (new_H_HR, new_W_HR),
            'LR_size': (H_LR, W_LR), 'HR_size': (H_HR, W_HR)}


def inverse_padding_size(p_size: tuple):
    return (-p_size[0], -p_size[1], -p_size[2], -p_size[3])
