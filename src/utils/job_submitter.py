import json
import os
import time
import zipfile
from multiprocessing import Pool
from shutil import copyfile

import torch.cuda


def get_args(argv):
    print(argv)
    working_folder = argv[1]
    dataset_folder = argv[2]
    experiment_name = argv[3]
    print(f"working_folder {working_folder}")
    print(f"dataset_folder {dataset_folder}")
    print(f"experiment_name {experiment_name}")
    return working_folder, dataset_folder, experiment_name


def install_dependency():
    os.system('pwd')
    os.system('ls')
    os.system('python -m pip install -U pip')
    os.system('python -m pip install -r requirements.txt')
    os.system('nvidia-smi')


def unzip_dataset(src_folder, dst_folder):
    print(f"unzipping from {src_folder} to {dst_folder}")
    for f in os.listdir(src_folder):
        if not f.endswith('.zip'):
            continue
        src_path = os.path.join(src_folder, f)
        with zipfile.ZipFile(src_path, 'r') as zip_ref:
            zip_ref.extractall(dst_folder)
        print(f"{time.ctime()} extracted {f}")


def upload_one_dataset(src_root, src_description, dst_folder):
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    src_path = os.path.join(src_root, src_description)
    dst_path = os.path.join(dst_folder, "description.json")
    copyfile(src_path, dst_path)

    with open(src_path) as json_file:
        datasets = json.load(json_file)
    for dataset in datasets:
        dataset_name = dataset['dataset_name']
        src_folder = os.path.join(src_root, dataset_name)
        unzip_dataset(src_folder, dst_folder)


def upload_dataset():
    train_dataset_src_json = 'vimeo_train.json'  # change this name for train set
    test_dataset_src_json = 'vimeo_test.json'  # change this name for test set
    train_dataset_dst_folder = '/data/liyao/vimeo/video_train'
    test_dataset_dst_folder = '/data/liyao/vimeo/vimeo_test'

    return train_dataset_dst_folder, test_dataset_dst_folder


def load_config(experiment_name, cluster='bitahub'):
    if cluster == 'bitahub':
        image_models = [
            '/model/EsakaK/My_Model/LSSVCM/IntraSS_v2/q1.pth.tar',
            '/model/EsakaK/My_Model/LSSVCM/IntraSS_v2/q2.pth.tar',
            '/model/EsakaK/My_Model/LSSVCM/IntraSS_v2/q3.pth.tar',
            '/model/EsakaK/My_Model/LSSVCM/IntraSS_v2/q4.pth.tar',
        ]

        bl_models = [
            '/model/EsakaK/My_Model/LSSVCM/BL/BL_4/q1.pth',
            '/model/EsakaK/My_Model/LSSVCM/BL/BL_4/q2.pth',
            '/model/EsakaK/My_Model/LSSVCM/BL/BL_4/q3.pth',
            '/model/EsakaK/My_Model/LSSVCM/BL/BL_4/q4.pth',
        ]
        mv_enc_dec_pretrain_models = [
            '/model/EsakaK/My_Model/LSSVCM/mv_enc_dec_pretrain_el/checkpoint_LSSVC_q1_epo_4.pth',
            '/model/EsakaK/My_Model/LSSVCM/mv_enc_dec_pretrain_el/checkpoint_LSSVC_q2_epo_4.pth',
            '/model/EsakaK/My_Model/LSSVCM/mv_enc_dec_pretrain_el/checkpoint_LSSVC_q3_epo_4.pth',
            '/model/EsakaK/My_Model/LSSVCM/mv_enc_dec_pretrain_el/checkpoint_LSSVC_q4_epo_4.pth',
        ]
        mv_pretrain_dir = '/model/EsakaK/My_Model/DCVC/flow_pretrain_np/'
        train_dataset_dst_folder = '/data/liyao/vimeo/video_train'
        test_dataset_dst_folder = '/data/liyao/vimeo/vimeo_test'
        save_model_dir = f'/model/EsakaK/My_Model/TMP/{experiment_name}'
        save_dir = f'/model/EsakaK/My_Model/TMP/{experiment_name}_ckpt'
        visual_dir = '/output'
        base_root = '/code'
    else:
        image_models = [
            '/gdata2/bianyf/model/IntraSS_v2/q1.pth.tar',
            '/gdata2/bianyf/model/IntraSS_v2/q2.pth.tar',
            '/gdata2/bianyf/model/IntraSS_v2/q3.pth.tar',
            '/gdata2/bianyf/model/IntraSS_v2/q4.pth.tar',
        ]

        bl_models = [
            '/gdata2/bianyf/model/BL4/q1.pth',
            '/gdata2/bianyf/model/BL4/q2.pth',
            '/gdata2/bianyf/model/BL4/q3.pth',
            '/gdata2/bianyf/model/BL4/q4.pth',
        ]
        mv_enc_dec_pretrain_models = None
        mv_pretrain_dir = '/gdata2/bianyf/model/flow_net/'
        train_dataset_dst_folder = '/gdata/shengxh/vimeo/vimeo_train'
        test_dataset_dst_folder = '/gdata/shengxh/vimeo/vimeo_test'
        save_model_dir = f'/gdata2/bianyf/model/TMP/{experiment_name}'
        save_dir = f'/gdata2/bianyf/model/TMP/{experiment_name}_ckpt'
        visual_dir = f'/gdata2/bianyf/visual_board/{experiment_name}'
        base_root = '/ghome/bianyf/Git_repos/LSSVC'

    return {
        'train_dir': train_dataset_dst_folder,
        'test_dir': test_dataset_dst_folder,
        'me_pretrain_dir': mv_pretrain_dir,
        'save_model_dir': save_model_dir,
        'save_dir': save_dir,
        'visual_dir': visual_dir,
        'image_models': image_models,
        'bl_models': bl_models,
        'root': base_root,
        'mv_enc_dec_pretrain_models': mv_enc_dec_pretrain_models
    }


def load_ablation_command(EL_model_name):
    model_path = os.listdir(f'/model/EsakaK/My_Model/TMP/{EL_model_name[:-7]}')
    model_path.sort()
    model_path = [f"/model/EsakaK/My_Model/TMP/{EL_model_name[:-7]}/{name}" for name in model_path]

    command = "python3 test.py " \
              "--i_frame_model_name IntraSS " \
              "--i_frame_model_path  " \
              "/model/EsakaK/My_Model/LSSVCM/IntraSS_v2/q1.pth.tar " \
              "/model/EsakaK/My_Model/LSSVCM/IntraSS_v2/q2.pth.tar " \
              "/model/EsakaK/My_Model/LSSVCM/IntraSS_v2/q3.pth.tar " \
              "/model/EsakaK/My_Model/LSSVCM/IntraSS_v2/q4.pth.tar " \
              "--model_path " \
              f"{model_path[0]} " \
              f"{model_path[1]} " \
              f"{model_path[2]} " \
              f"{model_path[3]} " \
              "--test_config /code/data/config/hevc_sequence_config_ablation.json " \
              "--cuda 1 " \
              "--worker 4 " \
              "--cuda_device 0,1,2,3 " \
              "--write_stream 1 " \
              f"--output_path /data/EsakaK/output/{EL_model_name} " \
              f"--stream_path /data/EsakaK/output/{EL_model_name}/bin " \
              "--save_decoded_mv 0 " \
              f"--EL_model_name {EL_model_name}"

    return command


def load_test_command(experiment_name, EL_model_name):
    model_path = os.listdir(f'/model/EsakaK/My_Model/TMP/{experiment_name}')
    model_path.sort()
    model_path = [f"/model/EsakaK/My_Model/TMP/{experiment_name}/{name}" for name in model_path]

    command = "python3 test.py " \
              "--i_frame_model_name IntraSS " \
              "--i_frame_model_path  " \
              "/model/EsakaK/My_Model/LSSVCM/IntraSS_v2/q1.pth.tar " \
              "/model/EsakaK/My_Model/LSSVCM/IntraSS_v2/q2.pth.tar " \
              "/model/EsakaK/My_Model/LSSVCM/IntraSS_v2/q3.pth.tar " \
              "/model/EsakaK/My_Model/LSSVCM/IntraSS_v2/q4.pth.tar " \
              "--model_path " \
              f"{model_path[0]} " \
              f"{model_path[1]} " \
              f"{model_path[2]} " \
              f"{model_path[3]} " \
              "--test_config /code/data/config/hevc_B.json " \
              "--cuda 1 " \
              "--worker 2 " \
              "--cuda_device 0,1 " \
              "--write_stream 1 " \
              f"--output_path /data/EsakaK/output/{experiment_name} " \
              f"--stream_path /data/EsakaK/output/{experiment_name}/bin " \
              "--save_decoded_mv 0 " \
              f"--EL_model_name {EL_model_name}"

    return command


def load_cuda_idx(argv):
    if len(argv) > 1:
        cuda_idxs = argv[1].split(',')
    else:
        n_cuda = torch.cuda.device_count()
        cuda_idxs = [i for i in range(n_cuda)]
    cuda_idxs = [int(i) for i in cuda_idxs]
    return cuda_idxs


def get_pretrained_weights():
    image_model_folder = '/data/1339417445/DMC/benchmark/IntraNoAR/'
    image_models = [
        f'{image_model_folder}/ckpt_q1.pth.tar',
        f'{image_model_folder}/ckpt_q2.pth.tar',
        f'{image_model_folder}/ckpt_q3.pth.tar',
        f'{image_model_folder}/ckpt_q4.pth.tar',
        f'{image_model_folder}/ckpt_q5.pth.tar',
        f'{image_model_folder}/ckpt_q6.pth.tar',
    ]
    me_net_path = "/data/1339417445/DMC/benchmark/spynet_finetune/cur_0622_FA_t04_t2_epo_5.pth"

    mv_enc_dec_folder = '/data/1339417445/DMC/benchmark/mv_enc_dec_pretrain/'
    mv_enc_dec_models = [
        f'{mv_enc_dec_folder}/mv_enc_dec_q1.pth',
        f'{mv_enc_dec_folder}/mv_enc_dec_q2.pth',
        f'{mv_enc_dec_folder}/mv_enc_dec_q3.pth',
        f'{mv_enc_dec_folder}/mv_enc_dec_q4.pth',
    ]

    return image_models, me_net_path, mv_enc_dec_models


def get_best_model_path(best_model_root):
    model_path = ""
    model_list = os.listdir(best_model_root)
    assert len(model_list) == 4
    for q in ['q1', 'q2', 'q3', 'q4']:
        for model_name in model_list:
            if q in model_name:
                model_path += f"{best_model_root}/{model_name} "
    return model_path


def worker(input_command):
    print(input_command)
    os.system(input_command)


def submit_commands(commands):
    with Pool(len(commands)) as p:
        p.map(worker, commands)
