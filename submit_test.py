import os

experiment_name = 'LSSVC_IP32'
model_name = 'LSSVC_extend'
command = "python3 test.py " \
          "--i_frame_model_name IntraSS " \
          "--i_frame_model_path  " \
          "/model/EsakaK/My_Model/LSSVCM/IntraSS/q1.pth.tar " \
          "/model/EsakaK/My_Model/LSSVCM/IntraSS/q2.pth.tar " \
          "/model/EsakaK/My_Model/LSSVCM/IntraSS/q3.pth.tar " \
          "/model/EsakaK/My_Model/LSSVCM/IntraSS/q4.pth.tar " \
          "--model_path " \
          "/model/EsakaK/My_Model/LSSVCM/LSSVC/q1.pth " \
          "/model/EsakaK/My_Model/LSSVCM/LSSVC/q2.pth " \
          "/model/EsakaK/My_Model/LSSVCM/LSSVC/q3.pth " \
          "/model/EsakaK/My_Model/LSSVCM/LSSVC/q4.pth " \
          "--test_config recommend_test_config.json " \
          "--cuda 1 " \
          "--worker 8 " \
          "--cuda_device 0,1,2,3,4,5,6,7 " \
          "--write_stream 0 " \
          f"--output_path /output/{experiment_name} " \
          "--stream_path /output/out_bin " \
          "--save_decoded_mv 0 " \
          f"--model_name {model_name}"

print(command)
os.system(command)
