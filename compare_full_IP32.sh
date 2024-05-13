echo -e '\n************x2_FL************'
python3 compare_rd_video.py \
    --compare_between class \
    --output_path stdout \
    --base_method SHM-12.4  \
    --plot_scheme separate \
    --distortion_metrics rgb_psnr \
    --plot_path output/LSSVC_rgb_psnr/x2_FL \
    --log_paths SHM-12.4 json_results/hevc/IP32/x2_FL.json \
    LSSVC json_results/LSSVC/IP32/x2_FL.json \
    VTM-21.2 json_results/VTM/IP32/x2_FL.json

echo -e '\n************x1.5_FL************'
python3 compare_rd_video.py \
    --compare_between class \
    --output_path stdout \
    --base_method SHM-12.4  \
    --plot_scheme separate \
    --distortion_metrics rgb_psnr \
    --plot_path output/LSSVC_rgb_psnr/x1_5_FL \
    --log_paths SHM-12.4 json_results/hevc/IP32/x1_5_FL.json \
    LSSVC json_results/LSSVC/IP32/x1_5_FL.json \
    VTM-21.2 json_results/VTM/IP32/x1_5_FL.json

echo -e '\n************x3_FL************'
python3 compare_rd_video.py \
    --compare_between class \
    --output_path stdout \
    --base_method SHM-12.4  \
    --plot_scheme separate \
    --distortion_metrics rgb_psnr \
    --plot_path output/LSSVC_rgb_psnr/x3_FL \
    --log_paths SHM-12.4 json_results/hevc/IP32/x3_FL.json \
    LSSVC json_results/LSSVC/IP32/x3_FL.json \
    VTM-21.2 json_results/VTM/IP32/x3_FL.json

echo -e '\n************x4_FL************'
python3 compare_rd_video.py \
    --compare_between class \
    --output_path stdout \
    --base_method SHM-12.4  \
    --plot_scheme separate \
    --distortion_metrics rgb_psnr \
    --plot_path output/LSSVC_rgb_psnr/x4_FL \
    --log_paths SHM-12.4 json_results/hevc/IP32/x4_FL.json \
    LSSVC json_results/LSSVC/IP32/x4_FL.json \
    VTM-21.2 json_results/VTM/IP32/x4_FL.json