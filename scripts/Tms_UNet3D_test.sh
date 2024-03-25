set -ex
python3 /home/sheng/Muscle_Seg_updated/test.py \
--dataroot /home/sheng/datasets/muscle/generated_results/all_data \
--checkpoints_dir /home/sheng/Muscle_Seg_updated/checkpoints \
--results_dir /home/sheng/Muscle_Seg_updated/results \
--eval \
--name TmsCoarseL1_UNet3D_50 \
--epoch latest \
--dataset_mode tms \
--num_threads 0 \
--serial_batches \
--batch_size 1 \
--input_patch_size 3 \
--model tms \
--net tms \
--input_nc 1 \
--output_nc 1 \
--save_prediction 1 \
--gpu_ids 1