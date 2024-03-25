set -ex
python3 /home/sheng/Muscle_Seg_updated/train.py \
--dataroot /home/sheng/datasets/muscle/generated_results/all_data \
--checkpoints_dir /home/sheng/Muscle_Seg_updated/checkpoints \
--name TmsCoarseL1Noisy_UNet3D_50_V3 \
--dataset_mode tms \
--num_threads 1 \
--batch_size 1 \
--input_patch_size 10 \
--model tms \
--net tms \
--input_nc 1 \
--output_nc 1 \
--n_epochs 50 \
--save_epoch_freq 10 \
--gpu_ids 0