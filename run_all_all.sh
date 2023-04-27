#python train_all.py \
#  --max_epochs 50 \
#  --batch_size 128 \
#  --lr 0.1 \
#  --run_name simclr \
#  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/RAF-DB \
#  --ckpt_path /mnt/mmtech01/usr/liuwenzhuo/code/test-code/ISSL-cka-fixbug/experiments/2023_04_13_12_06_58-reparam-fixbn-1x3
#
#python train_all.py \
#  --max_epochs 50 \
#  --batch_size 128 \
#  --lr 0.1 \
#  --run_name simclr-finetune \
#  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/RAF-DB \
#  --ckpt_path /mnt/mmtech01/usr/liuwenzhuo/code/test-code/ISSL-simclr/experiments/2023_04_17_13_15_47-finetune


python train_all.py \
  --max_epochs 50 \
  --batch_size 128 \
  --lr 0.1 \
  --run_name byol \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/RAF-DB \
  --ckpt_path /mnt/mmtech01/usr/liuwenzhuo/code/test-code/ISSL-byol/experiments/2023_04_13_15_27_14-reparam-fixbn-1x3

python train_all.py \
  --max_epochs 50 \
  --batch_size 128 \
  --lr 0.1 \
  --run_name byol-finetune \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/RAF-DB \
  --ckpt_path /mnt/mmtech01/usr/liuwenzhuo/code/test-code/ISSL-byol/experiments/2023_04_17_15_36_42-finetune



python train_all.py \
  --max_epochs 50 \
  --batch_size 128 \
  --lr 0.1 \
  --run_name barlow \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/RAF-DB \
  --ckpt_path /mnt/mmtech01/usr/liuwenzhuo/code/test-code/ISSL-barlow/experiments/2023_04_14_09_20_17-reparam-fixbn-3x3

python train_all.py \
  --max_epochs 50 \
  --batch_size 128 \
  --lr 0.1 \
  --run_name barlow-finetune \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/RAF-DB \
  --ckpt_path /mnt/mmtech01/usr/liuwenzhuo/code/test-code/ISSL-barlow/experiments/2023_04_17_12_58_50-finetune

python train_all.py \
  --max_epochs 50 \
  --batch_size 128 \
  --lr 0.1 \
  --run_name moco \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/RAF-DB \
  --ckpt_path /mnt/mmtech01/usr/liuwenzhuo/code/test-code/ISSL-mocov2plus/experiments/2023_04_15_12_47_20-reparam-fixbn-3x3

python train_all.py \
  --max_epochs 50 \
  --batch_size 128 \
  --lr 0.1 \
  --run_name moco-finetune \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/RAF-DB \
  --ckpt_path /mnt/mmtech01/usr/liuwenzhuo/code/test-code/ISSL-mocov2plus/experiments/2023_04_17_13_10_58-finetune


#/mnt/mmtech01/usr/liuwenzhuo/code/test-code/ISSL-cka-fixbug/experiments/2023_04_13_12_06_58-reparam-fixbn-1x3/1r8ki6ns/reparam-fixbn-1x3-task4-ep=500-1r8ki6ns.ckpt
#/mnt/mmtech01/usr/liuwenzhuo/code/test-code/ISSL-cka/experiments/2023_04_12_02_58_10-reparam-fixbn-1x3/w66mgepg/reparam-fixbn-1x3-task4-ep=499-w66mgepg.ckpt
