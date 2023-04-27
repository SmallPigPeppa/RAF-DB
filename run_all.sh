#python train_all.py \
#  --max_epochs 50 \
#  --batch_size 128 \
#  --lr 0.1 \
#  --run_name simclr \
#  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/RAF-DB \
#  --ckpt_path /mnt/mmtech01/usr/liuwenzhuo/code/test-code/ISSL-cka-fixbug/experiments/2023_04_13_12_06_58-reparam-fixbn-1x3

python train_all.py \
  --max_epochs 50 \
  --batch_size 128 \
  --lr 0.1 \
  --run_name simclr-finetune \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/RAF-DB \
  --ckpt_path /mnt/mmtech01/usr/liuwenzhuo/code/test-code/ISSL-simclr/experiments/2023_04_17_13_15_47-finetune

#/mnt/mmtech01/usr/liuwenzhuo/code/test-code/ISSL-cka-fixbug/experiments/2023_04_13_12_06_58-reparam-fixbn-1x3/1r8ki6ns/reparam-fixbn-1x3-task4-ep=500-1r8ki6ns.ckpt
#/mnt/mmtech01/usr/liuwenzhuo/code/test-code/ISSL-cka/experiments/2023_04_12_02_58_10-reparam-fixbn-1x3/w66mgepg/reparam-fixbn-1x3-task4-ep=499-w66mgepg.ckpt
