python train_all.py \
  --max_epochs 50 \
  --batch_size 128 \
  --lr 0.1 \
  --run_name simclr-joint \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/RAF-DB \
  --ckpt_path /mnt/mmtech01/usr/liuwenzhuo/code/test-code/ISSL-simclr/experiments/2023_04_27_16_21_45-upbound



python train_all.py \
  --max_epochs 50 \
  --batch_size 128 \
  --lr 0.1 \
  --run_name byol-joint \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/RAF-DB \
  --ckpt_path /mnt/mmtech01/usr/liuwenzhuo/code/test-code/ISSL-byol/experiments/2023_04_27_16_23_59-upbound



python train_all.py \
  --max_epochs 50 \
  --batch_size 128 \
  --lr 0.1 \
  --run_name barlow-joint \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/RAF-DB \
  --ckpt_path /mnt/mmtech01/usr/liuwenzhuo/code/test-code/ISSL-barlow/experiments/2023_04_27_16_25_43-upbound



python train_all.py \
  --max_epochs 50 \
  --batch_size 128 \
  --lr 0.1 \
  --run_name moco-joint \
  --data_path /mnt/mmtech01/usr/liuwenzhuo/torch_ds/RAF-DB \
  --ckpt_path /mnt/mmtech01/usr/liuwenzhuo/code/test-code/ISSL-mocov2plus/experiments/2023_04_27_16_24_51-upbound


