CUDA_VISIBLE_DEVICES=0 python metasd.py --dataset WN18RR --batch_size 1000 --reg 1e-1 \
--use_weight True --use_relaux False -save -id debug