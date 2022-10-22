# eval in dense mode
CUDA_VISIBLE_DEVICES=3 python metasd.py --dataset FB237 --batch_size 5000 --reg 5e-2 --prune_percent 0.8 --use_relaux True -save -id test -ckpt ../MetaSDFiles/logs/PruneRelComplEx_FB237_standard
# CUDA_VISIBLE_DEVICES=2 python metasd.py --dataset FB237 --batch_size 5000 --reg 5e-2 --prune_percent 0.9 --use_relaux True -train -save -id standard 