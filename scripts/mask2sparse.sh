# convert mask to sparse model
export CUDA_VISIBLE_DEVICES=6
python mask2sparse.py --dataset FB237 --prune_percent 0.8 -save -id test -ckpt ../MetaSDFiles/logs/PruneRelComplEx_FB237_standard
