# eval in sparse mode
CUDA_VISIBLE_DEVICES=3 python metasd.py --dataset FB237 --prune_percent 0.8 -save -id test_sparse -ckpt ../MetaSDFiles/logs/Sparse_0.8PruneRelComplEx_FB237_test/ -sparse_infer