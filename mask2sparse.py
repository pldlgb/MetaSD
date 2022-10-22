import os
import argparse

import torch

from datasets import Dataset
from prunemodel import PruneRelComplEx
from mask import Mask


def convert_mask_to_coo(prune_mask, prune_value):
    print(prune_mask.max())
    select_mask = (prune_mask > 0.5)
    print("Select value with {:.2f} percent".format(select_mask.sum() / select_mask.numel()))

    row_arange = torch.arange(0, prune_mask.size(0)).unsqueeze(-1)
    col_arange = torch.arange(0, prune_mask.size(1)).unsqueeze(0)

    row_index = torch.masked_select(row_arange, select_mask)
    col_index = torch.masked_select(col_arange, select_mask)
    index = torch.stack([row_index, col_index])
    value = torch.masked_select(prune_value, select_mask)

    print(index[:, :10])
    print(value[:10])

    print(prune_mask.size())
    print(index.size())
    print(value.size())
    return index, value


def convert_complex_to_coco(prune_mask, prune_value, rank):
    re_index, re_value = convert_mask_to_coo(prune_mask[:, :rank], prune_value[:, :rank])
    im_index, im_value = convert_mask_to_coo(prune_mask[:, rank:], prune_value[:, rank:])
    return {
        're_index': re_index,
        're_value': re_value,
        'im_index': im_index,
        'im_value': im_value
    }


if __name__ == "__main__":
    datasets = ['WN18RR', 'FB237', 'YAGO3-10']

    parser = argparse.ArgumentParser(
        description="Tensor Factorization for Knowledge Graph Completion")

    parser.add_argument('--dataset', choices=datasets,
                        default='WN18RR', help="Dataset in {}".format(datasets))
    parser.add_argument('--model', type=str, default='PruneRelComplEx')
    parser.add_argument('--rank', default=2000, type=int,
                        help="Factorization rank.")
    parser.add_argument('-save', '--do_save', action='store_true')
    parser.add_argument('-path', '--save_path', type=str, default='../MetaSDFiles/logs/')
    parser.add_argument('-id', '--model_id', type=str, default='0')
    parser.add_argument('-ckpt', '--checkpoint', type=str, default='')

    # contrast with pruned model
    parser.add_argument('--prune_percent', type=float,
                        default=0.9, help="whole prune percentage")
    args = parser.parse_args()
    print(args)
    args.verbose = True

    if args.do_save:
        assert args.save_path
        save_suffix = f'Sparse_{args.prune_percent}' + args.model + '_' + args.dataset + '_' + args.model_id

        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)

        save_path = os.path.join(args.save_path, save_suffix)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

    """ get dataset and build model """
    dataset = Dataset("../MetaSDFiles/data", args.dataset)
    t_model = PruneRelComplEx(dataset.get_shape(), 2000, 1e-3).cuda()
    t_model.load_state_dict(torch.load(os.path.join(
        args.checkpoint, 'final_checkpoint'), map_location='cuda:0'))

    # Prune Part
    pruneMask = Mask(t_model)
    prunePercent = args.prune_percent
    pruneMask.magnitudePruning(prunePercent)

    # Convert mask to sparse checkpoint
    coo_dict = {}
    coo_dict['embedding0'] = convert_complex_to_coco(
        t_model.embeddings[0].prune_mask,
        t_model.embeddings[0].weight,
        args.rank
    )
    coo_dict['embedding1'] = convert_complex_to_coco(
        t_model.embeddings[1].prune_mask,
        t_model.embeddings[1].weight,
        args.rank
    )
    torch.save(coo_dict, os.path.join(save_path, 'final_sparse_checkpoint'))
