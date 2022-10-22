from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from models import KBCModel


class PruneEmbedding(nn.Embedding):
    def __init__(self, *args, **kwargs):
        super(PruneEmbedding, self).__init__(*args, **kwargs)
        self.prune_mask = torch.ones(list(self.weight.shape))
        self.prune_flag = False

    def forward(self, input):
        if not self.prune_flag:
            weight = self.weight
        else:
            weight = self.weight * self.prune_mask
        return self._embed_forward(input, weight)

    def _embed_forward(self, input, weight):
        self.sparse = False
        return F.embedding(
            input, weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)

    def set_prune_flag(self, flag):
        self.prune_flag = flag


class PruneRelComplEx(KBCModel):
    def __init__(
        self, sizes: Tuple[int, int, int], rank: int,
        init_size: float = 1e-3
    ):
        super(PruneRelComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.prune_flag = False

        self.embeddings = nn.ModuleList([
            PruneEmbedding(s, 2 * rank, sparse=True)
            for s in sizes[:2]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size

        print("init")

    def forward(self, x, flagrel=0):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        rhs_scores, rel_scores = None, None
        if not self.prune_flag:
            to_score_entity = self.embeddings[0].weight
            to_score_rel = self.embeddings[1].weight
        else:
            to_score_entity = self.embeddings[0].weight * self.embeddings[0].prune_mask
            to_score_rel = self.embeddings[1].weight * self.embeddings[1].prune_mask
        to_score_entity = to_score_entity[:,
                                          :self.rank], to_score_entity[:, self.rank:]
        rhs_scores = (
            (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score_entity[0].transpose(0, 1) +
            (lhs[0] * rel[1] + lhs[1] * rel[0]
             ) @ to_score_entity[1].transpose(0, 1)
        )

        to_score_rel = to_score_rel[:, :self.rank], to_score_rel[:, self.rank:]
        rel_scores = (
            (lhs[0] * rhs[0] + lhs[1] * rhs[1]) @ to_score_rel[0].transpose(0, 1) +
            (lhs[0] * rhs[1] - lhs[1] * rhs[0]
             ) @ to_score_rel[1].transpose(0, 1)
        )
        factors = self.get_factor(x)
        if flagrel == 1:
            return rel_scores, factors
        elif flagrel == 0:
            return rhs_scores, factors
        else:
            return (rhs_scores, rel_scores), factors

    def get_factor(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        return [(torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2))]

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        rhs_scores = None

        to_score_entity = self.embeddings[0].weight
        to_score_entity = to_score_entity[:,
                                          :self.rank], to_score_entity[:, self.rank:]
        rhs_scores = (
            (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score_entity[0].transpose(0, 1) +
            (lhs[0] * rel[1] + lhs[1] * rel[0]
             ) @ to_score_entity[1].transpose(0, 1)
        )
        factors = self.get_factor(x)
        return rhs_scores, factors

    def set_prune_flag(self, flag):
        self.prune_flag = flag
        for module in self.embeddings:
            module.set_prune_flag(flag)

    # def get_ranking(
    #         self, queries: torch.Tensor,
    #         filters: Dict[Tuple[int, int], List[int]],
    #         batch_size: int = 1000, chunk_size: int = -1
    # ):
    #     """
    #     Returns filtered ranking for each queries.
    #     :param queries: a torch.LongTensor of triples (lhs, rel, rhs)
    #     :param filters: filters[(lhs, rel)] gives the rhs to filter from ranking
    #     :param batch_size: maximum number of queries processed at once
    #     :return:
    #     """
    #     ranks = torch.ones(len(queries))
    #     with tqdm(total=queries.shape[0], unit='ex') as bar:
    #         bar.set_description(f'Evaluation')
    #         with torch.no_grad():
    #             b_begin = 0
    #             while b_begin < len(queries):
    #                 these_queries = queries[b_begin:b_begin + batch_size]
    #                 target_idxs = these_queries[:, 2].cpu().tolist()
    #                 scores, _ = self.score(these_queries)
    #                 targets = torch.stack([scores[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)

    #                 for i, query in enumerate(these_queries):
    #                     filter_out = filters[(query[0].item(), query[1].item())]
    #                     filter_out += [queries[b_begin + i, 2].item()]   # Add the tail of this (b_begin + i) query
    #                     scores[i, torch.LongTensor(filter_out)] = -1e6
    #                 ranks[b_begin:b_begin + batch_size] += torch.sum(
    #                     (scores >= targets).float(), dim=1
    #                 ).cpu()
    #                 b_begin += batch_size
    #                 bar.update(batch_size)
    #     return ranks


class SparseEmbedding(nn.Module):
    def __init__(self, size, rank):
        super(SparseEmbedding, self).__init__()
        self.embed_num = size
        self.re_size = rank
        self.im_size = rank

        # coo tensor parameters
        self.re_index: torch.Tensor
        self.re_value: torch.Tensor
        self.im_index: torch.Tensor
        self.im_value: torch.Tensor

        # constant whole embedding
        self.dense_whole_re_feats = None
        self.dense_whole_im_feats = None

    def forward(self, input: torch.tensor):
        """generate sparse embedding features the type with space coo tensor

        Args:
            input (torch.Tensor): input index tensor

        Returns:
            Tuple[torch.spare_coco_tensor]: return real part and imaginary part features
        """
        assert input.dim() == 1, "Input tensor must be one dimension."

        if self.dense_whole_re_feats is None or self.dense_whole_im_feats is None:
            self.get_whole_embedding()

        # row selection
        dense_re_feats = torch.index_select(self.dense_whole_re_feats, 0, input)
        dense_im_feats = torch.index_select(self.dense_whole_im_feats, 0, input)

        return dense_re_feats, dense_im_feats

    def get_whole_embedding(self):
        if self.dense_whole_re_feats is None or self.dense_whole_im_feats is None:
            # make torch coo tensor
            re_feats = torch.sparse_coo_tensor(
                self.re_index, self.re_value, (self.embed_num, self.re_size)
            )
            im_feats = torch.sparse_coo_tensor(
                self.im_index, self.im_value, (self.embed_num, self.im_size)
            )

            # cast to dense tensor
            dense_re_feats = re_feats.to_dense()
            dense_im_feats = im_feats.to_dense()

            # set constant
            self.dense_whole_re_feats = dense_re_feats
            self.dense_whole_im_feats = dense_im_feats

        return self.dense_whole_re_feats, self.dense_whole_im_feats


class SparsePruneRelComplEx(KBCModel):
    """
    A model used for spare inference
    """
    def __init__(
        self, sizes: Tuple[int, int, int], rank: int
    ):
        super(SparsePruneRelComplEx, self).__init__()

        self.embeddings = nn.ModuleList([
            SparseEmbedding(s, rank)
            for s in sizes[:2]
        ])

        print("init sparse model")

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])

        to_score_entity = self.embeddings[0].get_whole_embedding()

        rhs_scores = None
        rhs_scores = (
            (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score_entity[0].transpose(0, 1) +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score_entity[1].transpose(0, 1)
        )

        # the factor is set to None
        return rhs_scores, None


if __name__ == "__main__":
    model = PruneRelComplEx((14541, 474, 14541), 100)
    model.debug()
    weights = []
    for name, module in model.named_modules():
        if hasattr(module, "prune_mask"):
            weights.append(module.weight.clone().cpu().detach().numpy())
