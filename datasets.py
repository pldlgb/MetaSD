import json
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import torch

from models import KBCModel


class Dataset(object):
    def __init__(self, data_path: str, name: str):
        self.root = os.path.join(data_path, name)

        self.data = {}
        for f in ['train', 'test', 'valid']:
            in_file = open(os.path.join(self.root, f + '.pickle'), 'rb')
            self.data[f] = pickle.load(in_file)

        print(self.data['train'].shape)

        maxis = np.max(self.data['train'], axis=0)
        self.n_entities = int(max(maxis[0], maxis[2]) + 1)
        self.n_predicates = int(maxis[1] + 1)
        self.n_predicates *= 2

        inp_f = open(os.path.join(self.root, 'to_skip.pickle'), 'rb')
        self.to_skip: Dict[str, Dict[Tuple[int, int],
                                     List[int]]] = pickle.load(inp_f)
        inp_f.close()

    def get_weight(self):
        appear_list = np.zeros(self.n_entities)
        copy = np.copy(self.data['train'])
        for triple in copy:
            h, r, t = triple
            appear_list[h] += 1
            appear_list[t] += 1

        w = appear_list / np.max(appear_list) * 0.9 + 0.1
        return w

    def get_examples(self, split):
        return self.data[split]

    def get_train(self, split="train"):
        copy = np.copy(self.data[split])
        tmp = np.copy(copy[:, 0])
        copy[:, 0] = copy[:, 2]
        copy[:, 2] = tmp
        copy[:, 1] += self.n_predicates // 2  # has been multiplied by two.
        return np.vstack((self.data[split], copy))

    def get_valid(self):
        copy = np.copy(self.data['valid'])
        tmp = np.copy(copy[:, 0])
        copy[:, 0] = copy[:, 2]
        copy[:, 2] = tmp
        copy[:, 1] += self.n_predicates // 2  # has been multiplied by two.
        return np.vstack((self.data['valid'], copy))

    def get_all(self):
        data = []
        for split in ["train", "valid", "test"]:
            copy = np.copy(self.data[split])
            tmp = np.copy(copy[:, 0])
            copy[:, 0] = copy[:, 2]
            copy[:, 2] = tmp
            copy[:, 1] += self.n_predicates // 2  # has been multiplied by two.
            data.append(np.vstack((self.data[split], copy)))
        return np.vstack((data[0], data[1], data[2]))

    def eval(
            self, model: KBCModel, split: str, n_queries: int = -1, missing_eval: str = 'both',
            at: Tuple[int] = (1, 3, 10), log_result=False, args=None, save_path=None
    ):
        model.eval()
        test = self.get_examples(split)
        examples = torch.from_numpy(test.astype('int64')).cuda()
        missing = [missing_eval]
        if missing_eval == 'both':
            missing = ['rhs', 'lhs']

        mean_reciprocal_rank = {}
        hits_at = {}

        flag = False
        for m in missing:
            q = examples.clone()
            if n_queries > 0:
                permutation = torch.randperm(len(examples))[:n_queries]
                q = examples[permutation]
            if m == 'lhs':
                tmp = torch.clone(q[:, 0])
                q[:, 0] = q[:, 2]
                q[:, 2] = tmp
                q[:, 1] += self.n_predicates // 2

            ranks = model.get_ranking(q, self.to_skip[m], batch_size=500)

            if log_result:
                if not flag:
                    results = np.concatenate((q.cpu().detach().numpy(),
                                              np.expand_dims(ranks.cpu().detach().numpy(), axis=1)), axis=1)
                    flag = True
                else:
                    results = np.concatenate((results, np.concatenate((q.cpu().detach().numpy(),
                                                                       np.expand_dims(ranks.cpu().detach().numpy(), axis=1)), axis=1)), axis=0)

            mean_reciprocal_rank[m] = torch.mean(1. / ranks).item()
            hits_at[m] = torch.FloatTensor((list(map(
                lambda x: torch.mean((ranks <= x).float()).item(),
                at
            ))))

        return mean_reciprocal_rank, hits_at

    def get_shape(self):
        return self.n_entities, self.n_predicates, self.n_entities

    def aux(self, dataset):
        train_dataset = self.data["train"]
        print(len(train_dataset))
        relcount = {}
        for triple in train_dataset:
            rel = str(triple[1])
            if rel not in relcount:
                relcount[rel] = 1
            else:
                relcount[rel] += 1
        sortv = sorted(relcount.items(), key=lambda x: x[1], reverse=True)
        relsort = []
        for i in sortv:
            relsort.append(i[1])

        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        pos_list = [0, 50, 100, 150, 200]  # ,236]
        pos_bias = [15, 50, 100, 150, 200]
        pos_loc = [15700, 1210+1000, 476+500, 246+400, 137+300]
        pos_value = [15989, 1210, 476, 246, 137]
        name_list = ["award nominee", "medal won", "domestic tuition",
                     "film release date", "organization founder"]  # ,"film_appearance"]
        plt.figure(figsize=(8, 4))
        fontsize = 16

        ax = plt.axes()
        ax.xaxis.set_major_locator(ticker.FixedLocator((pos_list)))
        ax.xaxis.set_major_formatter(ticker.FixedFormatter((name_list)))

        plt.bar(range(len(relsort)), relsort)

        labels = ax.get_xticklabels()
        plt.setp(labels, rotation=10, size=fontsize)

        for a, b, c in zip(pos_bias, pos_loc, pos_value):
            plt.text(a, b, c, ha='center', va='bottom', fontsize=fontsize)
        yt = range(0, 18001, 2000)
        # for i in yt:
        #     print(i)
        ytv = ["0", "2k", "4k", "6k", "8k", "10k", "12k", "14k", "16k", "18k"]
        plt.yticks(yt, ytv, size=fontsize)
        plt.ylabel('Count', size=fontsize)
        # ax.yaxis.set_label_coords(-0.1,17000)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.savefig(dataset+"rel.pdf")

        with open('reldata.txt', 'w') as f:
            json_str = json.dumps(sortv, indent=0)
            f.write(json_str)
            f.write('\n')

    def build_pruned(self):
        train_dataset = self.data["train"]
        print(len(train_dataset))
        relcount = {}
        pruned_train = []
        for triple in train_dataset:
            rel = int(triple[1])
            if rel not in relcount:
                relcount[rel] = 1
            else:
                relcount[rel] += 1

            if relcount[rel] <= 1000:
                pruned_train.append(triple)
        self.data["pruned"] = np.array(pruned_train)

    def build_longtail(self, index, span):
        train_dataset = self.data["train"]
        print(len(train_dataset))
        relcount = {}
        for triple in train_dataset:
            rel = int(triple[1])
            if rel not in relcount:
                relcount[rel] = 1
            else:
                relcount[rel] += 1
        sortv = sorted(relcount.items(), key=lambda x: x[1], reverse=True)
        # print(sortv)
        test_dataset = self.data["test"]
        long_tail = []
        for pair in sortv[index:index+span]:
            long_tail.append(pair[0])
        longtail_data = []
        for triple in test_dataset:
            if int(triple[1]) in long_tail:
                longtail_data.append(triple)

        longtail_data = np.array(longtail_data)
        print("ok")
        self.data["longtail"] = longtail_data
        return longtail_data.shape
        # relsort = []
        # for i in sortv:
        #     relsort.append(i[1])
