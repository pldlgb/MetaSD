
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict


class Mask(object):
    def __init__(self, model, no_reset=False):
        super(Mask, self).__init__()
        self.model = model
        if not no_reset:
            self.reset()

    @property
    def sparsity(self):
        """Return the percent of weights that have been pruned as a decimal."""
        prunableTensors = []
        for name, module in self.model.named_modules():
            if hasattr(module, "prune_mask"):
                prunableTensors.append(module.prune_mask.detach())

        unpruned = torch.sum(torch.tensor(
            [torch.sum(v) for v in prunableTensors]))
        total = torch.sum(torch.tensor(
            [torch.sum(torch.ones_like(v)) for v in prunableTensors]))
        return 1 - unpruned.float() / total.float()

    @property
    def density(self):
        return 1 - self.sparsity

    def magnitudePruning(self, magnitudePruneFraction, randomPruneFraction=0, ep=1e-8):
        weights = []
        for name, module in self.model.named_modules():
            if hasattr(module, "prune_mask"):
                weights.append(module.weight.clone().cpu().detach().numpy())

        # only support one time pruning
        self.reset()
        prunableTensors = []
        for name, module in self.model.named_modules():
            if hasattr(module, "prune_mask"):
                prunableTensors.append(module.prune_mask.detach())

        number_of_remaining_weights = torch.sum(torch.tensor(
            [torch.sum(v) for v in prunableTensors])).cpu().numpy()
        number_of_weights_to_prune_magnitude = np.ceil(
            magnitudePruneFraction * number_of_remaining_weights).astype(int)
        number_of_weights_to_prune_random = np.ceil(
            randomPruneFraction * number_of_remaining_weights).astype(int)
        random_prune_prob = number_of_weights_to_prune_random / \
            (number_of_remaining_weights - number_of_weights_to_prune_magnitude)

        # Create a vector of all the unpruned weights in the model.
        weight_vector = np.concatenate([v.flatten() for v in weights])
        threshold = np.sort(np.abs(weight_vector))[min(
            number_of_weights_to_prune_magnitude, len(weight_vector) - 1)]

        # apply the mask
        for name, module in self.model.named_modules():
            if hasattr(module, "prune_mask"):
                module.prune_mask = (
                    torch.abs(module.weight) >= threshold).float()
                module.prune_mask = module.prune_mask + ep
                # random weights been pruned
                module.prune_mask[torch.rand_like(
                    module.prune_mask) < random_prune_prob] = ep

    def reset(self):
        for name, module in self.model.named_modules():
            if hasattr(module, "prune_mask"):
                module.prune_mask = torch.ones_like(module.weight)


def save_mask(epoch, model, filename):
    pruneMask = OrderedDict()

    for name, module in model.named_modules():
        if hasattr(module, "prune_mask"):
            pruneMask[name] = module.prune_mask.cpu().type(torch.bool)

    torch.save({"epoch": epoch, "pruneMask": pruneMask}, filename)


def load_mask(model, state_dict, device):
    # set_trace()
    for name, module in model.named_modules():
        if hasattr(module, "prune_mask"):
            module.prune_mask.data = state_dict[name].to(device).float()

    return model


if __name__ == "__main__":
    import os
    print(os.getcwd())
    from prunemodel import PruneRelComplEx
    # net = prune_resnet18().cuda()
    net = PruneRelComplEx((14541, 474, 14541), 100).cuda()
    # image = torch.rand(3, 224, 224).cuda()
    input_batch = torch.randint(200, (21, 3))
    input_batch = input_batch.cuda()
    mask = Mask(net)

    for rate in (0, 0.5):
        # prune 0%
        # mask.magnitudePruning(0)
        mask.magnitudePruning(rate)
        net.set_prune_flag(True)
        # a = net(image)
        a, factors = net(input_batch)
        print("prune, density is {}, avg is {}".format(
            mask.density, a[0].mean()))
        # net.set_prune_flag(False)
        mask.magnitudePruning(rate+0.1)
        b, factors = net.forward(input_batch)
        print("no prune, density is {}, avg is {}".format(
            mask.density, b[0].mean()))
        loss = nn.CrossEntropyLoss(reduction='mean')
        ent_truth = input_batch[:, 2]
        l = loss(a[0], ent_truth) + loss(b[0], ent_truth)
        l.backward()
