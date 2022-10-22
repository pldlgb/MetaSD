import os
import random
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, kl

from models import *


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel()
                        for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def save(model, save_path, filename="checkpoint"):
    torch.save(model.state_dict(), os.path.join(save_path, filename))
    embeddings = model.embeddings
    len_emb = len(embeddings)
    if len_emb == 2:
        np.save(os.path.join(save_path, filename+'entity_embedding.npy'),
                embeddings[0].weight.detach().cpu().numpy())
        np.save(os.path.join(save_path, filename+'relation_embedding.npy'),
                embeddings[1].weight.detach().cpu().numpy())
    else:
        print('SAVE ERROR!')


def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
    """
    aggregate metrics for missing lhs and rhs
    :param mrrs: d
    :param hits:
    :return:
    """
    m = (mrrs['lhs'] + mrrs['rhs']) / 2.
    h = (hits['lhs'] + hits['rhs']) / 2.
    return {'MRR': m, 'hits@[1,3,10]': h}


def p_detach(prediction):
    if isinstance(prediction, tuple):
        return (prediction[0].detach(), prediction[1].detach())
    else:
        return prediction.detach()


def f_detach(factors):
    for i in factors:
        for e in i:
            e = e.detach()
    return factors


def myKL_loss(p, t):
    teacher = F.softmax(p[0]/t, -1)
    student = F.softmax(p[1]/t, -1)
    loss_1 = kl.kl_divergence(Categorical(teacher), Categorical(student))
    loss_2 = kl.kl_divergence(Categorical(student), Categorical(teacher))
    return 3 * loss_1.mean()+loss_2.mean()


def KL_loss(p):
    kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
    teacher = F.log_softmax(p[0], -1)
    student = F.log_softmax(p[1], -1)
    loss_1 = kl_loss(student, teacher)
    # loss_2 = kl_loss(teacher, student)
    return 4 * loss_1.mean()  # + loss_2.mean()


def MSEloss(p):
    return torch.mean((p[0] - p[1])**2)


def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask


def nt_xent(x, features2=None, t=0.5):

    if features2 is None:
        out = F.normalize(x, dim=-1)
        d = out.size()
        batch_size = d[0] // 2
        out = out.view(batch_size, 2, -1).contiguous()
        out_1 = out[:, 0]
        out_2 = out[:, 1]
    else:
        batch_size = x.shape[0]
        out_1 = F.normalize(x, dim=-1)
        out_2 = F.normalize(features2, dim=-1)

    # neg score
    out = torch.cat([out_1, out_2], dim=0)
    # print("temperature is {}".format(t))
    neg = torch.exp(torch.mm(out, out.t().contiguous()) / t)

    mask = get_negative_mask(batch_size).cuda()
    neg = neg.masked_select(mask).view(2 * batch_size, -1)

    # pos score
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / t)
    pos = torch.cat([pos, pos], dim=0)

    # estimator g()
    Ng = neg.sum(dim=-1)

    # contrastive loss

    loss = (- torch.log(pos / (pos + Ng)))

    return loss.mean()


def cat_factors(lhs, rel, rhs):
    return [(torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
             torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
             torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2))]


def factor_loss(factors, reg):
    norm = 0
    weight = reg
    for factor in factors:
        h, r, t = factor

        norm += 0.5 * torch.sum(t**2 + h**2)
        norm += 1.5 * torch.sum(h**2 * r**2 + t**2 * r**2)

    return weight * norm / h.shape[0]


def T_train(args, input_batch, t_model, real_optimizer):
    # input_batch = train_examples.cuda()
    t_model.train()
    ent_truth = input_batch[:, 2]
    loss = nn.CrossEntropyLoss(reduction='mean', weight=args.weight)
    if args.use_relaux:
        rel_truth = input_batch[:, 1]
        pred, factors = t_model.forward(input_batch, flagrel=2)
    else:
        pred, factors = t_model.forward(input_batch)
    l_reg = factor_loss(factors, args.reg)
    if args.use_relaux:
        l = loss(pred[0], ent_truth) + l_reg + 4 * loss(pred[1], rel_truth)
    else:
        l = loss(pred, ent_truth) + l_reg

    real_optimizer.zero_grad()
    l.backward()
    # torch.nn.utils.clip_grad_norm_(s_model.parameters(), args.max_grad_norm)
    real_optimizer.step()

    return t_model


def p_criterion(args, e, t_model, real_optimizer, input_batch):
    loss = nn.CrossEntropyLoss(reduction='mean', weight=args.weight)
    t = args.t
    truth = input_batch[:, 2]
    real_optimizer.zero_grad()
    # calculate the grad for non-pruned network
    with torch.no_grad():
        # branch with pruned network
        t_model.set_prune_flag(True)
        predictions_2, factors_2 = t_model.forward(input_batch)
        predictions_2_noGrad = p_detach(predictions_2)
        factors_2_noGrad = f_detach(factors_2)
    t_model.set_prune_flag(False)
    predictions_1, factors_1 = t_model.forward(input_batch)

    l_1 = loss(predictions_1, truth) + factor_loss(factors_1, args.reg) + \
        myKL_loss([predictions_1, predictions_2_noGrad], t).mean()

    l_1.backward()
    # calculate the grad for pruned network
    predictions_1_noGrad = p_detach(predictions_1)
    factors_1_noGrad = f_detach(factors_1)
    t_model.set_prune_flag(True)
    predictions_2, factors_2 = t_model.forward(input_batch)
    l_2 = loss(predictions_2, truth) + factor_loss(factors_2, args.reg) + \
        myKL_loss([predictions_2, predictions_1_noGrad], t).mean()
    l_2.backward()

    real_optimizer.step()
    t_model.set_prune_flag(False)
    return t_model


def rp_criterion(args, e, t_model, real_optimizer, input_batch):
    loss = nn.CrossEntropyLoss(reduction='mean')
    t = args.t
    ent_truth = input_batch[:, 2]
    rel_truth = input_batch[:, 1]
    real_optimizer.zero_grad()
    # calculate the grad for non-pruned network
    with torch.no_grad():
        # branch with pruned network
        t_model.set_prune_flag(True)
        predictions_2, factors_2 = t_model.forward(input_batch, 2)
        predictions_2_noGrad = p_detach(predictions_2)
        factors_2_noGrad = f_detach(factors_2)
    t_model.set_prune_flag(False)
    predictions_1, factors_1 = t_model.forward(input_batch, 2)

    l_1 = loss(predictions_1[0], ent_truth) + 4*loss(predictions_1[1], rel_truth) + factor_loss(factors_1, args.reg) + \
        myKL_loss([predictions_1[0], predictions_2_noGrad[0]], t).mean()

    l_1.backward()
    # calculate the grad for pruned network
    predictions_1_noGrad = p_detach(predictions_1)
    factors_1_noGrad = f_detach(factors_1)
    t_model.set_prune_flag(True)
    predictions_2, factors_2 = t_model.forward(input_batch, 2)
    l_2 = loss(predictions_2[0], ent_truth) + 4*loss(predictions_2[1], rel_truth) +\
        factor_loss(factors_2, args.reg) + \
        myKL_loss([predictions_2[0], predictions_1_noGrad[0]], t).mean()
    l_2.backward()

    real_optimizer.step()
    t_model.set_prune_flag(False)
    return t_model


def d_criterion(args, e, t_model, s_model, input_batch, mask=None, flagrel=False, teacher_grad=False):
    loss = nn.CrossEntropyLoss(reduction='mean', weight=args.weight)
    t = args.t
    if teacher_grad:
        pred_1, _ = t_model.forward(input_batch)
    else:
        with torch.no_grad():
            pred_1, _ = t_model.forward(input_batch)

    if isinstance(s_model, KBCModel):
        pred_2, factors_2 = s_model.forward(input_batch)
    else:
        pred_2, factors_2 = f_forward(s_model, input_batch, mask, flagrel)

    l_reg = factor_loss(factors_2, args.reg)

    truth = input_batch[:, 2]
    l_mul = myKL_loss([pred_1, pred_2], t).mean()
    # F.mse_loss(predictions_1, predictions_2)
    l = loss(pred_2, truth) + l_reg + l_mul

    return l


def f_forward(s_model, input_batch, mask, flagrel=False):
    x = input_batch
    if mask:
        weight0 = s_model['embeddings.0.weight'] * mask[0]
        weight1 = s_model['embeddings.1.weight'] * mask[1]
    else:
        weight0 = s_model['embeddings.0.weight']
        weight1 = s_model['embeddings.1.weight']

    lhs = F.embedding(x[:, 0], weight0, sparse=False)
    rel = F.embedding(x[:, 1], weight1, sparse=False)
    rhs = F.embedding(x[:, 0], weight0, sparse=False)

    rank = int(lhs.shape[1]/2)
    lhs = lhs[:, :rank], lhs[:, rank:]
    rel = rel[:, :rank], rel[:, rank:]
    rhs = rhs[:, :rank], rhs[:, rank:]

    rhs_scores = None
    rel_scores = None

    if flagrel:
        to_score_rel = weight1
        to_score_rel = to_score_rel[:, :rank], to_score_rel[:, rank:]
        rel_scores = (
            (lhs[0] * rhs[0] + lhs[1] * rhs[1]) @ to_score_rel[0].transpose(0, 1) +
            (lhs[0] * rhs[1] - lhs[1] * rhs[0]
             ) @ to_score_rel[1].transpose(0, 1)
        )
        return rel_scores, cat_factors(lhs, rel, rhs)
    else:
        to_score_entity = weight0
        to_score_entity = to_score_entity[:, :rank], to_score_entity[:, rank:]
        rhs_scores = (
            (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score_entity[0].transpose(0, 1) +
            (lhs[0] * rel[1] + lhs[1] * rel[0]
             ) @ to_score_entity[1].transpose(0, 1)
        )
        return rhs_scores, cat_factors(lhs, rel, rhs)


def s_criterion(args, s_model, input_batch, mask=None, flagrel=False):
    pred, _ = f_forward(s_model, input_batch, mask, flagrel)
    loss = nn.CrossEntropyLoss(reduction='mean', weight=args.weight)
    # predictions_2, factors_2 = s_model.forward(input_batch)
    if flagrel:
        truth = input_batch[:, 1]
    else:
        truth = input_batch[:, 2]
    l = loss(pred, truth)  # + factor_loss(factor)

    return l


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
