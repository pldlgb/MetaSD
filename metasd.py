# torch
import torch
from torch import optim

# utils
import os
import gc
import tqdm
import argparse
from copy import deepcopy as cp
from collections import OrderedDict
import json

# personal package
from datasets import Dataset
from mask import Mask
from otherutils import *
from prunemodel import *


def substitute(args, e, train_examples, t_model, mask):
    # substitute_loss = 0
    input_batch = train_examples.cuda()

    fast_weights = OrderedDict(cp((name, param))
                               for (name, param) in t_model.named_parameters())

    l = d_criterion(args, e, t_model=t_model, s_model=fast_weights,
                    input_batch=input_batch, mask=mask, teacher_grad=True)
    # s_optimizer.zero_grad()
    # l.backward()
    grads = torch.autograd.grad(l, fast_weights.values(),
                                create_graph=True, retain_graph=True)
    # torch.nn.utils.clip_grad_norm_(s_model.parameters(), 1.0)

    fast_weights = OrderedDict(
        (name, param - 1e-1 * grad) for ((name, param), grad) in zip(fast_weights.items(), grads))

    del grads
    return fast_weights, t_model


def quiz(args, held_dataset, t_model, fast_weights, t_optimizer, mask):
    s_prime_loss = None
    held_batch_num = 0
    held_examples = held_dataset[torch.randperm(held_dataset.shape[0]), :]

    b_begin = 0
    while b_begin < held_dataset.shape[0]:
        input_batch = held_examples[b_begin:b_begin + args.batch_size].cuda()
        # torch.autograd.set_detect_anomaly(True)

        s_prime_step_loss = s_criterion(
            args, s_model=fast_weights, input_batch=input_batch, mask=mask, flagrel=False)
        if s_prime_loss is None:
            s_prime_loss = s_prime_step_loss
        else:
            s_prime_loss += s_prime_step_loss

        held_batch_num += 1
        b_begin += args.batch_size
        # bar.update(input_batch.shape[0])

    s_prime_loss /= held_batch_num
    # bar.set_postfix(loss=f'{s_prime_loss.item():.1f}')
    t_grads = torch.autograd.grad(s_prime_loss, t_model.parameters())

    for p, gr in zip(t_model.parameters(), t_grads):
        p.grad = gr
    # torch.nn.utils.clip_grad_norm_(t_model.parameters(), args.max_grad_norm)
    t_optimizer.step()

    # Manual zero_grad
    for p in t_model.parameters():
        p.grad = None

    del t_grads
    del fast_weights

    return t_model


def real(args, e, train_example, t_model, real_optimizer):

    input_batch = train_example.cuda()
    if args.use_relaux:
        t_model = rp_criterion(args, e, t_model, real_optimizer, input_batch)
    else:
        t_model = p_criterion(args, e, t_model, real_optimizer, input_batch)

    return t_model


def metadistil(args, e, train_example, held_dataset, t_model, t_optimizer, real_optimizer, mask=None):
    """ Train the model """
    # t_model.zero_grad()
    #    Step 0: train T model
    # if e<5:
    t_model = T_train(args, train_example, t_model, real_optimizer)
    #    Step 1: Assume S'    #
    fast_weights, t_model = substitute(args, e, train_example, t_model, mask)

    # Step 2: Train T with S' on HELD set  #
    t_model.train()
    t_model = quiz(args, held_dataset, t_model,
                   fast_weights, t_optimizer, mask)
    # Step 3: Actually update S & T           #
    t_model = real(args, e, train_example, t_model, real_optimizer)
    if True:
        gc.collect()
        torch.cuda.empty_cache()

    return t_model


if __name__ == "__main__":
    datasets = ['WN18RR', 'FB237', 'YAGO3-10']

    parser = argparse.ArgumentParser(
        description="Tensor Factorization for Knowledge Graph Completion")

    parser.add_argument('--dataset', choices=datasets,
                        default='WN18RR', help="Dataset in {}".format(datasets))
    parser.add_argument('--model', type=str, default='PruneRelComplEx')
    parser.add_argument('--batch_size', default=1000, type=int,
                        help="Factorization rank.")
    parser.add_argument('--learning_rate', default=1e-1, type=float,
                        help="Learning rate")
    parser.add_argument('--reg', default=1e-1, type=float,
                        help="Regularization weight")
    parser.add_argument('--rank', default=2000, type=int,
                        help="Factorization rank.")
    parser.add_argument('--use_weight', default=False,
                        type=bool, help="dataset weight")
    parser.add_argument('--use_relaux', default=False,
                        type=bool, help="if use rel aux")
    # parser.add_argument('--et', default=50, type=int, help="epoch temp" )
    parser.add_argument('--t', default=3, type=int, help="temp")
    parser.add_argument('--max_epochs', default=300,
                        type=int, help="epoch temp")
    parser.add_argument('-train', '--do_train', action='store_true')
    parser.add_argument('-save', '--do_save', action='store_true')
    parser.add_argument('-sparse_infer', '--do_sparse_inference', action='store_true')
    parser.add_argument('-path', '--save_path', type=str, default='logs/')
    parser.add_argument('-id', '--model_id', type=str, default='0')
    parser.add_argument('-ckpt', '--checkpoint', type=str, default='')
    # contrast with pruned model
    parser.add_argument('--prune', action='store_true',
                        help="if contrasting with pruned model")
    parser.add_argument('--prune_percent', type=float,
                        default=0.9, help="whole prune percentage")
    parser.add_argument('--random_prune_percent', type=float,
                        default=0, help="random prune percentage")
    args = parser.parse_args()
    print(args)
    args.verbose = True

    if args.do_save:
        assert args.save_path
        save_suffix = args.model + '_' + args.dataset + '_' + args.model_id

        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)

        save_path = os.path.join(args.save_path, save_suffix)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        with open(os.path.join(save_path, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)

    setup_seed(2)
    """ get dataset and build model """
    dataset = Dataset("data", args.dataset)
    examples = torch.from_numpy(dataset.get_train().astype('int64'))
    held_examples = torch.from_numpy(dataset.get_valid().astype('int64'))

    if args.do_train:
        t_model = PruneRelComplEx(dataset.get_shape(), 2000, 1e-3).cuda()
        # regularizer = DURA_W(args.reg).cuda()

        t_optimizer = optim.Adagrad(t_model.parameters(), lr=1e-4)
        real_optimizer = optim.Adagrad(t_model.parameters(), lr=args.learning_rate)

        args.weight = None
        if args.use_weight:
            args.weight = torch.Tensor(dataset.get_weight()).cuda()

        sld = int(len(examples)/50)
        print(sld)
        held_dataset = examples[:sld]
        with open(os.path.join(save_path, 'train.log'), 'w') as log_file:
            for e in range(args.max_epochs):
                print("epoch : ", (e))
                actual_examples = examples[torch.randperm(
                    examples.shape[0]), :]
                # Prune Part
                pruneMask = Mask(t_model)
                prunePercent = args.prune_percent
                pruneMask.magnitudePruning(prunePercent)
                ent_mask = t_model.embeddings[0].prune_mask
                rel_mask = t_model.embeddings[1].prune_mask
                # mask=None
                mask = (ent_mask, rel_mask)
                with tqdm(total=actual_examples.shape[0], unit='ex', disable=not args.verbose) as bar:
                    bar.set_description(f'train loss')
                    b_begin = 0
                    while b_begin < actual_examples.shape[0]:
                        input_batch = actual_examples[
                            b_begin:b_begin + args.batch_size
                        ].cuda()
                        t_model = metadistil(
                            args, e, input_batch, held_dataset, t_model, t_optimizer, real_optimizer, mask)
                        b_begin += args.batch_size
                        bar.update(input_batch.shape[0])
                if (e+1) % 5 == 0:
                    print("look look student:")
                    t_model.set_prune_flag(True)
                    valid, test, train = [
                        avg_both(*dataset.eval(t_model, split, -
                                               1 if split != 'train' else 5000))
                        for split in ['valid', 'test', "train"]
                    ]
                    print("\t TRAIN: ", train)
                    print("\t VALID: ", valid)
                    print("\t TEST: ", test)
                    print("look look teacher:")
                    log_file.write("Epoch: {}\n".format(e+1))
                    log_file.write("look look teacher:")
                    log_file.write("\t TRAIN: {}\n".format(train))
                    log_file.write("\t VALID: {}\n".format(valid))
                    log_file.write("\t TEST: {}\n".format(test))
                    log_file.flush()

                    t_model.set_prune_flag(False)
                    valid, test, train = [
                        avg_both(*dataset.eval(t_model, split, -
                                               1 if split != 'train' else 5000))
                        for split in ['valid', 'test', "train"]
                    ]
                    print("\t TRAIN: ", train)
                    print("\t VALID: ", valid)
                    print("\t TEST: ", test)
                    log_file.write("look look student:")
                    log_file.write("\t TRAIN: {}\n".format(train))
                    log_file.write("\t VALID: {}\n".format(valid))
                    log_file.write("\t TEST: {}\n".format(test))
                    log_file.flush()
                    if test["MRR"] > 0.3948:
                        print(test["MRR"])
                        print("model save!")
                        save(t_model, save_path, "checkpoint")

        if args.do_save:
            save(t_model, save_path, "final_checkpoint")
    else:
        with torch.no_grad():
            # eval in dense mode or sparse mode
            if not args.do_sparse_inference:
                t_model = PruneRelComplEx(dataset.get_shape(), 2000, 1e-3).cuda()
                t_model.load_state_dict(torch.load(os.path.join(
                    args.checkpoint, 'final_checkpoint'), map_location='cuda:0'))
                # prune model
                pruneMask = Mask(t_model)
                prunePercent = args.prune_percent
                pruneMask.magnitudePruning(prunePercent, ep=0.)
                t_model.set_prune_flag(True)
            else:
                t_model = SparsePruneRelComplEx(dataset.get_shape(), 2000).cuda()
                coo_params = torch.load(os.path.join(
                    args.checkpoint, 'final_sparse_checkpoint'), map_location='cuda:0')

                # load sparse embedding
                def load(layer, layer_dict):
                    for k in layer_dict:
                        setattr(layer, k, layer_dict[k])
                load(t_model.embeddings[0], coo_params['embedding0'])
                load(t_model.embeddings[1], coo_params['embedding1'])

            # Test on the whole testing set
            test_result = avg_both(*dataset.eval(t_model, 'test', -1))
            print("MRR:", test_result["MRR"])
            print("hits@[1,3,10]:", test_result['hits@[1,3,10]'].tolist())
