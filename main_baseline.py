from __future__ import print_function, absolute_import
import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np
import random

from PIL import Image

# try:
import apex
from apex import amp
# except:
#     pass
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

import data_manager
from lr_schedulers import WarmupMultiStepLR
from video_loader import VideoDataset
import transforms as T
import models
from losses import CrossEntropyLabelSmooth, TripletLoss, TripletLossAttrWeightes, \
    CosineTripletLoss
from utils import AverageMeter, Logger, AttributesMeter, make_optimizer
from eval_metrics import evaluate_withoutrerank
from samplers import RandomIdentitySampler
import pandas as pd
from config import cfg

parser = argparse.ArgumentParser(description="ReID Baseline Training")
parser.add_argument(
    "--config_file", default="./configs/softmax_triplet_tlaw.yml", help="path to config file", type=str
)
parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                    nargs=argparse.REMAINDER)

args_ = parser.parse_args()

if args_.config_file != "":
    cfg.merge_from_file(args_.config_file)
cfg.merge_from_list(args_.opts)

tqdm_enable = False

def main():
    runId = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, runId)
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.mkdir(cfg.OUTPUT_DIR)
    print(cfg.OUTPUT_DIR)
    torch.manual_seed(cfg.RANDOM_SEED)
    random.seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    use_gpu = torch.cuda.is_available() and cfg.MODEL.DEVICE == "cuda"
    if not cfg.EVALUATE_ONLY:
        sys.stdout = Logger(osp.join(cfg.OUTPUT_DIR, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(cfg.OUTPUT_DIR, 'log_test.txt'))

    print("==========\nConfigs:{}\n==========".format(cfg))

    if use_gpu:
        print("Currently using GPU {}".format(cfg.MODEL.DEVICE_ID))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(cfg.RANDOM_SEED)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format(cfg.DATASETS.NAME))

    dataset = data_manager.init_dataset(root=cfg.DATASETS.ROOT_DIR, name=cfg.DATASETS.NAME)
    if cfg.ATTR_RECOG_ON:
        cfg.DATASETS.ATTR_LENS = dataset.attr_lens
    else:
        cfg.DATASETS.ATTR_LENS = []

    cfg.DATASETS.ATTR_COLUMNS = dataset.columns
    print("Initializing model: {}".format(cfg.MODEL.NAME))

    if cfg.MODEL.ARCH == 'video_baseline':
        torch.backends.cudnn.benchmark = False
        model = models.init_model(name=cfg.MODEL.ARCH, num_classes=dataset.num_train_pids, pretrain_choice=cfg.MODEL.PRETRAIN_CHOICE,
                                  last_stride=cfg.MODEL.LAST_STRIDE,
                                  neck=cfg.MODEL.NECK, model_name=cfg.MODEL.NAME, neck_feat=cfg.TEST.NECK_FEAT,
                                  model_path=cfg.MODEL.PRETRAIN_PATH, fusion_method=cfg.MODEL.FUSION_METHOD, attr_lens=cfg.DATASETS.ATTR_LENS, attr_loss=cfg.DATASETS.ATTRIBUTE_LOSS)

    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    transform_train = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
    ])
    transform_test = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # pin_memory = True if use_gpu else False
    pin_memory = False

    cfg.DATALOADER.NUM_WORKERS = 0

    cfg.DATASETS.ATTR_COLUMNS = dataset.columns

    trainloader = DataLoader(
        VideoDataset(dataset.train, seq_len=cfg.DATASETS.SEQ_LEN, sample=cfg.DATASETS.TRAIN_SAMPLE_METHOD, transform=transform_train,
                     attr_loss=cfg.DATASETS.ATTRIBUTE_LOSS, attr_lens=cfg.DATASETS.ATTR_LENS, dataset_name=cfg.DATASETS.NAME),
        sampler=RandomIdentitySampler(dataset.train, num_instances=cfg.DATALOADER.NUM_INSTANCE),
        batch_size=cfg.SOLVER.SEQS_PER_BATCH, num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=pin_memory, drop_last=True
    )

    queryloader = DataLoader(
        VideoDataset(dataset.query, seq_len=cfg.DATASETS.SEQ_LEN, sample=cfg.DATASETS.TEST_SAMPLE_METHOD, transform=transform_test,
                     max_seq_len=cfg.DATASETS.TEST_MAX_SEQ_NUM, dataset_name=cfg.DATASETS.NAME, attr_lens=cfg.DATASETS.ATTR_LENS, attr_loss=cfg.DATASETS.ATTRIBUTE_LOSS),
        batch_size=cfg.TEST.SEQS_PER_BATCH , shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=pin_memory, drop_last=False
    )

    galleryloader = DataLoader(
        VideoDataset(dataset.gallery, seq_len=cfg.DATASETS.SEQ_LEN, sample=cfg.DATASETS.TEST_SAMPLE_METHOD, transform=transform_test,
                     max_seq_len=cfg.DATASETS.TEST_MAX_SEQ_NUM, dataset_name=cfg.DATASETS.NAME, attr_lens=cfg.DATASETS.ATTR_LENS, attr_loss=cfg.DATASETS.ATTRIBUTE_LOSS),
        batch_size=cfg.TEST.SEQS_PER_BATCH , shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=pin_memory, drop_last=False,
    )

    if cfg.MODEL.SYN_BN:

        model = apex.parallel.convert_syncbn_model(model)


    optimizer = make_optimizer(cfg, model)

    if cfg.SOLVER.FP_16:
        model, optimizer = apex.amp.initialize(model.cuda(), optimizer, opt_level='O1')

    if use_gpu:
        model = nn.DataParallel(model)
        model.cuda()

    start_time = time.time()
    xent = CrossEntropyLabelSmooth(num_classes=dataset.num_train_pids)
    if cfg.DISTANCE == "cosine":
        tent = CosineTripletLoss(cfg.SOLVER.MARGIN)
    elif cfg.DISTANCE == "euclid":
        tent = TripletLoss(cfg.SOLVER.MARGIN)



    if cfg.DATASETS.ATTRIBUTE_LOSS == "mce":
        attr_criter = nn.CrossEntropyLoss()
    elif cfg.DATASETS.ATTRIBUTE_LOSS == "bce":
        attr_criter = nn.BCEWithLogitsLoss()

    tlaw = TripletLossAttrWeightes(dis_type=cfg.DISTANCE)


    scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                  cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

    ema = None
    no_rise = 0
    metrics = test(model, queryloader, galleryloader, cfg.TEST.TEMPORAL_POOL_METHOD, use_gpu)
    # return
    best_rank1 = 0
    start_epoch = 0
    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHS):
        # if no_rise == 10:
        #     break
        scheduler.step()
        print("noriase:", no_rise)
        print("==> Epoch {}/{}".format(epoch + 1, cfg.SOLVER.MAX_EPOCHS))
        print("current lr:", scheduler.get_lr()[0])

        train(model, trainloader, xent, tent, attr_criter, optimizer, use_gpu, tlaw=tlaw)
        if cfg.SOLVER.EVAL_PERIOD > 0 and ((epoch + 1) % cfg.SOLVER.EVAL_PERIOD == 0 or (epoch + 1) == cfg.SOLVER.MAX_EPOCHS):
            print("==> Test")

            metrics = test(model, queryloader, galleryloader, cfg.TEST.TEMPORAL_POOL_METHOD, use_gpu)
            rank1 = metrics[0]
            if rank1 > best_rank1:
                best_rank1 = rank1
                no_rise = 0
            else:
                no_rise += 1
                continue

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            torch.save(state_dict, osp.join(cfg.OUTPUT_DIR, "rank1_" + str(rank1) + '_checkpoint_ep' + str(epoch + 1) + '.pth'))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))


def train(model, trainloader, xent, tent, attr_criter, optimizer, use_gpu, tlaw=None):
    model.train()
    xent_losses = AverageMeter()
    tent_losses = AverageMeter()
    losses = AverageMeter()
    attr_losses = AverageMeter()
    tlaw_losses_unrelated = AverageMeter()
    tlaw_losses_related = AverageMeter()

    for batch_idx, (imgs, pids, _, attrs) in enumerate(trainloader):
        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()
            if cfg.DATASETS.ATTRIBUTE_LOSS == "mce":
                attrs = [a.view(-1).cuda() for a in attrs]
            else:
                attrs = [a.cuda() for a in attrs]
        outputs, features, attr_preds = model(imgs)
        # combine hard triplet loss with cross entropy loss
        xent_loss = xent(outputs, pids)
        tent_loss, _, _ = tent(features, pids)
        xent_losses.update(xent_loss.item(), 1)
        tent_losses.update(tent_loss.item(), 1)

        if cfg.ATTR_RECOG_ON:
            attr_loss = attr_criter(attr_preds[0], attrs[0])
            for i in range(1, len(attrs)):
                attr_loss += attr_criter(attr_preds[i], attrs[i])
            if cfg.DATASETS.ATTRIBUTE_LOSS == "mce":
                attr_loss /= len(attr_preds)
            attr_losses.update(attr_loss.item(), 1)
            # loss = xent_loss + tent_loss
            if cfg.DATASETS.ATTRIBUTE_LOSS == "mce":
                unrelated_attrs = torch.cat(attr_preds[:len(cfg.DATASETS.ATTR_LENS[0])], 1)
                related_attrs = torch.cat(attr_preds[len(cfg.DATASETS.ATTR_LENS[0]) : len(cfg.DATASETS.ATTR_LENS[0]) + len(cfg.DATASETS.ATTR_LENS[1])], 1)
            if cfg.DATASETS.ATTRIBUTE_LOSS == "bce":
                unrelated_attrs = attr_preds[0]
                related_attrs = attr_preds[1]
            tlaw_loss_unrelated = tlaw(features, pids, unrelated_attrs)
            tlaw_loss_related = tlaw(features, pids, related_attrs)
            tlaw_losses_unrelated.update(tlaw_loss_unrelated.item(), 1)
            tlaw_losses_related.update(tlaw_loss_related.item(), 1)
            tent_loss = (tent_loss + tlaw_loss_unrelated + tlaw_loss_related) / 3
            loss = xent_loss + tent_loss

            loss += attr_loss

            optimizer.zero_grad()

            if cfg.SOLVER.FP_16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()
            # ema.update()

        else:
            loss = xent_loss + tent_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # ema.update()
        losses.update(loss.item(), 1)


        # attr_losses.update(attr_loss.item(), pids.size(0))
    print("Batch {}/{}\t Loss {:.6f} ({:.6f}), attr Loss {:.6f} ({:.6f}), related attr triplet Loss {:.6f} ({:.6f}), unrelated attr triplet Loss {:.6f} ({:.6f}),  xent Loss {:.6f} ({:.6f}), tent Loss {:.6f} ({:.6f})".format(
        batch_idx + 1, len(trainloader), losses.val, losses.avg, attr_losses.val, attr_losses.avg,
        tlaw_losses_unrelated.val, tlaw_losses_unrelated.avg, tlaw_losses_related.val, tlaw_losses_related.avg, xent_losses.val, xent_losses.avg,
        tent_losses.val, tent_losses.avg))

    return losses.avg


def test(model, queryloader, galleryloader, pool, use_gpu, ranks=[1, 5, 10, 20]):
    temp_recog_on = False
    if cfg.DATASETS.ATTRIBUTE_LOSS == "bce":
        temp_recog_on = cfg.ATTR_RECOG_ON
        cfg.ATTR_RECOG_ON = False

    if cfg.ATTR_RECOG_ON:
        attr_metrics = AttributesMeter(len(cfg.DATASETS.ATTR_LENS[0]) + len(cfg.DATASETS.ATTR_LENS[1]))
    with torch.no_grad():
        model.eval()
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids, attrs, img_path) in enumerate(tqdm(queryloader)):

            if use_gpu:
                imgs = imgs.cuda()
                attrs = [a.view(-1) for a in attrs]
            b, n, s, c, h, w = imgs.size()
            assert (b == 1)
            imgs = imgs.view(b * n, s, c, h, w)
            features, outputs = model(imgs)
            q_pids.extend(pids)
            q_camids.extend(camids)

            features = features.view(n, -1)

            features = torch.mean(features, 0)
            qf.append(features)
            if cfg.ATTR_RECOG_ON:
                outputs = [torch.mean(out, 0).view(1, -1) for out in outputs]
                preds = []
                gts = []
                acces = np.array([0 for _ in range(len(cfg.DATASETS.ATTR_LENS[0]) + len(cfg.DATASETS.ATTR_LENS[1]))])
                for i in range(len(outputs)):
                    outs = outputs[i].cpu().numpy()
                    # outs = torch.mean(outs, 0)
                    if cfg.DATASETS.ATTRIBUTE_LOSS == "ce":
                        preds.append(np.argmax(outs, 1)[0])
                        gts.append(attrs[i].cpu().numpy()[0])
                        acces[i] += np.sum(np.argmax(outs, 1) == attrs[i].numpy())
                attr_metrics.update(preds, gts, acces, 1)
            del imgs
            del outputs
            del features
        qf = torch.stack(qf)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)


        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        gallery_pathes = []
        for batch_idx, (imgs, pids, camids, attrs, img_path) in enumerate(tqdm(galleryloader)):
            # if batch_idx > 10:
            #     break
            gallery_pathes.append(img_path[0])
            if use_gpu:
                imgs = imgs.cuda()
                attrs = [a.view(-1) for a in attrs]
            b, n, s, c, h, w = imgs.size()
            imgs = imgs.view(b * n, s, c, h, w)
            assert (b == 1)
            features, outputs = model(imgs)

            features = features.view(n, -1)
            if pool == 'avg':
                features = torch.mean(features, 0)
            else:
                features, _ = torch.max(features, 0)
            g_pids.extend(pids)
            g_camids.extend(camids)
            gf.append(features)
            if cfg.ATTR_RECOG_ON and len(attrs) != 0:
                outputs = [torch.mean(out, 0).view(1, -1) for out in outputs]
                preds = []
                gts = []
                acces = np.array([0 for _ in range(len(cfg.DATASETS.ATTR_LENS[0]) + len(cfg.DATASETS.ATTR_LENS[1]))])
                for i in range(len(outputs)):
                    outs = outputs[i].cpu().numpy()
                    # outs = torch.mean(outs, 0)
                    if cfg.DATASETS.ATTRIBUTE_LOSS == "ce":
                        preds.append(np.argmax(outs, 1)[0])
                        gts.append(attrs[i].cpu().numpy()[0])
                        acces[i] += np.sum(np.argmax(outs, 1) == attrs[i].numpy())
                attr_metrics.update(preds, gts, acces, 1)
            del imgs
            del outputs
            del features

        gf = torch.stack(gf)
        # gf = torch.stack(gf)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)


        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
        print("Computing distance matrix")

        if cfg.DATASETS.NAME == "duke":
            print("gallary with query result:")
            gf = torch.cat([gf, qf], 0)
            g_pids = np.concatenate([g_pids, q_pids], 0)
            g_camids = np.concatenate([g_camids, q_camids], 0)
            metrics = evaluate_withoutrerank(qf, q_pids, q_camids, gf, g_pids, g_camids, ranks, dis_type=cfg.DISTANCE)
        else:
            metrics = evaluate_withoutrerank(qf, q_pids, q_camids, gf, g_pids, g_camids, ranks, dis_type=cfg.DISTANCE)
        if cfg.ATTR_RECOG_ON:

            print("Attributes:")
            print("single performance:")
            f1_score_macros, acces_avg, acc_mean_all, f1_mean_all = attr_metrics.get_f1_and_acc()
            colum_str = "|".join(["%15s" % c for c in cfg.DATASETS.ATTR_COLUMNS])
            acc_str = "|".join(["%15f" % acc for acc in acces_avg])
            f1_scores_macros_str = "|".join(["%15f" % f for f in f1_score_macros])
            print(colum_str)
            print(acc_str)
            print(f1_scores_macros_str)

            mean_columns = ["mean_all", "mean_related", "mean_no_quality"]
            mean_acces = [acc_mean_all, np.mean(acces_avg[range(len(cfg.DATASETS.ATTR_LENS[0]), len(cfg.DATASETS.ATTR_LENS[0]) + len(cfg.DATASETS.ATTR_LENS[1]))]), \
                          np.mean(acces_avg[[0, 1] + list(range(len(cfg.DATASETS.ATTR_LENS[0]), len(cfg.DATASETS.ATTR_LENS[0]) + len(cfg.DATASETS.ATTR_LENS[1])))])]
            mean_f1s = [f1_mean_all, np.mean(f1_score_macros[range(len(cfg.DATASETS.ATTR_LENS[0]), len(cfg.DATASETS.ATTR_LENS[0]) + len(cfg.DATASETS.ATTR_LENS[1]))]), \
                          np.mean(f1_score_macros[[0, 1] + list(range(len(cfg.DATASETS.ATTR_LENS[0]), len(cfg.DATASETS.ATTR_LENS[0]) + len(cfg.DATASETS.ATTR_LENS[1])))])]

            colum_str = "|".join(["%15s" % c for c in mean_columns])
            acc_str = "|".join(["%15f" % acc for acc in mean_acces])
            f1_scores_macros_str = "|".join(["%15f" % f for f in mean_f1s])
            print(colum_str)
            print(acc_str)
            print(f1_scores_macros_str)

        cfg.ATTR_RECOG_ON = temp_recog_on
        return metrics


if __name__ == '__main__':

    main()




