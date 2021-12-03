#!/usr/bin/env python
# coding: utf-8

import gc
import itertools
import json
import math
import os
import warnings
from utils import *
import functools
from tqdm import tqdm
import re
from pprint import pprint
from transformers import AutoModel, BertTokenizerFast
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.optim as optim
import glob
import time
import logging
from common.utils import Preprocessor, DefaultLogger
from tplinker import (HandshakingTaggingScheme, DataMaker4Bert,
                      DataMaker4BiLSTM, TPLinkerBert, TPLinkerBiLSTM,
                      MetricsCalculator)
# import wandb
import config
from glove import Glove
import random
import numpy as np
import os
import time, json
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, random_split, Subset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import f1_score
from collections import Counter
import sys
import pickle
import collections

# Global setting start
# Argument parse
test_config = config.eval_config
config = config.train_config
# save name tmp file
with open("_run_id_tmp", "w") as f:
    f.write(config["path_to_save_model"])
# Setting
sys.setrecursionlimit(10000)

hyper_parameters = config["hyper_parameters"]
# Gpu set
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = str(config["device_num"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# for reproductivity
np.random.seed(hyper_parameters["seed"])
torch.manual_seed(hyper_parameters["seed"])  # pytorch random seed
torch.backends.cudnn.deterministic = True

# params
use_two_model = config["use_two_model"]
data_home = config["data_home"]
experiment_name = config["exp_name"]
train_data_path = os.path.join(data_home, experiment_name,
                               config["train_data"])
valid_data_path = os.path.join(data_home, experiment_name,
                               config["valid_data"])
rel2id_path = os.path.join(data_home, experiment_name, config["rel2id"])

# For test
hyper_parameters_test = test_config["hyper_parameters"]
data_home = test_config["data_home"]
experiment_name = test_config["exp_name"]
test_data_path = os.path.join(data_home, experiment_name, test_config["test_data"])
rel2id_path = os.path.join(data_home, experiment_name, test_config["rel2id"])
save_res_dir = os.path.join(test_config["save_res_dir"], experiment_name)
max_test_seq_len = hyper_parameters_test["max_test_seq_len"]
sliding_len = hyper_parameters_test["sliding_len"]
force_split = hyper_parameters_test["force_split"]
RS_logger = ResultRestore(os.path.join(config["log_path"], "results.log"), 0, True)
RS_logger.add_file2pool(os.path.join(config["log_path"], "results.json"))
# save path and logger
if config["logger"] == "wandb":
    # init wandb
    wandb.init(
        project=experiment_name,
        name=config["run_name"],
        config=hyper_parameters  # Initialize config
    )

    wandb.config.note = config["note"]

    model_state_dict_dir = wandb.run.dir
    logger = wandb
else:
    model_state_dict_dir = config["path_to_save_model"]
    if not os.path.exists(model_state_dict_dir):
        os.makedirs(model_state_dict_dir)
    logger = DefaultLogger(os.path.join(config["log_path"], "train_log.log"), experiment_name,
                           config["run_name"], config["run_id"],
                           config)


# Global setting end
# In[ ]:

def teacher_training_process(model, FIRST_EPOCHS, optimizer, loss_func, scheduler, train_dataloader, valid_dataloader):
    for ep in range(FIRST_EPOCHS):
        ## train
        model.train()
        t_ep = time.time()
        start_lr = optimizer.param_groups[0]['lr']
        total_loss, total_ent_sample_acc, total_head_rel_sample_acc, total_tail_rel_sample_acc = 0., 0., 0., 0.
        for batch_ind, batch_train_data in enumerate(train_dataloader):
            t_batch = time.time()
            z = (2 * len(rel2id) + 1)
            steps_per_ep = len(train_dataloader)
            total_steps = hyper_parameters["loss_weight_recover_steps"] + 1  # + 1 avoid division by zero error
            current_step = steps_per_ep * ep + batch_ind
            w_ent = max(1 / z + 1 - current_step / total_steps, 1 / z)
            w_rel = min((len(rel2id) / z) * current_step / total_steps,
                        (len(rel2id) / z))
            loss_weights = {"ent": w_ent, "rel": w_rel}

            if config["encoder"] == "BERT":
                sample_list, batch_input_ids, batch_attention_mask, batch_token_type_ids, tok2char_span_list, batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = batch_train_data
                batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = (
                    batch_input_ids.to(device), batch_attention_mask.to(device),
                    batch_token_type_ids.to(device), batch_ent_shaking_tag.to(device),
                    batch_head_rel_shaking_tag.to(device),
                    batch_tail_rel_shaking_tag.to(device))

            elif config["encoder"] in {
                "BiLSTM",
            }:
                sample_list, batch_input_ids, tok2char_span_list, batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = batch_train_data

                batch_input_ids, batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = (
                    batch_input_ids.to(device), batch_ent_shaking_tag.to(device),
                    batch_head_rel_shaking_tag.to(device),
                    batch_tail_rel_shaking_tag.to(device))
            ## concat pseudo training data

            # model forward
            # zero the parameter gradients
            optimizer.zero_grad()
            if config["encoder"] == "BERT":
                ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs = model(
                    batch_input_ids,
                    batch_attention_mask,
                    batch_token_type_ids,
                )
            elif config["encoder"] in {
                "BiLSTM",
            }:
                ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs = model(
                    batch_input_ids)

            w_ent, w_rel = loss_weights["ent"], loss_weights["rel"]
            loss = w_ent * loss_func(
                ent_shaking_outputs, batch_ent_shaking_tag) + w_rel * loss_func(
                head_rel_shaking_outputs,
                batch_head_rel_shaking_tag) + w_rel * loss_func(
                tail_rel_shaking_outputs, batch_tail_rel_shaking_tag)

            loss.backward()
            optimizer.step()

            ent_sample_acc = metrics.get_sample_accuracy(ent_shaking_outputs,
                                                         batch_ent_shaking_tag)
            head_rel_sample_acc = metrics.get_sample_accuracy(
                head_rel_shaking_outputs, batch_head_rel_shaking_tag)
            tail_rel_sample_acc = metrics.get_sample_accuracy(
                tail_rel_shaking_outputs, batch_tail_rel_shaking_tag)

            loss, ent_sample_acc, head_rel_sample_acc, tail_rel_sample_acc = loss.item(), ent_sample_acc.item(), head_rel_sample_acc.item(
            ), tail_rel_sample_acc.item()
            scheduler.step()

            total_loss += loss
            total_ent_sample_acc += ent_sample_acc
            total_head_rel_sample_acc += head_rel_sample_acc
            total_tail_rel_sample_acc += tail_rel_sample_acc

            avg_loss = total_loss / (batch_ind + 1)
            avg_ent_sample_acc = total_ent_sample_acc / (batch_ind + 1)
            avg_head_rel_sample_acc = total_head_rel_sample_acc / (batch_ind +
                                                                   1)
            avg_tail_rel_sample_acc = total_tail_rel_sample_acc / (batch_ind +
                                                                   1)

            batch_print_format = "\rproject: {}, run_name: {}, Epoch: {}/{}, batch: {}/{}, train_loss: {}, " + "t_ent_sample_acc: {}, t_head_rel_sample_acc: {}, t_tail_rel_sample_acc: {}," + "lr: {}, batch_time: {}, total_time: {} -------------"

            print(batch_print_format.format(
                experiment_name,
                config["run_name"],
                ep + 1,
                FIRST_EPOCHS,
                batch_ind + 1,
                len(train_dataloader),
                avg_loss,
                avg_ent_sample_acc,
                avg_head_rel_sample_acc,
                avg_tail_rel_sample_acc,
                optimizer.param_groups[0]['lr'],
                time.time() - t_batch,
                time.time() - t_ep,
            ),
                end="")

        if config["logger"] == "wandb" and batch_ind % hyper_parameters[
            "log_interval"] == 0:
            logger.log({
                "train_loss": avg_loss,
                "train_ent_seq_acc": avg_ent_sample_acc,
                "train_head_rel_acc": avg_head_rel_sample_acc,
                "train_tail_rel_acc": avg_tail_rel_sample_acc,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "time": time.time() - t_ep,
            })

        if config["logger"] != "wandb":  # only log once for training if logger is not wandb
            logger.log({
                "train_loss": avg_loss,
                "train_ent_seq_acc": avg_ent_sample_acc,
                "train_head_rel_acc": avg_head_rel_sample_acc,
                "train_tail_rel_acc": avg_tail_rel_sample_acc,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "time": time.time() - t_ep,
            })

    ## valid
    model.eval()
    t_ep = time.time()
    total_ent_sample_acc, total_head_rel_sample_acc, total_tail_rel_sample_acc = 0., 0., 0.
    total_rel_correct_num, total_rel_pred_num, total_rel_gold_num = 0, 0, 0
    for batch_ind, batch_valid_data in enumerate(
            tqdm(valid_dataloader, desc="Validating", disable=config["disable_tqdm"])):
        if config["encoder"] == "BERT":
            sample_list, batch_input_ids, batch_attention_mask, batch_token_type_ids, tok2char_span_list, batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = batch_valid_data

            batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = (
                batch_input_ids.to(device), batch_attention_mask.to(device),
                batch_token_type_ids.to(device), batch_ent_shaking_tag.to(device),
                batch_head_rel_shaking_tag.to(device),
                batch_tail_rel_shaking_tag.to(device))

        elif config["encoder"] in {
            "BiLSTM",
        }:
            sample_list, batch_input_ids, tok2char_span_list, batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = batch_valid_data

            batch_input_ids, batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = (
                batch_input_ids.to(device), batch_ent_shaking_tag.to(device),
                batch_head_rel_shaking_tag.to(device),
                batch_tail_rel_shaking_tag.to(device))

        with torch.no_grad():
            if config["encoder"] == "BERT":
                ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs = model(
                    batch_input_ids,
                    batch_attention_mask,
                    batch_token_type_ids,
                )
            elif config["encoder"] in {
                "BiLSTM",
            }:
                ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs = model(
                    batch_input_ids)

        ent_sample_acc = metrics.get_sample_accuracy(ent_shaking_outputs,
                                                     batch_ent_shaking_tag)
        head_rel_sample_acc = metrics.get_sample_accuracy(
            head_rel_shaking_outputs, batch_head_rel_shaking_tag)
        tail_rel_sample_acc = metrics.get_sample_accuracy(
            tail_rel_shaking_outputs, batch_tail_rel_shaking_tag)

        ent_sample_acc, head_rel_sample_acc, tail_rel_sample_acc = ent_sample_acc.item(), head_rel_sample_acc.item(
        ), tail_rel_sample_acc.item()

        total_ent_sample_acc += ent_sample_acc
        total_head_rel_sample_acc += head_rel_sample_acc
        total_tail_rel_sample_acc += tail_rel_sample_acc

    avg_ent_sample_acc = total_ent_sample_acc / len(valid_dataloader)
    avg_head_rel_sample_acc = total_head_rel_sample_acc / len(valid_dataloader)
    avg_tail_rel_sample_acc = total_tail_rel_sample_acc / len(valid_dataloader)

    log_dict = {
        "val_ent_seq_acc": avg_ent_sample_acc,
        "val_head_rel_acc": avg_head_rel_sample_acc,
        "val_tail_rel_acc": avg_tail_rel_sample_acc,
        "time": time.time() - t_ep,
    }
    logger.log(log_dict)
    RS_logger.ts_log().add_scalars("Teacher Valid", {
        "e_acc": avg_ent_sample_acc,
        "h_acc": avg_head_rel_sample_acc,
        "t_acc": avg_tail_rel_sample_acc,
    }, RS_logger.get_cur_ep())
    # RS_logger.epoch_log("teacher model", str((avg_ent_sample_acc, avg_head_rel_sample_acc, avg_tail_rel_sample_acc)))
    return (avg_ent_sample_acc, avg_head_rel_sample_acc, avg_tail_rel_sample_acc)


# def bs_right(nums,target):
#     # nums descending order
#     left = 0
#     right = len(nums)
#     while (left < right):
#         mid = left + (right - left) // 2
#         if nums[mid] > target:
#             left = mid + 1
#         elif nums[mid] < target:
#             right = mid
#         else:
#             left = mid + 1
#     return left

def sift(li, low, high):  # 小根堆向下调整
    # 现在每个元素[-1]是得分，[-2]是acc
    '''
li:列表
low:堆的根节点位置
high:堆的最后一个元素位置
    '''
    i = low  # 最开始指向根节点
    j = 2 * i + 1  # 左孩子
    tmp = li[low]  # 把堆顶存起来，注意tmp的第一个元素才是分数
    while j <= high:  # 只要j位置有数
        if j + 1 <= high and li[j + 1][-1] < li[j][-1]:  # 右孩有且比较大
            j += 1  # 指向右孩
        if li[j][-1] < tmp[-1]:
            li[i] = li[j]
            i = j  # 往下看一层
            j = 2 * i + 1
        else:
            break
    li[i] = tmp


def get_information(index, batch_valid_data, batch_pred_ent_shaking_tag, batch_pred_head_rel_shaking_tag,
                    batch_pred_tail_rel_shaking_tag, model_output):
    '''
    需要的信息
    return:list[sample_list, batch_input_ids, batch_attention_mask,
    batch_token_type_ids, tok2char_span_list,matrix_spots_list,
        (handshaking_tagger.get_sharing_spots_fr_shaking_tag(batch_pred_ent_shaking_tag[i]),
        handshaking_tagger.get_spots_fr_shaking_tag(batch_pred_head_rel_shaking_tag[i]),
        handshaking_tagger.get_spots_fr_shaking_tag(batch_pred_tail_rel_shaking_tag[i])
            )
        ]
    '''
    info = []
    for var in batch_valid_data:
        if isinstance(var, torch.Tensor):
            info.append(var[index].to("cpu"))
        elif isinstance(var, list):
            info.append(var[index])
        else:
            raise Exception
    true_labels = info[-3:]
    info = info[:-3]
    acc = []

    info.append((handshaking_tagger.get_sharing_spots_fr_shaking_tag(batch_pred_ent_shaking_tag[index]),
                 handshaking_tagger.get_spots_fr_shaking_tag(batch_pred_head_rel_shaking_tag[index]),
                 handshaking_tagger.get_spots_fr_shaking_tag(batch_pred_tail_rel_shaking_tag[index])))
    for i in range(3):
        acc.append(
            metrics.get_sample_accuracy(model_output[i][index].unsqueeze(0), true_labels[i].unsqueeze(0)).item())
        # 第一个参数为[3, 24, 5050, 3] 第二个为[3, 24, 5050]
    info.append(acc)

    return info


def generate_pseudo(modelf1, seq_val_acc, unlabeled_dataloader_all, STRATEGY=1):
    TEST = 0
    if STRATEGY == 1:
        # Z_RATIO = 3.074*(sum(seq_val_acc)**0.162)-2.699
        # Z_RATIO=2.274 * (sum(seq_val_acc)** 0.162) - 2.159
        # Z_RATIO = 2.574 * (sum(seq_val_acc) ** 0.162) - 2.349
        # # if sum(seq_val_acc)>1.2:
        # #     Z_RATIO-=0.2
        # # if sum(seq_val_acc) > 2.1:
        # #     Z_RATIO += random.gauss(0, 0.15)
        # Z_RATIO = max(0.15, Z_RATIO)
        Z_RATIO = config["strategy_hyper_parameters"]["Z_RATIO"]
        # ************
        k = int(len(unlabeled_dataloader_all) * Z_RATIO * BATCH_SIZE)
        print(f'k为{k}')
        topk_list = list(torch.zeros((k, 2)).to('cpu'))
        # topk_list = [(-1, -float("inf"))]
        # 初始化堆，每个元素都是[0,0]方便调用
        # ************
        # enh_rate =  seq_val_acc[0]  # 系数提升几倍
        # enh_rate = max(0.3, enh_rate)
        # relh_rate1 = seq_val_acc[1]
        # relh_rate2 = seq_val_acc[2]  # From valid step
        # relh_rate1 = max(0.3, relh_rate1)
        # relh_rate2 = max(0.3, relh_rate2)
        enh_rate = 1  # 系数提升几倍
        relh_rate1 = 1
        relh_rate2 = 1 # From valid step

    else:
        enh_rate = 1
        relh_rate1 = 1
        relh_rate2 = 1
        Z_RATIO = config["strategy_hyper_parameters"]["Z_RATIO"]
        batch_new_data_list = []

    # print(f"generate pseudo label,Z_RATIO: {Z_RATIO}, NUMBER: {int(Z_RATIO*BATCH_SIZE)} \n")

    logger.log("generate pseudo label,Z_RATIO: {}, NUMBER: {} \n".format(Z_RATIO,
                                                                         int(Z_RATIO * len(
                                                                             unlabeled_dataloader_all) * BATCH_SIZE)))
    # RS_logger.epoch_log("Z_ratio", str(Z_RATIO))
    RS_logger.add_json("Z_ratio", Z_RATIO)
    RS_logger.add_json("total use pseudo num", int(Z_RATIO * len(unlabeled_dataloader_all) * BATCH_SIZE))
    RS_logger.ts_log().add_scalar(f"{RS_logger.get_self_text()} Pseudo/Z_RATIO", Z_RATIO, RS_logger.get_cur_ep())
    ## valid
    modelf1.eval()

    pseudo_count = 0
    batch_new_data_list = []
    batch_acc = {'ent': [], 'head_rel': [], 'tail_rel': []}
    topk_time, batch_time = 0, 0

    t_ep = time.time()
    for batch_ind, batch_valid_data in enumerate(
            tqdm(unlabeled_dataloader_all, desc="Validating", disable=config["disable_tqdm"])):
        # bs = time.time()
        if config["encoder"] == "BERT":
            sample_list, batch_input_ids, batch_attention_mask, batch_token_type_ids, tok2char_span_list, matrix_spots_list, batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = batch_valid_data
            # 从batch_valid_data中取数据
            batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = (
                batch_input_ids.to(device), batch_attention_mask.to(device),
                batch_token_type_ids.to(device), batch_ent_shaking_tag.to(device),
                batch_head_rel_shaking_tag.to(device),
                batch_tail_rel_shaking_tag.to(device))
            # 把需要的数据放到gpu上
        elif config["encoder"] in {
            "BiLSTM",
        }:
            sample_list, batch_input_ids, tok2char_span_list, matrix_spots_list, batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = batch_valid_data
            batch_input_ids, batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = (
                batch_input_ids.to(device), batch_ent_shaking_tag.to(device),
                batch_head_rel_shaking_tag.to(device),
                batch_tail_rel_shaking_tag.to(device))

        with torch.no_grad():
            if config["encoder"] == "BERT":
                ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs = modelf1(
                    batch_input_ids,
                    batch_attention_mask,
                    batch_token_type_ids,
                )
                # 获取模型输出
            elif config["encoder"] in {
                "BiLSTM",
            }:
                ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs = modelf1(
                    batch_input_ids)

            # 得到hand shaking结果
            ## 反思，为啥不用循环呢？

            model_output = [ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs]
            # 模型输出打包
            sequence_weights = []
            if STRATEGY == 1:
                # 评分函数2
                # entity
                # ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs

                sequence_weights_without_weight = []
                pred_weight, batch_pred_ent_shaking_tag = torch.max(ent_shaking_outputs, dim=-1)
                # [24, 5050]
                sequence_weight_without_weight = torch.mean(pred_weight, dim=-1)
                # pred_weight = pred_weight.sigmoid()
                sequence_weight = torch.mean(pred_weight, dim=-1) * enh_rate
                # [24]
                sequence_weights.append(sequence_weight)
                sequence_weights_without_weight.append(sequence_weight_without_weight)

                # head rel
                pred_weight, batch_pred_head_rel_shaking_tag = torch.max(head_rel_shaking_outputs, dim=-1)
                # [24, 24, 5050]
                pred_weight_without_weight = torch.mean(pred_weight, dim=-1)
                # [24,24]
                sequence_weight_without_weight = torch.mean(pred_weight_without_weight, dim=-1)
                # [24]
                sequence_weights_without_weight.append(sequence_weight_without_weight)
                # strategy
                # pred_weight = pred_weight.sigmoid()
                pred_weight = torch.mean(pred_weight, dim=-1)
                pred_weight, _ = torch.topk(pred_weight, 5, dim=1, largest=True, sorted=False, out=None)
                # [24,5]
                sequence_weight = torch.mean(pred_weight, dim=-1) * relh_rate1
                # [24]
                sequence_weights.append(sequence_weight)

                # tail rel
                pred_weight, batch_pred_tail_rel_shaking_tag = torch.max(tail_rel_shaking_outputs, dim=-1)
                # [24, 24, 5050]
                pred_weight_without_weight = torch.mean(pred_weight, dim=-1)
                # [24,24]
                sequence_weight_without_weight = torch.mean(pred_weight_without_weight, dim=-1)
                # [24]
                sequence_weights_without_weight.append(sequence_weight_without_weight)
                # strategy
                # pred_weight = pred_weight.sigmoid()
                pred_weight = torch.mean(pred_weight, dim=-1)
                pred_weight, _ = torch.topk(pred_weight, 5, dim=1, largest=True, sorted=False, out=None)
                # [24,5]
                sequence_weight = torch.mean(pred_weight, dim=-1) * relh_rate1
                # [24]
                sequence_weights.append(sequence_weight)
                final_sequence_weight = sum(sequence_weights)
                final_sequence_weight_without_weight = sum(sequence_weights_without_weight)
                # [24]
                topk_start = time.time()
                # ***比较分数
                # 能加速吗

                # update strategy score
                for i in range(len(model_output)):
                    model_output[i] = model_output[i].to('cpu')

                final_sequence_weight = final_sequence_weight.to('cpu')
                batch_pred_ent = batch_pred_ent_shaking_tag.to('cpu')
                batch_pred_head = batch_pred_head_rel_shaking_tag.to('cpu')
                batch_pred_tail = batch_pred_tail_rel_shaking_tag.to('cpu')

                # print(f'此时堆顶得分为{topk_list[0][0]}',end=' ** ')
                num = 0
                for i in range(len(final_sequence_weight)):
                    if final_sequence_weight[i] > topk_list[0][-1]:
                        info = get_information(i, batch_valid_data, batch_pred_ent, batch_pred_head, batch_pred_tail,
                                               model_output)
                        info.append(final_sequence_weight[i])
                        topk_list[0] = info
                        # 获取第i个样本作为数据的信息，但是第一个元素是得分,其后才是信息
                        sift(topk_list, 0, k - 1)  # 小根堆向下调整（需要修改sift）
                        num += 1
                # info.append(final_sequence_weight[i])
                topk_end = time.time()
                # 每轮topk用时
                topk_time += topk_end - topk_start

                # if num!=0:
                #     print(f'第{batch_ind}批加入{num}个pseuod label，耗时{round(be-bs,7)}秒')
                # print(f'本轮耗时{round(be-bs,7)}秒')

                # update no strategy score
                # *************************************************
                'BATCH METHOD!!!!'
                # *************************************************
                batch_start = time.time()
                _, inter_index = torch.topk(final_sequence_weight_without_weight,
                                            int(len(final_sequence_weight_without_weight) * Z_RATIO) if int(
                                                len(final_sequence_weight_without_weight) * Z_RATIO) > 0 else 1, dim=-1,
                                            largest=True, sorted=False, out=None)
                # [ 8, 17, 15]
                true_labels = []
                for var in [batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag]:
                    true_labels.append(var[inter_index])
                model_output = [var[inter_index].to(device) for var in model_output]
                # 此操作耗时，得到序号inter_index再解析
                for i, key in enumerate(batch_acc.keys()):
                    batch_acc[key].append(metrics.get_sample_accuracy(model_output[i], true_labels[i]).item())
                batch_end = time.time()
                batch_time += batch_end - batch_start
            else:
                batch_pred_ent_shaking_tag = torch.argmax(ent_shaking_outputs, dim=-1)
                batch_pred_head_rel_shaking_tag = torch.argmax(head_rel_shaking_outputs, dim=-1)
                batch_pred_tail_rel_shaking_tag = torch.argmax(tail_rel_shaking_outputs, dim=-1)
                for shaking_outputs in model_output:  # 16,2020,2  16,2020 , 0:22,1:5  0:5,1:23
                    pred_weight, label = torch.max(shaking_outputs, dim=-1)
                    if len(pred_weight.shape) == 2:  # entity seq
                        pred_weight = pred_weight * enh_rate
                        sequence_weight = torch.mean(pred_weight, dim=-1)
                    else:  # rel seq# 16,171,2020,3
                        sequence_weight = torch.mean(pred_weight, dim=1)  #
                        # entity\head\rel的分数综合考虑
                        sequence_weight = torch.mean(sequence_weight, dim=-1) * (relh_rate1 + relh_rate2) / 2
                    sequence_weights.append(sequence_weight)

                batch_start = time.time()
                final_sequence_weight = sum(sequence_weights)
                final_sort = torch.argsort(final_sequence_weight, descending=True)
                final_sort = final_sort[:int(len(final_sort) * Z_RATIO) if int(len(final_sort) * Z_RATIO) > 0 else 1]
                inter_index = final_sort
                sort_input = []
                for var in batch_valid_data:
                    if isinstance(var, torch.Tensor):
                        sort_input.append(var[inter_index].to("cpu"))
                    elif isinstance(var, list):
                        sort_input.append([var[i] for i in inter_index])
                    else:
                        raise Exception
                model_output = [var[inter_index].to("cpu") for var in model_output]
                true_labels = sort_input[-3:]
                pseudo_count += len(inter_index)
                # 此操作耗时，得到序号inter_index再解析
                sorted_spots_list = []
                # t_dec_start=time.time()
                for i in inter_index:
                    sorted_spots_list.append((
                        handshaking_tagger.get_sharing_spots_fr_shaking_tag(batch_pred_ent_shaking_tag[i]),
                        handshaking_tagger.get_spots_fr_shaking_tag(batch_pred_head_rel_shaking_tag[i]),
                        handshaking_tagger.get_spots_fr_shaking_tag(batch_pred_tail_rel_shaking_tag[i])))
                    # sorted_spots_list.append(matrix_spots_list[i])
                # print(f"pseudo decoder cost:{time.time()-t_dec_start}")
                for i, key in enumerate(batch_acc.keys()):
                    batch_acc[key].append(metrics.get_sample_accuracy(model_output[i], true_labels[i]).item())
                batch_new_data = batch2dataset(
                    *(sort_input[:-3] + [sorted_spots_list]))  # sorted_spots_list is pseudo label
                batch_new_data_list.extend(batch_new_data)
                batch_end = time.time()
                batch_time += batch_end - batch_start

    if STRATEGY == 1:
        topk_acc = {'ent': [], 'head_rel': [], 'tail_rel': []}
        # batch_acc  # 因为batch_acc是每个batch选中的样本一起算的，所以总共只有batch个元素

        for i in range(k):
            topk_list[i].pop(-1)  # 弹出分数
            _acc = topk_list[i].pop(-1)  # 弹出acc
            for index, key in enumerate(topk_acc.keys()):
                topk_acc[key].append(_acc[index])
            topk_list[i] = tuple(topk_list[i])

        t_ep_end = time.time()
        log_dict = {
            "use pseudo number": k,
            "acc of topk method": {'ent': np.mean(topk_acc['ent']),
                                   'head_rel': np.mean(topk_acc['head_rel']),
                                   'tail_rel': np.mean(topk_acc['tail_rel']),
                                   'overall': np.mean(list(itertools.chain(*topk_acc.values()))),},
            "acc of batch method": {'ent': np.mean(batch_acc['ent']),
                                    'head_rel': np.mean(batch_acc['head_rel']),
                                    'tail_rel': np.mean(batch_acc['tail_rel']),
                                    'overall': np.mean(list(itertools.chain(*batch_acc.values()))), },
            "time cost of selecting": {'topk': round(topk_time, 2), 'batch_method': round(batch_time, 2)},
            "total time": round(t_ep_end - t_ep, 2)}

        RS_logger.ts_log().add_scalars(f"{RS_logger.get_self_text()}/pseudo strategy", log_dict["acc of topk method"],
                                       RS_logger.get_cur_ep())
        RS_logger.ts_log().add_scalars(f"{RS_logger.get_self_text()}/pseudo no strategy",
                                       log_dict["acc of batch method"],
                                       RS_logger.get_cur_ep())
        RS_logger.add_json("pseudo overall", log_dict)
        print(f'总耗时{log_dict["total time"]}秒')
        print(
            f"topk+weight耗时{log_dict['time cost of selecting']['topk']}秒,准确率 ent:{log_dict['acc of topk method']['ent']},head_rel:{log_dict['acc of topk method']['head_rel']},tail_rel:{log_dict['acc of topk method']['tail_rel']}")
        print(
            f"batch_method+without_weight耗时{log_dict['time cost of selecting']['batch_method']}秒,准确率 ent:{log_dict['acc of batch method']['ent']},head_rel:{log_dict['acc of batch method']['head_rel']},tail_rel:{log_dict['acc of batch method']['tail_rel']}")

        return topk_list
    else:
        store_dict = {'ent': np.mean(batch_acc['ent']),
                      'head_rel': np.mean(batch_acc['head_rel']),
                      'tail_rel': np.mean(batch_acc['tail_rel']),
                      'overall': np.mean(list(itertools.chain(*batch_acc.values())))}
        print(
            f"batch_method select耗时{round(batch_time, 2)}秒")

        print(f"total time: {round(time.time() - t_ep, 2)}")
        RS_logger.ts_log().add_scalars(f"{RS_logger.get_self_text()}/pseudo acc", store_dict, RS_logger.get_cur_ep())
        RS_logger.extent_json(store_dict)
        return batch_new_data_list


def student_training_process(model, FIRST_EPOCHS, optimizer, loss_func, scheduler, train_dataloader, valid_dataloader,
                             seq_val_acc):
    if FIRST_EPOCHS == 0:
        return [0]
    mu = 1
    # ent_p=seq_val_acc[0]/2 if seq_val_acc[0]>0.6 else seq_val_acc[0]
    # rel_p=seq_val_acc[0] if (seq_val_acc[1]+seq_val_acc[2])/2<0.2 else (seq_val_acc[1]+seq_val_acc[2])/2
    ent_p = 1
    rel_p = 1
    for ep in range(FIRST_EPOCHS):
        ## train
        model.train()
        t_ep = time.time()
        total_loss, total_ent_sample_acc, total_head_rel_sample_acc, total_tail_rel_sample_acc = 0., 0., 0., 0.
        for batch_ind, batch_train_data in enumerate(train_dataloader):
            t_batch = time.time()
            z = (2 * len(rel2id) + 1)
            steps_per_ep = len(train_dataloader)
            total_steps = hyper_parameters["loss_weight_recover_steps"] + 1  # + 1 avoid division by zero error
            current_step = steps_per_ep * ep + batch_ind
            w_ent = max(1 / z + 1 - current_step / total_steps, 1 / z) * ent_p
            w_rel = min((len(rel2id) / z) * current_step / total_steps,
                        (len(rel2id) / z)) * rel_p
            loss_weights = {"ent": w_ent, "rel": w_rel}

            if config["encoder"] == "BERT":
                sample_list, batch_input_ids, batch_attention_mask, batch_token_type_ids, tok2char_span_list, pseudo_flag, batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = batch_train_data
                batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = (
                    batch_input_ids.to(device), batch_attention_mask.to(device),
                    batch_token_type_ids.to(device), batch_ent_shaking_tag.to(device),
                    batch_head_rel_shaking_tag.to(device),
                    batch_tail_rel_shaking_tag.to(device))

            elif config["encoder"] in {
                "BiLSTM",
            }:
                sample_list, batch_input_ids, tok2char_span_list, pseudo_flag, batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = batch_train_data

                batch_input_ids, batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = (
                    batch_input_ids.to(device), batch_ent_shaking_tag.to(device),
                    batch_head_rel_shaking_tag.to(device),
                    batch_tail_rel_shaking_tag.to(device))
            ## concat pseudo training data

            # model forward
            # zero the parameter gradients
            optimizer.zero_grad()
            if config["encoder"] == "BERT":
                ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs = model(
                    batch_input_ids,
                    batch_attention_mask,
                    batch_token_type_ids,
                )
            elif config["encoder"] in {
                "BiLSTM",
            }:
                ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs = model(
                    batch_input_ids)
            factor = 1 + (mu - 1) * sum(pseudo_flag) / len(pseudo_flag)
            w_ent, w_rel = loss_weights["ent"], loss_weights["rel"]
            loss = factor * (w_ent * loss_func(
                ent_shaking_outputs, batch_ent_shaking_tag) + w_rel * loss_func(
                head_rel_shaking_outputs,
                batch_head_rel_shaking_tag) + w_rel * loss_func(
                tail_rel_shaking_outputs, batch_tail_rel_shaking_tag))

            loss.backward()
            optimizer.step()

            ent_sample_acc = metrics.get_sample_accuracy(ent_shaking_outputs,
                                                         batch_ent_shaking_tag)
            head_rel_sample_acc = metrics.get_sample_accuracy(
                head_rel_shaking_outputs, batch_head_rel_shaking_tag)
            tail_rel_sample_acc = metrics.get_sample_accuracy(
                tail_rel_shaking_outputs, batch_tail_rel_shaking_tag)

            loss, ent_sample_acc, head_rel_sample_acc, tail_rel_sample_acc = loss.item(), ent_sample_acc.item(), head_rel_sample_acc.item(
            ), tail_rel_sample_acc.item()
            scheduler.step()

            total_loss += loss
            total_ent_sample_acc += ent_sample_acc
            total_head_rel_sample_acc += head_rel_sample_acc
            total_tail_rel_sample_acc += tail_rel_sample_acc

            avg_loss = total_loss / (batch_ind + 1)
            avg_ent_sample_acc = total_ent_sample_acc / (batch_ind + 1)
            avg_head_rel_sample_acc = total_head_rel_sample_acc / (batch_ind +
                                                                   1)
            avg_tail_rel_sample_acc = total_tail_rel_sample_acc / (batch_ind +
                                                                   1)

            batch_print_format = "\rproject: {}, run_name: {}, Epoch: {}/{}, batch: {}/{}, train_loss: {}, " + "t_ent_sample_acc: {}, t_head_rel_sample_acc: {}, t_tail_rel_sample_acc: {}," + "lr: {}, batch_time: {}, total_time: {} -------------"

            print(batch_print_format.format(
                experiment_name,
                config["run_name"],
                ep + 1,
                FIRST_EPOCHS,
                batch_ind + 1,
                len(train_dataloader),
                avg_loss,
                avg_ent_sample_acc,
                avg_head_rel_sample_acc,
                avg_tail_rel_sample_acc,
                optimizer.param_groups[0]['lr'],
                time.time() - t_batch,
                time.time() - t_ep,
            ),
                end="")

        if config["logger"] == "wandb" and batch_ind % hyper_parameters[
            "log_interval"] == 0:
            logger.log({
                "train_loss": avg_loss,
                "train_ent_seq_acc": avg_ent_sample_acc,
                "train_head_rel_acc": avg_head_rel_sample_acc,
                "train_tail_rel_acc": avg_tail_rel_sample_acc,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "time": time.time() - t_ep,
            })

        if config[
            "logger"] != "wandb":  # only log once for training if logger is not wandb
            logger.log({
                "train_loss": avg_loss,
                "train_ent_seq_acc": avg_ent_sample_acc,
                "train_head_rel_acc": avg_head_rel_sample_acc,
                "train_tail_rel_acc": avg_tail_rel_sample_acc,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "time": time.time() - t_ep,
            })

    ## valid
    model.eval()
    t_ep = time.time()
    total_ent_sample_acc, total_head_rel_sample_acc, total_tail_rel_sample_acc = 0., 0., 0.
    total_rel_correct_num, total_rel_pred_num, total_rel_gold_num = 0, 0, 0
    for batch_ind, batch_valid_data in enumerate(
            tqdm(valid_dataloader, desc="Validating", disable=config["disable_tqdm"])):
        if config["encoder"] == "BERT":
            sample_list, batch_input_ids, batch_attention_mask, batch_token_type_ids, tok2char_span_list, batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = batch_valid_data

            batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = (
                batch_input_ids.to(device), batch_attention_mask.to(device),
                batch_token_type_ids.to(device), batch_ent_shaking_tag.to(device),
                batch_head_rel_shaking_tag.to(device),
                batch_tail_rel_shaking_tag.to(device))

        elif config["encoder"] in {
            "BiLSTM",
        }:
            sample_list, batch_input_ids, tok2char_span_list, batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = batch_valid_data

            batch_input_ids, batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = (
                batch_input_ids.to(device), batch_ent_shaking_tag.to(device),
                batch_head_rel_shaking_tag.to(device),
                batch_tail_rel_shaking_tag.to(device))

        with torch.no_grad():
            if config["encoder"] == "BERT":
                ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs = model(
                    batch_input_ids,
                    batch_attention_mask,
                    batch_token_type_ids,
                )
            elif config["encoder"] in {
                "BiLSTM",
            }:
                ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs = model(
                    batch_input_ids)

        ent_sample_acc = metrics.get_sample_accuracy(ent_shaking_outputs,
                                                     batch_ent_shaking_tag)
        head_rel_sample_acc = metrics.get_sample_accuracy(
            head_rel_shaking_outputs, batch_head_rel_shaking_tag)
        tail_rel_sample_acc = metrics.get_sample_accuracy(
            tail_rel_shaking_outputs, batch_tail_rel_shaking_tag)

        ent_sample_acc, head_rel_sample_acc, tail_rel_sample_acc = ent_sample_acc.item(), head_rel_sample_acc.item(
        ), tail_rel_sample_acc.item()

        total_ent_sample_acc += ent_sample_acc
        total_head_rel_sample_acc += head_rel_sample_acc
        total_tail_rel_sample_acc += tail_rel_sample_acc

    avg_ent_sample_acc = total_ent_sample_acc / len(valid_dataloader)
    avg_head_rel_sample_acc = total_head_rel_sample_acc / len(valid_dataloader)
    avg_tail_rel_sample_acc = total_tail_rel_sample_acc / len(valid_dataloader)

    # rel_prf = metrics.get_prf_scores(total_rel_correct_num,
    #                                  total_rel_pred_num,
    #                                  total_rel_gold_num)

    # log_dict = {
    #     "val_ent_seq_acc": avg_ent_sample_acc,
    #     "val_head_rel_acc": avg_head_rel_sample_acc,
    #     "val_tail_rel_acc": avg_tail_rel_sample_acc,
    #     "val_prec": rel_prf[0],
    #     "val_recall": rel_prf[1],
    #     "val_f1": rel_prf[2],
    #     "time": time.time() - t_ep,
    # }
    log_dict = {
        "student val_ent_seq_acc": avg_ent_sample_acc,
        "student val_head_rel_acc": avg_head_rel_sample_acc,
        "student val_tail_rel_acc": avg_tail_rel_sample_acc,
        "time": time.time() - t_ep,
    }
    logger.log(log_dict)
    RS_logger.ts_log().add_scalars("Student Valid", {
        "e_acc": avg_ent_sample_acc,
        "h_acc": avg_head_rel_sample_acc,
        "t_acc": avg_tail_rel_sample_acc,
    }, RS_logger.get_cur_ep())
    # RS_logger.epoch_log("student model", str((avg_ent_sample_acc, avg_head_rel_sample_acc, avg_tail_rel_sample_acc)))
    return (avg_ent_sample_acc, avg_head_rel_sample_acc, avg_tail_rel_sample_acc)


def test_step(teacher_model, student_model, save_res_dir, max_test_seq_len):
    # For test
    test_data_path_dict = {}
    for file_path in glob.glob(test_data_path):
        file_name = re.search("(.*?)\.json", file_path.split("/")[-1]).group(1)
        test_data_path_dict[file_name] = file_path
    test_data_dict = {}  # filename2data
    for file_name, path in test_data_path_dict.items():
        test_data_dict[file_name] = json.load(open(path, "r", encoding="utf-8"))
    all_data = []  # 数据全加入alldata中
    for data in list(test_data_dict.values()):
        all_data.extend(data)

    max_tok_num = 0
    for sample in tqdm(all_data, desc="Calculate the max token number", disable=config["disable_tqdm"]):
        tokens = tokenize(sample["text"])
        max_tok_num = max(len(tokens), max_tok_num)

    split_test_data = False
    if max_tok_num > max_test_seq_len:
        split_test_data = True
        print("max_tok_num: {}, lagger than max_test_seq_len: {}, test data will be split!".format(max_tok_num,
                                                                                                   max_test_seq_len))
    else:
        print("max_tok_num: {}, less than or equal to max_test_seq_len: {}, no need to split!".format(max_tok_num,
                                                                                                      max_test_seq_len))
    max_test_seq_len = min(max_tok_num, max_test_seq_len)  # 137
    print(f"max_test_seq_len: {max_test_seq_len}")
    if force_split:
        split_test_data = True
        print("force to split the test dataset!")

    ori_test_data_dict = copy.deepcopy(test_data_dict)
    if split_test_data:
        test_data_dict = {}
        for file_name, data in ori_test_data_dict.items():
            test_data_dict[file_name] = preprocessor.split_into_short_samples(data,
                                                                              max_seq_len,
                                                                              sliding_len=sliding_len,
                                                                              encoder=test_config["encoder"],
                                                                              data_type="test")

    # get model state paths
    model_state_dir = test_config["model_state_dict_dir"]
    target_run_ids = set(test_config["run_ids"])
    run_id2model_state_paths = {}  # model state path是由id来命名的
    # for root, dirs, files in os.walk(
    #         model_state_dir):  # 对这个目录及其子目录都进行访问，找出target_run_ids界定的那些ids并存到runid2modelstatepath中
    #     for file_name in files:
    #         #         set_trace()
    #         run_id = root.split("/")[-1].split("-")[-1]
    #         if re.match(".*model_state.*\.pt", file_name) and run_id in target_run_ids:
    #             if run_id not in run_id2model_state_paths:
    #                 run_id2model_state_paths[run_id] = []
    #             model_state_path = os.path.join(root, file_name)
    #             run_id2model_state_paths[run_id].append(model_state_path)

    res_dict = {}
    predict_statistics = {}
    save_dir4run = os.path.join(save_res_dir, config["run_id"])
    if test_config["save_res"] and not os.path.exists(save_dir4run):
        os.makedirs(save_dir4run)

    for file_name, short_data in test_data_dict.items():
        res_num = 0  # re.search("(\d+)", model_state_path.split("/")[-1]).group(1)
        # if len(res_num) == 0:
        #     res_num = 0
        save_path = os.path.join(save_dir4run, "{}_res_{}.json".format(file_name, res_num))
        # if os.path.exists(save_path):
        #     pred_sample_list = [json.loads(line) for line in open(save_path, "r", encoding="utf-8")]
        #     print("{} already exists, load it directly!".format(save_path))
        # else:
        #     # predict
        ori_test_data = ori_test_data_dict[file_name]
        pred_sample_list = predict(short_data, ori_test_data, split_test_data, teacher_model)

        res_dict[save_path] = pred_sample_list
        predict_statistics[save_path] = len([s for s in pred_sample_list if len(s["relation_list"]) > 0])
        pprint(predict_statistics)
        if student_model is not None:
            save_path_student = os.path.join(save_dir4run, "{}_res_{}.json".format(file_name, res_num + 1))
            # if os.path.exists(save_path_student):
            #     pred_sample_list_student = [json.loads(line) for line in open(save_path_student, "r", encoding="utf-8")]
            #     print("{} already exists, load it directly!".format(save_path_student))
            # else:
            #     # predict
            ori_test_data = ori_test_data_dict[file_name]
            pred_sample_list_student = predict(short_data, ori_test_data, split_test_data, student_model)
            res_dict[save_path_student] = pred_sample_list_student
            predict_statistics[save_path_student] = len(
                [s for s in pred_sample_list_student if len(s["relation_list"]) > 0])

    # check
    for path, res in res_dict.items():
        for sample in tqdm(res, desc="check char span", disable=config["disable_tqdm"]):
            text = sample["text"]
            for rel in sample["relation_list"]:
                assert rel["subject"] == text[rel["subj_char_span"][0]:rel["subj_char_span"][1]]
                assert rel["object"] == text[rel["obj_char_span"][0]:rel["obj_char_span"][1]]

    # save
    if test_config["save_res"]:
        for path, res in res_dict.items():
            with open(path, "w", encoding="utf-8") as file_out:
                for sample in tqdm(res, desc="Output", disable=config["disable_tqdm"]):
                    if len(sample["relation_list"]) == 0:
                        continue
                    json_line = json.dumps(sample, ensure_ascii=False)
                    file_out.write("{}\n".format(json_line))

    # score
    f = open(save_dir4run + "test_results.json", "w+")
    if test_config["score"]:
        score_dict = {}
        correct = hyper_parameters_test["match_pattern"]
        #     correct = "whole_text"
        for file_path, pred_samples in res_dict.items():
            run_id = file_path.split("/")[-2]
            log_filename = re.search("(.*?)_res_\d+\.json", file_path.split("/")[-1]).group(1) + "-" + str(
                re.search(".*?_res_(\d+)\.json", file_path.split("/")[-1]).group(1))
            file_name = re.search("(.*?)_res_\d+\.json", file_path.split("/")[-1]).group(1)
            gold_test_data = ori_test_data_dict[file_name]
            prf = get_test_prf(pred_samples, gold_test_data, pattern=correct)
            if run_id not in score_dict:
                score_dict[run_id] = {}
            score_dict[run_id][log_filename] = prf
            log_dict = {
                "run_id": run_id,
                "type": file_path,
                "test_prec": prf[0],
                "val_recall": prf[1],
                "val_f1": prf[2],
            }
            logger.log(log_dict)
            RS_logger.ts_log().add_scalar(f"{RS_logger.get_self_text()}/Test/{log_filename}", prf[2],
                                           RS_logger.get_cur_ep())
            json.dump(log_dict, f)
        print("---------------- Results -----------------------")
        pprint(score_dict)
        # RS_logger.ts_log().add_scalars("Test", {
        #     "e_acc": avg_ent_sample_acc,
        #     "h_acc": avg_head_rel_sample_acc,
        #     "t_acc": avg_tail_rel_sample_acc,
        # }, RS_logger.get_cur_ep())
        # RS_logger.epoch_log("test results",str(score_dict))
        RS_logger["Test Results"] = score_dict
    f.close()


def self_training():
    print("self-training start...\n")
    """
    two aspect:
    1. data handle:dataloader
    2. model,loss handle
    """
    # count how many rounds the whole big model has to train
    # a round means all unlabel data are labeled
    train_dataloader = labeled_dataloader
    # fine_map = collections.defaultdict(tuple)  # key:train id ; value:correct number
    max_student_val = -1
    max_val = -1
    seq_val_acc_student = [-2]
    last_val_acc = 0
    last_val_acc_stu = 0
    optional = False
    first_copy = True

    for total_epoch in range(TOTAL_EPOCHS):
        # Add an inner loop to judge the entire unlabeled data set finished

        print(f"Total epoch{total_epoch}:\n")
        # FIRST_EPOCHS=min(8,hyper_parameters["epochs"]-2+ 2*total_epoch)
        seq_val_acc = teacher_training_process(modelf1, FIRST_EPOCHS, optimizer1, loss_func1, scheduler1,
                                               train_dataloader, valid_dataloader)
        # -------train f1 end---------
        # -------generate pseudo label---------
        # RS_logger.log("teacher pseudo:")
        RS_logger["Epoch"] = total_epoch
        RS_logger["teacher val"] = seq_val_acc
        RS_logger.set_text("teacher")
        batch_new_data_list = generate_pseudo(modelf1, seq_val_acc, unlabeled_dataloader_all,
                                              STRATEGY=config["use_strategy"])
        last_val_acc = seq_val_acc
        # -------generate pseudo label end---------
        # -------generate student data---------
        train_add_dataset = labeled_dataloader.dataset + MyDataset(batch_new_data_list)
        if use_two_model:
            student_train_dataloader = DataLoader(
                train_add_dataset,  # The training samples.
                batch_size=hyper_parameters["batch_size"],
                shuffle=True,
                num_workers=0,
                drop_last=False,
                collate_fn=functools.partial(data_maker.generate_batch, data_type="student"),
            )
            # SECOND_EPOCHS = int(hyper_parameters["student_epochs"] * sum(seq_val_acc))
            SECOND_EPOCHS = int(hyper_parameters["student_epochs"])

            if sum(seq_val_acc) < 1.2 or sum(seq_val_acc_student) < 1.2:
                print("Use teacher information.")
                train_dataloader = DataLoader(
                    train_add_dataset,  # The training samples.
                    batch_size=hyper_parameters["batch_size"],
                    shuffle=True,
                    num_workers=0,
                    drop_last=False,
                    collate_fn=data_maker.generate_batch,
                )
            else:
                if first_copy:
                    modelf2 = copy.deepcopy(modelf1)
                    first_copy = False
                print("Use student information.")
                # -------student model train---------
                print("student training start:")
                RS_logger.set_text("student")
                seq_val_acc_student = student_training_process(modelf2, SECOND_EPOCHS, optimizer2, loss_func2,
                                                               scheduler2,
                                                               student_train_dataloader, valid_dataloader, seq_val_acc)

                RS_logger["student val"] = seq_val_acc_student
                # -------student model end---------
                # -------student generate pseudo label---------
                print("student generate pseudo")
                # RS_logger.log("student pseudo:")
                batch_new_data_list = generate_pseudo(modelf2, seq_val_acc_student, unlabeled_dataloader_all,
                                                      STRATEGY=config["use_strategy"])
                # -------generate next data---------
                train_add_dataset = labeled_dataloader.dataset + MyDataset(batch_new_data_list)
                train_dataloader = DataLoader(
                    train_add_dataset,  # The training samples.
                    batch_size=hyper_parameters["batch_size"],
                    shuffle=True,
                    num_workers=0,
                    drop_last=False,
                    collate_fn=data_maker.generate_batch,
                )
                if SECOND_EPOCHS != 0:
                    max_student_val = save_model(current=sum(seq_val_acc_student), last=max_student_val,
                                                 save_path=os.path.join(
                                                     model_state_dict_dir,
                                                     "model_state_dict_student_best.pt"), current_model=modelf2)
            # save model
            max_val = save_model(current=sum(seq_val_acc), last=max_val, save_path=os.path.join(
                model_state_dict_dir,
                "model_state_dict_best.pt"), current_model=modelf1)

            if optional:
                modelf1.load_state_dict(torch.load(os.path.join(
                    model_state_dict_dir,
                    "model_state_dict_student_best.pt")), strict=False)
                modelf2 = TPLinkerBert(
                    encoder,
                    len(rel2id),
                    hyper_parameters["shaking_type"],
                    hyper_parameters["inner_enc_type"],
                    hyper_parameters["dist_emb_size"],
                    hyper_parameters["ent_add_dist"],
                    hyper_parameters["rel_add_dist"],
                    dropout=config["two_models_hyper_parameters"]["student_dropout"],
                    is_dropout=True
                ).to(device)

        else:
            train_dataloader = DataLoader(
                train_add_dataset,  # The training samples.
                batch_size=hyper_parameters["batch_size"],
                shuffle=True,
                num_workers=0,
                drop_last=False,
                collate_fn=data_maker.generate_batch,
            )
            # save model
            max_val = save_model(current=sum(seq_val_acc), last=max_val, save_path=os.path.join(
                model_state_dict_dir,
                "model_state_dict_best.pt"), current_model=modelf1)

        # test
        if config["fr_scratch"]:
            if total_epoch > 0:
                if use_two_model and first_copy == False:
                    test_step(modelf1, modelf2, save_res_dir, max_test_seq_len)
                else:
                    test_step(modelf1, None, save_res_dir, max_test_seq_len)
        else:

            if total_epoch > 0:
                if use_two_model and SECOND_EPOCHS != 0:
                    test_step(modelf1, modelf2, save_res_dir, max_test_seq_len)
                else:
                    test_step(modelf1, None, save_res_dir, max_test_seq_len)
        RS_logger.get_file(os.path.join(config["log_path"], "results.json")).write(json.dumps(RS_logger.get_json()))
        RS_logger.get_file(os.path.join(config["log_path"], "results.json")).write("\n")
        RS_logger._json_pool = {}
        RS_logger.add_epoch()
        torch.cuda.empty_cache()

    modelf1.load_state_dict(torch.load(os.path.join(
        model_state_dict_dir,
        "model_state_dict_best.pt")))
    if use_two_model and SECOND_EPOCHS != 0:
        modelf2.load_state_dict(torch.load(os.path.join(
            model_state_dict_dir,
            "model_state_dict_student_best.pt")))
        test_step(modelf1, modelf2, save_res_dir, max_test_seq_len)
    else:
        test_step(modelf1, None, save_res_dir, max_test_seq_len)


def increment_training():
    print("increment_training start...\n")
    """
    two aspect:
    1. data handle:dataloader
    2. model,loss handle
    """
    # count how many rounds the whole big model has to train
    # a round means all unlabel data are labeled

    # fine_map = collections.defaultdict(tuple)  # key:train id ; value:correct number
    max_student_val = -1
    max_val = -1
    last_val_acc = 0
    last_val_acc_stu = 0
    # TOTAL_EPOCHS = hyper_parameters["TOTAL_EPOCHS"]
    # TOTAL_EPOCHS=TOTAL_EPOCHS//META_EPOCHS# to keep total iterations equal.
    for total_epoch in range(TOTAL_EPOCHS):
        train_dataloader = labeled_dataloader
        # Add an inner loop to judge the entire unlabeled data set finished
        print(f"Total epoch{total_epoch}:\n")
        for meta_epoch in range(META_EPOCHS):
            print(f"Mate epoch{meta_epoch}:\n")
            seq_val_acc = teacher_training_process(modelf1, FIRST_EPOCHS, optimizer1, loss_func1, scheduler1,
                                                   train_dataloader, valid_dataloader)
            RS_logger.log("teacher pseudo:")
            batch_new_data_list = generate_pseudo(modelf1, seq_val_acc, unlabeled_dataloader_all[meta_epoch],
                                                  STRATEGY=config["use_strategy"])
            train_add_dataset = labeled_dataloader.dataset + MyDataset(batch_new_data_list)
            train_dataloader = DataLoader(
                train_add_dataset,  # The training samples.
                batch_size=hyper_parameters["batch_size"],
                shuffle=True,
                num_workers=0,
                drop_last=False,
                collate_fn=data_maker.generate_batch,
            )
            seq_val_acc = teacher_training_process(modelf1, FIRST_EPOCHS, optimizer1, loss_func1, scheduler1,
                                                   train_dataloader, valid_dataloader)

            RS_logger.add_epoch()
        max_val = save_model(current=sum(seq_val_acc), last=max_val, save_path=os.path.join(
            model_state_dict_dir,
            "model_state_dict_best.pt"), current_model=modelf1)
        torch.cuda.empty_cache()

        # test
        if config["fr_scratch"]:
            if total_epoch == TOTAL_EPOCHS - 1 or total_epoch == TOTAL_EPOCHS - 2:
                if use_two_model and SECOND_EPOCHS != 0:
                    test_step(modelf1, modelf2, save_res_dir, max_test_seq_len)
                else:
                    test_step(modelf1, None, save_res_dir, max_test_seq_len)
        else:

            if total_epoch > 0:
                if use_two_model and SECOND_EPOCHS != 0:
                    test_step(modelf1, modelf2, save_res_dir, max_test_seq_len)
                else:
                    test_step(modelf1, None, save_res_dir, max_test_seq_len)

    modelf1.load_state_dict(torch.load(os.path.join(
        model_state_dict_dir,
        "model_state_dict_best.pt")))
    if use_two_model and SECOND_EPOCHS != 0:
        modelf2.load_state_dict(torch.load(os.path.join(
            model_state_dict_dir,
            "model_state_dict_student_best.pt")))
        test_step(modelf1, modelf2, save_res_dir, max_test_seq_len)
    else:
        test_step(modelf1, None, save_res_dir, max_test_seq_len)


def mean_teacher():
    print("mean teacher start...\n")
    """
    two aspect:
    1. data handle:dataloader
    2. model,loss handle
    """
    # count how many rounds the whole big model has to train
    # a round means all unlabel data are labeled
    train_dataloader = labeled_dataloader
    # fine_map = collections.defaultdict(tuple)  # key:train id ; value:correct number
    max_student_val = -1
    max_val = -1
    last_val_acc = 0
    last_val_acc_stu = 0
    for total_epoch in range(TOTAL_EPOCHS):
        # Add an inner loop to judge the entire unlabeled data set finished
        print(f"Total epoch{total_epoch}:\n")
        SECOND_EPOCHS = int(hyper_parameters["student_epochs"])
        seq_val_acc_student = student_training_process(modelf2, SECOND_EPOCHS, optimizer2, loss_func2,
                                                       scheduler2,
                                                       student_train_dataloader, valid_dataloader, seq_val_acc)

        seq_val_acc = teacher_training_process(modelf1, FIRST_EPOCHS, optimizer1, loss_func1, scheduler1,
                                               train_dataloader, valid_dataloader)
        # -------train f1 end---------
        # -------generate pseudo label---------
        RS_logger.log("teacher pseudo:")
        batch_new_data_list = generate_pseudo(modelf1, seq_val_acc, unlabeled_dataloader_all,
                                              STRATEGY=config["use_strategy"])
        last_val_acc = seq_val_acc
        # -------generate pseudo label end---------
        # -------generate student data---------
        train_add_dataset = labeled_dataloader.dataset + MyDataset(batch_new_data_list)
        if use_two_model:
            student_train_dataloader = DataLoader(
                train_add_dataset,  # The training samples.
                batch_size=hyper_parameters["batch_size"],
                shuffle=True,
                num_workers=0,
                drop_last=False,
                collate_fn=functools.partial(data_maker.generate_batch, data_type="student"),
            )
            if sum(seq_val_acc) < 1.2:
                SECOND_EPOCHS = int(hyper_parameters["student_epochs"] * sum(seq_val_acc) // 1.2)
                print("Use teacher information.")
                train_dataloader = DataLoader(
                    train_add_dataset,  # The training samples.
                    batch_size=hyper_parameters["batch_size"],
                    shuffle=True,
                    num_workers=0,
                    drop_last=False,
                    collate_fn=data_maker.generate_batch,
                )
                # -------student model train---------
                print("student training start:")
                seq_val_acc_student = student_training_process(modelf2, SECOND_EPOCHS, optimizer2, loss_func2,
                                                               scheduler2,
                                                               student_train_dataloader, valid_dataloader, seq_val_acc)

                # -------student model end---------
            else:
                # SECOND_EPOCHS = int(hyper_parameters["student_epochs"] * sum(seq_val_acc))
                SECOND_EPOCHS = int(hyper_parameters["student_epochs"])
                print("Use student information.")
                # -------student model train---------
                print("student training start:")
                seq_val_acc_student = student_training_process(modelf2, SECOND_EPOCHS, optimizer2, loss_func2,
                                                               scheduler2,
                                                               student_train_dataloader, valid_dataloader, seq_val_acc)
                # -------student model end---------
                # -------student generate pseudo label---------
                print("student generate pseudo")
                RS_logger.log("student pseudo:")
                batch_new_data_list = generate_pseudo(modelf2, seq_val_acc_student, unlabeled_dataloader_all,
                                                      STRATEGY=config["use_strategy"])

                # -------generate next data---------
                train_add_dataset = labeled_dataloader.dataset + MyDataset(batch_new_data_list)
                train_dataloader = DataLoader(
                    train_add_dataset,  # The training samples.
                    batch_size=hyper_parameters["batch_size"],
                    shuffle=True,
                    num_workers=0,
                    drop_last=False,
                    collate_fn=data_maker.generate_batch,
                )
            last_val_acc_stu = seq_val_acc_student
            # save model
            max_val = save_model(current=sum(seq_val_acc), last=max_val, save_path=os.path.join(
                model_state_dict_dir,
                "model_state_dict_best.pt"), current_model=modelf1)
            if SECOND_EPOCHS != 0:
                max_student_val = save_model(current=sum(seq_val_acc_student), last=max_student_val,
                                             save_path=os.path.join(
                                                 model_state_dict_dir,
                                                 "model_state_dict_student_best.pt"), current_model=modelf2)
        else:
            train_dataloader = DataLoader(
                train_add_dataset,  # The training samples.
                batch_size=hyper_parameters["batch_size"],
                shuffle=True,
                num_workers=0,
                drop_last=False,
                collate_fn=data_maker.generate_batch,
            )
            # save model
            max_val = save_model(current=sum(seq_val_acc), last=max_val, save_path=os.path.join(
                model_state_dict_dir,
                "model_state_dict_best.pt"), current_model=modelf1)
        batch_new_data_list = []
        # # 清理尾部数据
        # if len(batch_new_data_list) > 0:  # 1000大概2~3GB
        #     # 文件流操作
        #     with open("Trunk" + str(trunk_count) + ".pkl", "wb") as f:
        #         pickle.dump(batch_new_data_list, f)
        #     trunk_count += 1
        #     size_count.append(len(batch_new_data_list))
        #     batch_new_data_list = []
        # excellent set record and update
        # if len(exct_indexs_list)>0:#如果集合有值，应该被每次都加到train_dataloader中
        #     # 可靠集合
        #     # extract origin data firstly
        #     pprint({"excellent set: {}".format(len(exct_indexs_list))})
        #     train_add_dataset = train_dataloader.dataset + MyDataset(exct_indexs_list)
        #     train_dataloader = DataLoader(
        #         train_add_dataset,  # The training samples.
        #         batch_size=hyper_parameters["batch_size"],
        #         shuffle=True,
        #         num_workers=0,
        #         drop_last=False,
        #         collate_fn=data_maker.generate_batch,
        #     )

        # exct_indexs_list = exct_extract(fine_map)
        # if len(exct_indexs_list)!=0:
        #     pprint({"excellent set: {}".format(len(exct_indexs_list))})
        #     train_add_dataset = train_dataloader.dataset + MyDataset(exct_indexs_list)
        #     train_dataloader = DataLoader(
        #         train_add_dataset,  # The training samples.
        #         batch_size=hyper_parameters["batch_size"],
        #         shuffle=True,
        #         num_workers=0,
        #         drop_last=False,
        #         collate_fn=data_maker.generate_batch,
        #     )
        # exct_indexs_list=[]
        torch.cuda.empty_cache()

        # test
        if config["fr_scratch"]:
            if total_epoch == TOTAL_EPOCHS - 1 or total_epoch == TOTAL_EPOCHS - 2:
                if use_two_model and SECOND_EPOCHS != 0:
                    test_step(modelf1, modelf2, save_res_dir, max_test_seq_len)
                else:
                    test_step(modelf1, None, save_res_dir, max_test_seq_len)
        else:

            if total_epoch > 5:
                if use_two_model and SECOND_EPOCHS != 0:
                    test_step(modelf1, modelf2, save_res_dir, max_test_seq_len)
                else:
                    test_step(modelf1, None, save_res_dir, max_test_seq_len)

    modelf1.load_state_dict(torch.load(os.path.join(
        model_state_dict_dir,
        "model_state_dict_best.pt")))
    if use_two_model and SECOND_EPOCHS != 0:
        modelf2.load_state_dict(torch.load(os.path.join(
            model_state_dict_dir,
            "model_state_dict_student_best.pt")))
        test_step(modelf1, modelf2, save_res_dir, max_test_seq_len)
    else:
        test_step(modelf1, None, save_res_dir, max_test_seq_len)


def predict(test_data, ori_test_data, split_test_data, model):
    '''
    test_data: if split, it would be samples with subtext
    ori_test_data: the original data has not been split, used to get original text here
    '''
    model.eval()
    indexed_test_data = data_maker.get_indexed_data(test_data, max_seq_len, data_type="test")  # fill up to max_seq_len
    test_dataloader = DataLoader(MyDataset(indexed_test_data),
                                 batch_size=hyper_parameters["batch_size"],
                                 shuffle=False,
                                 num_workers=6,
                                 drop_last=False,
                                 collate_fn=lambda data_batch: data_maker.generate_batch(data_batch, data_type="test"),
                                 )

    pred_sample_list = []
    for batch_test_data in tqdm(test_dataloader, desc="Predicting", disable=config["disable_tqdm"]):
        if config["encoder"] == "BERT":
            sample_list, batch_input_ids, \
            batch_attention_mask, batch_token_type_ids, \
            tok2char_span_list, _, _, _ = batch_test_data

            batch_input_ids, \
            batch_attention_mask, \
            batch_token_type_ids = (batch_input_ids.to(device),
                                    batch_attention_mask.to(device),
                                    batch_token_type_ids.to(device))

        elif config["encoder"] in {"BiLSTM", }:
            sample_list, batch_input_ids, tok2char_span_list, _, _, _ = batch_test_data
            batch_input_ids = batch_input_ids.to(device)

        with torch.no_grad():
            if config["encoder"] == "BERT":
                batch_ent_shaking_outputs, \
                batch_head_rel_shaking_outputs, \
                batch_tail_rel_shaking_outputs = model(batch_input_ids,
                                                       batch_attention_mask,
                                                       batch_token_type_ids,
                                                       )
            elif config["encoder"] in {"BiLSTM", }:
                batch_ent_shaking_outputs, \
                batch_head_rel_shaking_outputs, \
                batch_tail_rel_shaking_outputs = model(batch_input_ids)

        batch_ent_shaking_tag, \
        batch_head_rel_shaking_tag, \
        batch_tail_rel_shaking_tag = torch.argmax(batch_ent_shaking_outputs, dim=-1), \
                                     torch.argmax(batch_head_rel_shaking_outputs, dim=-1), \
                                     torch.argmax(batch_tail_rel_shaking_outputs, dim=-1)

        for ind in range(len(sample_list)):
            gold_sample = sample_list[ind]
            text = gold_sample["text"]
            text_id = gold_sample["id"]
            tok2char_span = tok2char_span_list[ind]
            ent_shaking_tag, \
            head_rel_shaking_tag, \
            tail_rel_shaking_tag = batch_ent_shaking_tag[ind], \
                                   batch_head_rel_shaking_tag[ind], \
                                   batch_tail_rel_shaking_tag[ind]

            tok_offset, char_offset = 0, 0
            if split_test_data:
                tok_offset, char_offset = gold_sample["tok_offset"], gold_sample["char_offset"]
            rel_list = handshaking_tagger.decode_rel_fr_shaking_tag(text,
                                                                    ent_shaking_tag,
                                                                    head_rel_shaking_tag,
                                                                    tail_rel_shaking_tag,
                                                                    tok2char_span,
                                                                    tok_offset=tok_offset, char_offset=char_offset)
            pred_sample_list.append({
                "text": text,
                "id": text_id,
                "relation_list": rel_list,
            })

    # merge
    text_id2rel_list = {}
    for sample in pred_sample_list:
        text_id = sample["id"]
        if text_id not in text_id2rel_list:
            text_id2rel_list[text_id] = sample["relation_list"]
        else:
            text_id2rel_list[text_id].extend(sample["relation_list"])

    text_id2text = {sample["id"]: sample["text"] for sample in ori_test_data}
    merged_pred_sample_list = []
    for text_id, rel_list in text_id2rel_list.items():
        merged_pred_sample_list.append({
            "id": text_id,
            "text": text_id2text[text_id],
            "relation_list": filter_duplicates(rel_list),
        })

    return merged_pred_sample_list


def get_test_prf(pred_sample_list, gold_test_data, pattern="only_head_text"):
    text_id2gold_n_pred = {}
    for sample in gold_test_data:
        text_id = sample["id"]
        text_id2gold_n_pred[text_id] = {
            "gold_relation_list": sample["relation_list"],
        }

    for sample in pred_sample_list:
        text_id = sample["id"]
        text_id2gold_n_pred[text_id]["pred_relation_list"] = sample["relation_list"]

    correct_num, pred_num, gold_num = 0, 0, 0
    for gold_n_pred in text_id2gold_n_pred.values():
        gold_rel_list = gold_n_pred["gold_relation_list"]
        pred_rel_list = gold_n_pred["pred_relation_list"] if "pred_relation_list" in gold_n_pred else []
        if pattern == "only_head_index":
            gold_rel_set = set(
                ["{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["predicate"], rel["obj_tok_span"][0]) for rel
                 in gold_rel_list])
            pred_rel_set = set(
                ["{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0], rel["predicate"], rel["obj_tok_span"][0]) for rel
                 in pred_rel_list])
        elif pattern == "whole_span":
            gold_rel_set = set(["{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0],
                                                                            rel["subj_tok_span"][1], rel["predicate"],
                                                                            rel["obj_tok_span"][0],
                                                                            rel["obj_tok_span"][1]) for rel in
                                gold_rel_list])
            pred_rel_set = set(["{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0],
                                                                            rel["subj_tok_span"][1], rel["predicate"],
                                                                            rel["obj_tok_span"][0],
                                                                            rel["obj_tok_span"][1]) for rel in
                                pred_rel_list])
        elif pattern == "whole_text":
            gold_rel_set = set(
                ["{}\u2E80{}\u2E80{}".format(rel["subject"], rel["predicate"], rel["object"]) for rel in gold_rel_list])
            pred_rel_set = set(
                ["{}\u2E80{}\u2E80{}".format(rel["subject"], rel["predicate"], rel["object"]) for rel in pred_rel_list])
        elif pattern == "only_head_text":
            gold_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"].split(" ")[0], rel["predicate"],
                                                            rel["object"].split(" ")[0]) for rel in gold_rel_list])
            pred_rel_set = set(["{}\u2E80{}\u2E80{}".format(rel["subject"].split(" ")[0], rel["predicate"],
                                                            rel["object"].split(" ")[0]) for rel in pred_rel_list])

        for rel_str in pred_rel_set:
            if rel_str in gold_rel_set:
                correct_num += 1

        pred_num += len(pred_rel_set)
        gold_num += len(gold_rel_set)
    #     print((correct_num, pred_num, gold_num))
    prf = metrics.get_prf_scores(correct_num, pred_num, gold_num)
    return prf


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':

    # Load Data
    LABEL_OF_TRAIN = config["LABEL_OF_TRAIN"]  # Label ratio
    train_data = json.load(open(train_data_path, "r", encoding="utf-8"))
    valid_data = json.load(open(valid_data_path, "r", encoding="utf-8"))
    train_data_path = os.path.join(data_home, experiment_name,
                                   config["train_data"])
    if os.path.exists(os.path.join(data_home, experiment_name, "label.pkl")) and os.path.exists(
            os.path.join(data_home, experiment_name, "unlabel.pkl")) and not config["RE_DATA"]:
        with open(os.path.join(data_home, experiment_name, "label.pkl"), "rb") as f1, open(
                os.path.join(data_home, experiment_name, "unlabel.pkl"), "rb") as f2:
            train_data, unlabeled_train_data = pickle.load(f1), pickle.load(f2)
    else:
        train_data, unlabeled_train_data = stratified_sample(train_data, LABEL_OF_TRAIN)
        with open(os.path.join(data_home, experiment_name, "label.pkl"), "wb") as f1, open(
                os.path.join(data_home, experiment_name, "unlabel.pkl"), "wb") as f2:
            pickle.dump(train_data, f1)
            pickle.dump(unlabeled_train_data, f2)
    # # Extract part data
    # with open(train_data_path+'-sample',"w",encoding="utf-8") as f:
    #     f.write(json.dumps(train_data[:1000]))
    # with open(valid_data_path+'-sample',"w",encoding="utf-8") as f:
    #     f.write(json.dumps(valid_data[:200]))

    # Tokenizer
    if config["encoder"] == "BERT":
        tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"],
                                                      add_special_tokens=False,
                                                      do_lower_case=False)
        tokenize = tokenizer.tokenize
        get_tok2char_span_map = lambda text: tokenizer.encode_plus(
            text, return_offsets_mapping=True, add_special_tokens=False)[
            "offset_mapping"]
    elif config["encoder"] in {
        "BiLSTM",
    }:
        tokenize = lambda text: text.split(" ")


        def get_tok2char_span_map(text):
            tokens = text.split(" ")
            tok2char_span = []
            char_num = 0
            for tok in tokens:
                tok2char_span.append((char_num, char_num + len(tok)))
                char_num += len(tok) + 1  # +1: whitespace
            return tok2char_span

    # preprocess
    preprocessor = Preprocessor(tokenize_func=tokenize,
                                get_tok2char_span_map_func=get_tok2char_span_map)

    # In[ ]:
    # train and valid max token num
    max_tok_num = 0
    all_data = train_data + valid_data + unlabeled_train_data

    for sample in all_data:
        tokens = tokenize(sample["text"])
        max_tok_num = max(max_tok_num, len(tokens))
    max_tok_num

    # In[ ]:

    if max_tok_num > hyper_parameters["max_seq_len"]:
        train_data = preprocessor.split_into_short_samples(
            train_data,
            hyper_parameters["max_seq_len"],
            sliding_len=hyper_parameters["sliding_len"],
            encoder=config["encoder"])
        unlabeled_train_data = preprocessor.split_into_short_samples(
            unlabeled_train_data,
            hyper_parameters["max_seq_len"],
            sliding_len=hyper_parameters["sliding_len"],
            encoder=config["encoder"])
        valid_data = preprocessor.split_into_short_samples(
            valid_data,
            hyper_parameters["max_seq_len"],
            sliding_len=hyper_parameters["sliding_len"],
            encoder=config["encoder"])

    # In[ ]:

    print("train: {}".format(len(train_data)), "unlabeled train: {}".format(len(unlabeled_train_data)),
          "valid: {}".format(len(valid_data)))

    # # Tagger (Decoder)

    # In[ ]:

    max_seq_len = min(max_tok_num, hyper_parameters["max_seq_len"])
    rel2id = json.load(open(rel2id_path, "r", encoding="utf-8"))
    handshaking_tagger = HandshakingTaggingScheme(rel2id=rel2id,
                                                  max_seq_len=max_seq_len)

    # # Dataset

    # In[ ]:

    if config["encoder"] == "BERT":
        tokenizer = BertTokenizerFast.from_pretrained(config["bert_path"],
                                                      add_special_tokens=False,
                                                      do_lower_case=False)
        data_maker = DataMaker4Bert(tokenizer, handshaking_tagger)

    elif config["encoder"] in {
        "BiLSTM",
    }:
        token2idx_path = os.path.join(data_home, experiment_name,
                                      config["token2idx"])
        token2idx = json.load(open(token2idx_path, "r", encoding="utf-8"))
        idx2token = {idx: tok for tok, idx in token2idx.items()}


        def text2indices(text, max_seq_len):
            input_ids = []
            tokens = text.split(" ")
            for tok in tokens:
                if tok not in token2idx:
                    input_ids.append(token2idx['<UNK>'])
                else:
                    input_ids.append(token2idx[tok])
            if len(input_ids) < max_seq_len:
                input_ids.extend([token2idx['<PAD>']] *
                                 (max_seq_len - len(input_ids)))
            input_ids = torch.tensor(input_ids[:max_seq_len])
            return input_ids


        data_maker = DataMaker4BiLSTM(text2indices, get_tok2char_span_map,
                                      handshaking_tagger)

    # In[ ]:
    # 得到要输入的数据格式形式
    indexed_train_data = data_maker.get_indexed_data(train_data, max_seq_len)
    indexed_unlabeled_train_data = data_maker.get_indexed_data(unlabeled_train_data, max_seq_len)
    indexed_valid_data = data_maker.get_indexed_data(valid_data, max_seq_len)

    # In[ ]:

    train_dataloader = DataLoader(
        MyDataset(indexed_train_data),
        batch_size=hyper_parameters["batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=False,
        collate_fn=data_maker.generate_batch,
    )
    valid_dataloader = DataLoader(
        MyDataset(indexed_valid_data),
        batch_size=hyper_parameters["batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=False,
        collate_fn=data_maker.generate_batch,
    )

    # In[ ]:

    # # have a look at dataloader
    # train_data_iter = iter(train_dataloader)
    # batch_data = next(train_data_iter)
    # text_id_list, text_list, batch_input_ids, \
    # batch_attention_mask, batch_token_type_ids, \
    # offset_map_list, batch_ent_shaking_tag, \
    # batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = batch_data
    # print(text_list[0])
    # print()
    # print(tokenizer.decode(batch_input_ids[0].tolist()))
    # print(batch_input_ids.size())
    # print(batch_attention_mask.size())
    # print(batch_token_type_ids.size())
    # print(len(offset_map_list))
    # print(batch_ent_shaking_tag.size())
    # print(batch_head_rel_shaking_tag.size())
    # print(batch_tail_rel_shaking_tag.size())

    # # Model

    # In[ ]:
    # model loading
    if config["encoder"] == "BERT":
        encoder = AutoModel.from_pretrained(config["bert_path"])
        hidden_size = encoder.config.hidden_size
        fake_inputs = torch.zeros(
            [hyper_parameters["batch_size"], max_seq_len, hidden_size]).to(device)
        rel_extractor = TPLinkerBert(
            encoder,
            len(rel2id),
            hyper_parameters["shaking_type"],
            hyper_parameters["inner_enc_type"],
            hyper_parameters["dist_emb_size"],
            hyper_parameters["ent_add_dist"],
            hyper_parameters["rel_add_dist"],
        )
        if use_two_model:
            modelf2 = TPLinkerBert(
                encoder,
                len(rel2id),
                hyper_parameters["shaking_type"],
                hyper_parameters["inner_enc_type"],
                hyper_parameters["dist_emb_size"],
                hyper_parameters["ent_add_dist"],
                hyper_parameters["rel_add_dist"],
                dropout=config["two_models_hyper_parameters"]["student_dropout"],
                is_dropout=True
            )
            modelf2.to(device)
    elif config["encoder"] in {
        "BiLSTM",
    }:
        glove = Glove()
        glove = glove.load(config["pretrained_word_embedding_path"])
        # prepare embedding matrix
        word_embedding_init_matrix = np.random.normal(
            -1, 1, size=(len(token2idx), hyper_parameters["word_embedding_dim"]))
        count_in = 0
        # 在预训练词向量中的用该预训练向量
        # 不在预训练集里的用随机向量
        for ind, tok in tqdm(idx2token.items(),
                             desc="Embedding matrix initializing...", disable=config["disable_tqdm"]):
            if tok in glove.dictionary:
                count_in += 1
                word_embedding_init_matrix[ind] = glove.word_vectors[
                    glove.dictionary[tok]]

        print("{:.4f} tokens are in the pretrain word embedding matrix".format(
            count_in / len(idx2token)))  # 命中预训练词向量的比例
        word_embedding_init_matrix = torch.FloatTensor(word_embedding_init_matrix)

        fake_inputs = torch.zeros([
            hyper_parameters["batch_size"], max_seq_len,
            hyper_parameters["dec_hidden_size"]
        ]).to(device)
        rel_extractor = TPLinkerBiLSTM(
            word_embedding_init_matrix,
            hyper_parameters["emb_dropout"],
            hyper_parameters["enc_hidden_size"],
            hyper_parameters["dec_hidden_size"],
            hyper_parameters["rnn_dropout"],
            len(rel2id),
            hyper_parameters["shaking_type"],
            hyper_parameters["inner_enc_type"],
            hyper_parameters["dist_emb_size"],
            hyper_parameters["ent_add_dist"],
            hyper_parameters["rel_add_dist"],
        )
        if use_two_model:
            modelf2 = TPLinkerBiLSTM(
                word_embedding_init_matrix,
                hyper_parameters["emb_dropout"],
                hyper_parameters["enc_hidden_size"],
                hyper_parameters["dec_hidden_size"],
                hyper_parameters["rnn_dropout"],
                len(rel2id),
                hyper_parameters["shaking_type"],
                hyper_parameters["inner_enc_type"],
                hyper_parameters["dist_emb_size"],
                hyper_parameters["ent_add_dist"],
                hyper_parameters["rel_add_dist"],
                fc_dropout=config["two_models_hyper_parameters"]["student_dropout"],
                is_fc_dropout=True
            )
            modelf2.to(device)
    modelf1 = copy.deepcopy(rel_extractor)
    modelf1.to(device)


    # In[ ]:

    # all_paras = sum(x.numel() for x in rel_extractor.parameters())
    # enc_paras = sum(x.numel() for x in encoder.parameters())

    # In[ ]:

    # print(all_paras, enc_paras)
    # print(all_paras - enc_paras)

    # # Metrics

    # In[ ]:
    def bias_loss(weights=None):
        if weights is not None:
            weights = torch.FloatTensor(weights).to(device)
        cross_en = nn.CrossEntropyLoss(weight=weights)
        return lambda pred, target: cross_en(pred.view(-1,
                                                       pred.size()[-1]),
                                             target.view(-1))


    loss_func1 = bias_loss()
    loss_func2 = bias_loss()

    # In[ ]:
    metrics = MetricsCalculator(handshaking_tagger)

    # In[ ]:

    # optimizer
    init_learning_rate = float(hyper_parameters["lr"])
    optimizer1 = torch.optim.Adam(modelf1.parameters(), lr=init_learning_rate)
    if use_two_model:
        optimizer2 = torch.optim.Adam(modelf2.parameters(), lr=init_learning_rate,
                                      weight_decay=config["two_models_hyper_parameters"]["student_decay"])

    if hyper_parameters["scheduler"] == "CAWR":
        T_mult = hyper_parameters["T_mult"]
        rewarm_epoch_num = hyper_parameters["rewarm_epoch_num"]
        scheduler1 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer1,
            len(train_dataloader) * rewarm_epoch_num, T_mult)
        if use_two_model:
            scheduler2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer2,
                len(train_dataloader) * rewarm_epoch_num, T_mult)
    elif hyper_parameters["scheduler"] == "Step":
        decay_rate = hyper_parameters["decay_rate"]
        decay_steps = hyper_parameters["decay_steps"]
        scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1,
                                                     step_size=decay_steps,
                                                     gamma=decay_rate)
        if use_two_model:
            scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2,
                                                         step_size=decay_steps,
                                                         gamma=decay_rate)

    # In[ ]:

    # load existing model
    if not config["fr_scratch"]:
        model_state_path = config["model_state_dict_path"]
        modelf1.load_state_dict(torch.load(model_state_path), strict=False)
        if config["same_ts"]:
            modelf2.load_state_dict(torch.load(model_state_path), strict=False)
        if config["is_load_2"]:
            modelf2.load_state_dict(torch.load(config["student_model_state_dict_path"]), strict=False)
        print("------------model state {} loaded ----------------".format(
            model_state_path.split("/")[-1]))

    # parameters
    # 测试参数
    # BATCH_SIZE=32
    # LABEL_OF_TRAIN = 0.4  # Label ratio
    # FIRST_EPOCHS=2
    # TOTAL_EPOCHS = 1
    # META_EPOCHS = 2
    # 正常参数
    BATCH_SIZE = hyper_parameters["batch_size"]
    FIRST_EPOCHS = hyper_parameters["epochs"]
    SECOND_EPOCHS = hyper_parameters["student_epochs"]
    TOTAL_EPOCHS = hyper_parameters["TOTAL_EPOCHS"]
    # train_data_path = os.path.join(data_home, experiment_name,
    #                                config["train_data"])
    # if os.path.exists(os.path.join(data_home, experiment_name,"label.pkl")) and os.path.exists(os.path.join(data_home, experiment_name,"unlabel.pkl")):
    #     with open(os.path.join(data_home, experiment_name,"label.pkl"),"rb") as f1,open(os.path.join(data_home, experiment_name,"unlabel.pkl"),"rb") as f2:
    #         labeled_dataset, unlabeled_dataset_total=pickle.load(f1),pickle.load(f2)
    # else:
    #     labeled_dataset, unlabeled_dataset_total = stratified_dataset(MyDataset(indexed_train_data),LABEL_OF_TRAIN)
    #     with open(os.path.join(data_home, experiment_name, "label.pkl"), "wb") as f1, open(
    #             os.path.join(data_home, experiment_name, "unlabel.pkl"), "wb") as f2:
    #         pickle.dump(labeled_dataset,f1)
    #         pickle.dump(unlabeled_dataset_total,f2)

    # # build train dataloader
    # for i in range(META_EPOCHS):
    #     unlabeled_dataset_now, unlabeled_dataset_total = stratified_dataset(unlabeled_dataset_total,
    #                                                                        UNLABEL_OF_TRAIN / META_EPOCHS)
    #     unlabeled_dataset.append(unlabeled_dataset_now)

    # Create the DataLoaders for our label and unlabel sets.
    labeled_dataloader = DataLoader(
        MyDataset(indexed_train_data),
        batch_size=hyper_parameters["batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=False,
        collate_fn=data_maker.generate_batch,
    )
    if config["training_method"] == "self-training":
        unlabeled_dataloader_all = DataLoader(
            MyDataset(indexed_unlabeled_train_data),  # The training samples.
            batch_size=hyper_parameters["batch_size"],
            shuffle=False,
            num_workers=0,
            drop_last=False,
            collate_fn=functools.partial(data_maker.generate_batch, data_type="pseudo_training"),
        )
    elif config["training_method"] == "increment-training":
        META_EPOCHS = 4
        unlabeled_dataset = random_split(MyDataset(indexed_unlabeled_train_data),
                                         split_sample(MyDataset(indexed_unlabeled_train_data), n_part=META_EPOCHS))
        unlabeled_dataloader_all = []
        for i in range(META_EPOCHS):
            unlabeled_dataloader_now = DataLoader(
                unlabeled_dataset[i],  # The training samples.
                batch_size=hyper_parameters["batch_size"],
                shuffle=True,
                num_workers=0,
                drop_last=False,
                collate_fn=functools.partial(data_maker.generate_batch, data_type="pseudo_training"),
            )
            unlabeled_dataloader_all.append(unlabeled_dataloader_now)

    if test_config["only_test"]:
        if use_two_model:
            test_step(modelf1, modelf2, save_res_dir, max_test_seq_len)
        else:
            test_step(modelf1, None, save_res_dir, max_test_seq_len)
    elif config["training_method"] == "self-training":
        self_training()
    elif config["training_method"] == "increment-training":
        increment_training()
    # ----------------------training complete-----------------------

    print("Training complete!")

# print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

# def excellent_calulate(metrics,fine_map,data):
#     # 算法要求连续产生一样的结果才行
#     """
#     Use data to calculate, then store data.
#     """
#     for one_data in data:
#         idx = one_data[0]['id'].split("_")[1]
#         if idx in fine_map:
#             last_score,last_data = fine_map[idx]
#             score = metrics.get_pseudo_rel_cpg(one_data[0], one_data[4],
#                                                last_data[6],one_data[6],
#                                                pattern="whole_text")
#             if score>=0.9:
#                 if score>last_score:
#                     fine_map[idx] = (score, one_data)
#         else:
#             fine_map[idx] = (0,one_data)
#
# def exct_extract(fine_map):
#     fine_map=dict(sorted(fine_map.items(), key=lambda k_v: k_v[1][0], reverse=True))
#     exct_list=[]
#     for values in fine_map.values():
#         if values[0]>0.95:
#             exct_list.append(values[1])#只将data添加进去
#         else:
#             break
#     return exct_list
