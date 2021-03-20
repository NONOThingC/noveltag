#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import gc
import json
import os
import warnings

import functools
import objgraph
from tqdm import tqdm
import re

from pprint import pprint
from transformers import AutoModel, BertTokenizerFast
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset,random_split
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
import torch.nn.functional as F
import random
import numpy as np
import os
import time, json
import datetime
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, random_split, Subset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import f1_score
from collections import Counter
import sys
import pickle
import collections
# In[ ]:

# try:
#     from yaml import CLoader as Loader, CDumper as Dumper
# except ImportError:
#     from yaml import Loader, Dumper
# config = yaml.load(open("train_config.yaml", "r"), Loader = yaml.FullLoader)

# Setting
sys.setrecursionlimit(10000)
config = config.train_config
hyper_parameters = config["hyper_parameters"]
# Gpu set
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = str(config["device_num"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# for reproductivity
np.random.seed(hyper_parameters["seed"])
torch.manual_seed(hyper_parameters["seed"])  # pytorch random seed
torch.backends.cudnn.deterministic = True

# In[ ]:

data_home = config["data_home"]
experiment_name = config["exp_name"]
train_data_path = os.path.join(data_home, experiment_name,
                               config["train_data"])
valid_data_path = os.path.join(data_home, experiment_name,
                               config["valid_data"])
rel2id_path = os.path.join(data_home, experiment_name, config["rel2id"])

# In[ ]:

# if config["logger"] == "wandb":
#     # init wandb
#     wandb.init(
#         project=experiment_name,
#         name=config["run_name"],
#         config=hyper_parameters  # Initialize config
#     )
#
#     wandb.config.note = config["note"]
#
#     model_state_dict_dir = wandb.run.dir
#     logger = wandb
# else:
logger = DefaultLogger(config["log_path"], experiment_name,
                       config["run_name"], config["run_id"],
                       hyper_parameters)
model_state_dict_dir = config["path_to_save_model"]
if not os.path.exists(model_state_dict_dir):
    os.makedirs(model_state_dict_dir)
logger = DefaultLogger(config["log_path"], experiment_name,
                       config["run_name"], config["run_id"],
                       hyper_parameters)
model_state_dict_dir = config["path_to_save_model"]
if not os.path.exists(model_state_dict_dir):
    os.makedirs(model_state_dict_dir)
# # Load Data

# In[ ]:

train_data = json.load(open(train_data_path, "r", encoding="utf-8"))
valid_data = json.load(open(valid_data_path, "r", encoding="utf-8"))
# with open(train_data_path+'1',"w",encoding="utf-8") as f:
#     f.write(json.dumps(train_data[:1000]))
# with open(valid_data_path+'1',"w",encoding="utf-8") as f:
#     f.write(json.dumps(valid_data[:200]))
# # Split

# In[ ]:

# @specific
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


# In[ ]:

preprocessor = Preprocessor(tokenize_func=tokenize,
                            get_tok2char_span_map_func=get_tok2char_span_map)

# In[ ]:

# train and valid max token num
max_tok_num = 0
all_data = train_data + valid_data

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
    valid_data = preprocessor.split_into_short_samples(
        valid_data,
        hyper_parameters["max_seq_len"],
        sliding_len=hyper_parameters["sliding_len"],
        encoder=config["encoder"])

# In[ ]:

print("train: {}".format(len(train_data)), "valid: {}".format(len(valid_data)))

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
class PseudoDataset(Dataset):

    def __init__(self, filepath):
        number = 0
        with open(filepath, "r") as f:
            # 获得训练数据的总行数
            for _ in tqdm(f, desc="load training dataset"):
                number += 1
        self.number = number
        self.fopen = open(filepath, 'r')

    def __len__(self):
        return self.number

    def __getitem__(self, index):
        line = self.fopen.__next__()
        # 自定义transform()对训练数据进行预处理
        return line

    def __getitem__(self, index):

        return self.data[index]

    def __len__(self):
        return len(self.data)

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class MultiFileDataset(Dataset):
    r"""Dataset as a concatenation of multiple file datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
        datasets: List[Dataset[T_co]]
    cumulative_sizes: List[int]
    """
    @staticmethod
    def cumsum_dataset(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    @staticmethod
    def cumsum_num(nums):
        r, s = [], 0
        for num in nums:
            s+=num
            r.append(s)
        return r


    def read_file(self,filename):
        """
        intelligent read
        """
        if self.current_filename=="":
            self.file_io=open("Trunk" + str(filename) + ".pkl", 'rb')
        if self.current_filename != filename:
            self.file_io.close()
            self.file_io=open("Trunk" + str(filename) + ".pkl", 'rb')
        self.current_filename = filename
        return pickle.load(self.file_io)

    def __init__(self, trunk_sizes) -> None:

        self.cumulative_sizes = self.cumsum_num(trunk_sizes)
        self.trunk_count=len(self.cumulative_sizes)
        self.current_filename=''

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        import bisect
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        return self.read_file(dataset_idx)[sample_idx]


    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes


# In[ ]:
# 得到要输入的数据格式形式
indexed_train_data = data_maker.get_indexed_data(train_data, max_seq_len)
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
                         desc="Embedding matrix initializing..."):
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

# two model
modelf1=copy.deepcopy(rel_extractor)
modelf2 = copy.deepcopy(rel_extractor)
modelf1.to(device)
modelf2.to(device)

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

## utils
def save_model(current,best,save_path,model,mode):
    if mode=="best":
        if current >= best:
            best = current
            torch.save(
                    model.state_dict(),
                    os.path.join(
                        model_state_dict_dir,
                        "model_state_dict_best.pt"))
    return best






# In[ ]:

# optimizer
init_learning_rate = float(hyper_parameters["lr"])
optimizer1 = torch.optim.Adam(modelf1.parameters(), lr=init_learning_rate)
optimizer2 = torch.optim.Adam(modelf2.parameters(), lr=init_learning_rate)

if hyper_parameters["scheduler"] == "CAWR":
    T_mult = hyper_parameters["T_mult"]
    rewarm_epoch_num = hyper_parameters["rewarm_epoch_num"]
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer1,
        len(train_dataloader) * rewarm_epoch_num, T_mult)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer2,
        len(train_dataloader) * rewarm_epoch_num, T_mult)
elif hyper_parameters["scheduler"] == "Step":
    decay_rate = hyper_parameters["decay_rate"]
    decay_steps = hyper_parameters["decay_steps"]
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1,
                                                step_size=decay_steps,
                                                gamma=decay_rate)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2,
                                                step_size=decay_steps,
                                                gamma=decay_rate)

# In[ ]:
## utilis
# Takes a time in seconds and returns a string hh:mm:ss
def format_time(elapsed):
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def move_list_to_device(input,device,**kwargs):
    return [i.to(device) for i in input]

## 需要更改
if not config["fr_scratch"]:
    model_state_path = config["model_state_dict_path"]
    rel_extractor.load_state_dict(torch.load(model_state_path))
    print("------------model state {} loaded ----------------".format(
        model_state_path.split("/")[-1]))



## iterative update
def split_sample(dataset,n_part):
    """
    return n_part
    """
    factor=len(dataset)//n_part
    res=len(dataset)%n_part
    return [factor + int((factor * (i + 1) + res) // len(dataset)) * res for i in range(n_part)]


def stratified_sample(dataset, ratio):
    """
    根据样本中数据的比例来采样
    """
    import collections
    data_dict = collections.defaultdict(list)
    for i in range(len(dataset)):
        j=dataset[i][-1][1]#每个例子中每个类加入一个
        if len(j) != 0:
            rel_record=[]
            for every_rel_ins in j:
                rel_id=every_rel_ins[0]
                if rel_id not in rel_record:
                    data_dict[rel_id].append(i)# data_dict[label]+=train_index
                    rel_record.append(rel_id)
        else:
            data_dict[-1].append(i)

    sampled_indices = []
    rest_indices = []
    for indices in data_dict.values():
        random.shuffle(indices)
        index=int(len(indices) * ratio+1)
        sampled_indices += indices[:index]
        rest_indices += indices[index:]
    return [Subset(dataset, sampled_indices), Subset(dataset, rest_indices)]


# parameters
# 测试参数
# BATCH_SIZE=32
# LABEL_OF_TRAIN = 0.4  # Label ratio
# FIRST_EPOCHS=2
# TOTAL_EPOCHS = 1
# MATE_EPOCHS = 2
# seed_val = 19
# LAMBD = 0.2
# 正常参数
BATCH_SIZE=16
LABEL_OF_TRAIN = 0.1  # Label ratio
FIRST_EPOCHS= 5
TOTAL_EPOCHS = 10
seed_val = 19
LAMBD = 0.2
# stratified data
labeled_dataset, unlabeled_dataset_total = stratified_sample(MyDataset(indexed_train_data),LABEL_OF_TRAIN)

# build train dataloader
# for i in range(MATE_EPOCHS):
#     unlabeled_dataset_now, unlabeled_dataset_total = stratified_sample(unlabeled_dataset_total,
#                                                                        UNLABEL_OF_TRAIN / MATE_EPOCHS)
#     unlabeled_dataset.append(unlabeled_dataset_now)

# Create the DataLoaders for our label and unlabel sets.
labeled_dataloader = DataLoader(
    labeled_dataset,
    batch_size=hyper_parameters["batch_size"],
    shuffle=True,
    num_workers=0,
    drop_last=False,
    collate_fn=data_maker.generate_batch,
)

unlabeled_dataloader_all = DataLoader(
    unlabeled_dataset_total, # The training samples.
    batch_size=hyper_parameters["batch_size"],
    shuffle=False,
    num_workers=0,
    drop_last=False,
    collate_fn=functools.partial(data_maker.generate_batch,data_type="pseudo_training"),
)

indexed_valid_data = data_maker.get_indexed_data(valid_data, max_seq_len)

# build valid dataloader
valid_dataloader = DataLoader(
    MyDataset(indexed_valid_data),
    batch_size=hyper_parameters["batch_size"],
    shuffle=True,
    num_workers=0,
    drop_last=False,
    collate_fn=data_maker.generate_batch,
)

# build
def batch2dataset(*args):
    """
    将batch数据解析成需要的格式
    (sample,input_ids,attention_mask,token_type_ids,tok2char_span,spots_tuple,)
    """
    a = []
    for i in zip(*args):
        a.append(i)
    return a

def excellent_calulate(metrics,fine_map,data):
    # 算法要求连续产生一样的结果才行
    """
    Use data to calculate, then store data.
    """
    for one_data in data:
        idx = one_data[0]['id'].split("_")[1]
        if idx in fine_map:
            last_score,last_data = fine_map[idx]
            score = metrics.get_pseudo_rel_cpg(one_data[0], one_data[4],
                                               last_data[6],one_data[6],
                                               pattern="whole_text")
            if score>=0.9:
                if score>last_score:
                    fine_map[idx] = (score, one_data)
        else:
            fine_map[idx] = (0,one_data)

def exct_extract(fine_map):
    fine_map=dict(sorted(fine_map.items(), key=lambda k_v: k_v[1][0], reverse=True))
    exct_list=[]
    for values in fine_map.values():
        if values[0]>0.95:
            exct_list.append(values[1])#只将data添加进去
        else:
            break
    return exct_list

def w_rel_seq_score(pre_seq,seq_val_acc):
    """
    first entity to a baseline,
    then focus on relations.

    """
    enh_rate =2#系数提升几倍
    ent_rate=6*(1-seq_val_acc[0])
    rel_rate=6*seq_val_acc[0]
    if seq_val_acc[0]<0.6:
        for shaking_outputs in model_output:
            pred_weight, label = torch.max(shaking_outputs, dim=-1)
            pred_weight=pred_weight+torch.mul(enh_rate*label,pred_weight)
            pred_weights.append(pred_weight)
            sequence_weight = torch.mean(pred_weight, dim=-1)
            if len(sequence_weight.shape) == 2:  # entity\head\rel的分数综合考虑
                sequence_weight = torch.mean(sequence_weight, dim=-1)
            sequence_weights.append(sequence_weight)
    else:
        for shaking_outputs in model_output:
            pred_weight, label = torch.max(shaking_outputs, dim=-1)
            if len(pred_weight.shape) == 3:
                pred_weight = pred_weight + torch.mul(rel_rate * label, pred_weight)
            else:
                pred_weight = pred_weight + torch.mul(ent_rate * label, pred_weight)
            pred_weights.append(pred_weight)
            sequence_weight = torch.mean(pred_weight, dim=-1)
            if len(sequence_weight.shape) == 2:  # entity\head\rel的分数综合考虑
                sequence_weight = torch.mean(sequence_weight, dim=-1)
            sequence_weights.append(sequence_weight)




print("training start...\n")
cnt = 0
"""
two aspect:
1. data handle:dataloader
2. model,loss handle
"""
# count how many rounds the whole big model has to train
# a round means all unlabel data are labeled

train_dataloader = labeled_dataloader
fine_map = collections.defaultdict(tuple)  # key:train id ; value:correct number
for total_epoch in range(TOTAL_EPOCHS):
    # Add an inner loop to judge the entire unlabeled data set finished

    print(f"Total epoch{total_epoch}:\n")
    for ep in range(FIRST_EPOCHS):
        ## train
        modelf1.train()
        t_ep = time.time()
        start_lr = optimizer1.param_groups[0]['lr']
        total_loss, total_ent_sample_acc, total_head_rel_sample_acc, total_tail_rel_sample_acc = 0., 0., 0., 0.
        for batch_ind, batch_train_data in enumerate(train_dataloader):
            t_batch = time.time()
            z = (2 * len(rel2id) + 1)
            steps_per_ep = len(train_dataloader)
            total_steps = hyper_parameters[ "loss_weight_recover_steps"] + 1  # + 1 avoid division by zero error
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

            # modelf1 forward
            # zero the parameter gradients
            optimizer1.zero_grad()
            if config["encoder"] == "BERT":
                ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs = modelf1(
                    batch_input_ids,
                    batch_attention_mask,
                    batch_token_type_ids,
                )
            elif config["encoder"] in {
                "BiLSTM",
            }:
                ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs = modelf1(
                    batch_input_ids)

            w_ent, w_rel = loss_weights["ent"], loss_weights["rel"]
            loss1 = w_ent * loss_func1(
                ent_shaking_outputs, batch_ent_shaking_tag) + w_rel * loss_func1(
                head_rel_shaking_outputs,
                batch_head_rel_shaking_tag) + w_rel * loss_func1(
                tail_rel_shaking_outputs, batch_tail_rel_shaking_tag)

            loss1.backward()
            optimizer1.step()

            ent_sample_acc = metrics.get_sample_accuracy(ent_shaking_outputs,
                                                         batch_ent_shaking_tag)
            head_rel_sample_acc = metrics.get_sample_accuracy(
                head_rel_shaking_outputs, batch_head_rel_shaking_tag)
            tail_rel_sample_acc = metrics.get_sample_accuracy(
                tail_rel_shaking_outputs, batch_tail_rel_shaking_tag)

            loss1, ent_sample_acc, head_rel_sample_acc, tail_rel_sample_acc = loss1.item(), ent_sample_acc.item(), head_rel_sample_acc.item(
            ), tail_rel_sample_acc.item()
            scheduler1.step()

            total_loss += loss1
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
                optimizer1.param_groups[0]['lr'],
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
                "learning_rate": optimizer1.param_groups[0]['lr'],
                "time": time.time() - t_ep,
            })

        if config[
            "logger"] != "wandb":  # only log once for training if logger is not wandb
            logger.log({
                "train_loss": avg_loss,
                "train_ent_seq_acc": avg_ent_sample_acc,
                "train_head_rel_acc": avg_head_rel_sample_acc,
                "train_tail_rel_acc": avg_tail_rel_sample_acc,
                "learning_rate": optimizer1.param_groups[0]['lr'],
                "time": time.time() - t_ep,
            })

        ## valid
        modelf1.eval()
        t_ep = time.time()
        total_ent_sample_acc, total_head_rel_sample_acc, total_tail_rel_sample_acc = 0., 0., 0.
        total_rel_correct_num, total_rel_pred_num, total_rel_gold_num = 0, 0, 0
        for batch_ind, batch_valid_data in enumerate(tqdm(valid_dataloader, desc="Validating")):
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
                    ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs = modelf1(
                        batch_input_ids,
                        batch_attention_mask,
                        batch_token_type_ids,
                    )
                elif config["encoder"] in {
                    "BiLSTM",
                }:
                    ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs = modelf1(
                        batch_input_ids)

            ent_sample_acc = metrics.get_sample_accuracy(ent_shaking_outputs,
                                                         batch_ent_shaking_tag)
            head_rel_sample_acc = metrics.get_sample_accuracy(
                head_rel_shaking_outputs, batch_head_rel_shaking_tag)
            tail_rel_sample_acc = metrics.get_sample_accuracy(
                tail_rel_shaking_outputs, batch_tail_rel_shaking_tag)


            ent_sample_acc, head_rel_sample_acc, tail_rel_sample_acc= ent_sample_acc.item(), head_rel_sample_acc.item(
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
            "val_ent_seq_acc": avg_ent_sample_acc,
            "val_head_rel_acc": avg_head_rel_sample_acc,
            "val_tail_rel_acc": avg_tail_rel_sample_acc,
            "time": time.time() - t_ep,
        }
        logger.log(log_dict)
        pprint(log_dict)

    seq_val_acc=(avg_ent_sample_acc,avg_head_rel_sample_acc,avg_tail_rel_sample_acc)
        # valid_f1 = rel_prf[2]

        ## save when better
        # if valid_f1 >= max_f1:
        #     max_f1 = valid_f1
        #     if valid_f1 > config["f1_2_save"]:  # save the best model
        #         modle_state_num = len(
        #             glob.glob(model_state_dict_dir + "/model_state_dict_*.pt"))
        #         torch.save(
        #             model.state_dict(),
        #             os.path.join(
        #                 model_state_dict_dir,
        #                 "model_state_dict_{}.pt".format(modle_state_num)))

        # save best
        # if valid_f1 >= max_f1:
        #     max_f1 = valid_f1
        #     if valid_f1 > config["f1_2_save"]:  # save the best model
        #         torch.save(
        #             modelf1.state_dict(),
        #             os.path.join(
        #                 model_state_dict_dir,
        #                 "model_state_dict_best.pt"))
    #                 scheduler_state_num = len(glob.glob(schedule_state_dict_dir + "/scheduler_state_dict_*.pt"))
    #                 torch.save(scheduler.state_dict(), os.path.join(schedule_state_dict_dir, "scheduler_state_dict_{}.pt".format(scheduler_state_num)))    print("Current avf_f1: {}, Best f1: {}".format(valid_f1, max_f1))
    # -------train f1 end---------
    # -------generate pseudo label---------
    #
    print("generate pseudo label\n")
    Z = 12  # Incremental Epoch Number
    Z_RATIO = (1-(sum(seq_val_acc)//3)-0.1)*BATCH_SIZE
    ## valid
    modelf1.eval()
    t_ep = time.time()
    all_gold_labels = []
    all_pred_labels = []

    pseudo_count = 0
    batch_new_data_list=[]
    # 注意，这里更新了train_loader
    train_dataloader = labeled_dataloader
    trunk_count=0
    size_count=[]
    results = []
    for batch_ind, batch_valid_data in enumerate(tqdm(unlabeled_dataloader_all, desc="Validating")):

        if config["encoder"] == "BERT":
            sample_list, batch_input_ids, batch_attention_mask, batch_token_type_ids, tok2char_span_list,matrix_spots_list, batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = batch_valid_data
            batch_input_ids, batch_attention_mask, batch_token_type_ids, batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = (
                batch_input_ids.to(device), batch_attention_mask.to(device),
                batch_token_type_ids.to(device), batch_ent_shaking_tag.to(device),
                batch_head_rel_shaking_tag.to(device),
                batch_tail_rel_shaking_tag.to(device))
        elif config["encoder"] in {
            "BiLSTM",
        }:
            sample_list, batch_input_ids, tok2char_span_list,matrix_spots_list, batch_ent_shaking_tag, batch_head_rel_shaking_tag, batch_tail_rel_shaking_tag = batch_valid_data
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
            elif config["encoder"] in {
                "BiLSTM",
            }:
                ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs = modelf1(
                    batch_input_ids)

            batch_pred_ent_shaking_tag = torch.argmax(ent_shaking_outputs, dim=-1)
            batch_pred_head_rel_shaking_tag = torch.argmax(head_rel_shaking_outputs, dim=-1)
            batch_pred_tail_rel_shaking_tag = torch.argmax(tail_rel_shaking_outputs, dim=-1)

            # 得到hand shaking结果
            ## 反思，为啥不用循环呢？
            # 三个都选按出来以后label的交集呢？ 因为要考虑到这种时候可能会存在着交集为空集以及变长的每次选择序列数目
            pred_weights = []
            labels = []
            sequence_weights = []
            sort_indexs = []
            # fine_indexs= []
            model_output=[ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs]
            enh_rate = 2  # 系数提升几倍
            ent_rate = 10 * (1 - seq_val_acc[0])
            rel_rate = 10 * seq_val_acc[0] # From valid step
            if seq_val_acc[0] < 0.6:
                for shaking_outputs in model_output:
                    pred_weight, label = torch.max(shaking_outputs, dim=-1)
                    pred_weight = pred_weight + torch.mul(enh_rate * label, pred_weight)
                    pred_weights.append(pred_weight)
                    sequence_weight = torch.mean(pred_weight, dim=-1)
                    if len(sequence_weight.shape) == 2:  # entity\head\rel的分数综合考虑
                        sequence_weight = torch.mean(sequence_weight, dim=-1)
                    sequence_weights.append(sequence_weight)
            else:
                for shaking_outputs in model_output:
                    pred_weight, label = torch.max(shaking_outputs, dim=-1)
                    if len(pred_weight.shape) == 3:
                        pred_weight = pred_weight + torch.mul(rel_rate * label, pred_weight)
                    else:
                        pred_weight = pred_weight + torch.mul(ent_rate * label, pred_weight)
                    pred_weights.append(pred_weight)
                    sequence_weight = torch.mean(pred_weight, dim=-1)
                    if len(sequence_weight.shape) == 2:  # entity\head\rel的分数综合考虑
                        sequence_weight = torch.mean(sequence_weight, dim=-1)
                    sequence_weights.append(sequence_weight)

                # fine_indexs.append(set(fine_index.tolist()))

            # # Pseudo Label Selection, top Z%

            final_sequence_weight = sum(sequence_weights)
            final_sort = torch.argsort(final_sequence_weight, descending=True)
            final_sort = final_sort[0:int(len(final_sort) * Z_RATIO) if int(len(final_sort) * Z_RATIO)>0 else 1]
            inter_index = final_sort
            sort_input=[]
            for var in batch_valid_data:
                if isinstance(var, torch.Tensor):
                    sort_input.append(var[inter_index].to("cpu"))
                elif isinstance(var, list):
                    sort_input.append([var[i] for i in inter_index])
                else:
                    raise Exception
            model_output=[var[inter_index].to("cpu") for var in model_output]
            gold_labels = sort_input[-3:]
            pseudo_labels = [label[inter_index] for label in labels]
            pseudo_count += len(inter_index)
            #此操作耗时，得到序号inter_index再解析
            sorted_spots_list = []

            for i in inter_index:
                sorted_spots_list.append((handshaking_tagger.get_sharing_spots_fr_shaking_tag(batch_pred_ent_shaking_tag[i]),handshaking_tagger.get_spots_fr_shaking_tag(batch_pred_head_rel_shaking_tag[i]),handshaking_tagger.get_spots_fr_shaking_tag(batch_pred_tail_rel_shaking_tag[i])))
                # sorted_spots_list.append(matrix_spots_list[i])
            for i in range(3):
                results.append(metrics.get_sample_accuracy(model_output[i], gold_labels[i]).item())
            print("Pseudo label acc is:{}".format(np.mean(results)))
            # pred_id = torch.argmax(pred, dim=-1)
            # # (batch_size, ..., seq_len) -> (batch_size, )，把每个sample压成一条seq
            # pred_id = pred_id.view(pred_id.size()[0], -1)
            # record id and corresponding label
            # update training data


            batch_new_data=batch2dataset(*(sort_input[:-3]+[sorted_spots_list]))#sorted_spots_list is pseudo label
            batch_new_data_list.extend(batch_new_data)
            # excellent_calulate(metrics, fine_map, batch_new_data)


            # if total_epoch != TOTAL_EPOCHS-1:
            #     batch_new_data=batch2dataset(*(sort_input[:-2]+pseudo_labels))#-2 because placeholder,not -3
            #     batch_new_data_list.extend(batch_new_data)


            # if len(batch_new_data_list)>trunk_size:#1000大概20~30GB
            #     # 文件流操作
            #     print("Trunk {}".format(trunk_count))
            #     with open("Trunk" + str(trunk_count) + ".pkl", "wb") as f:
            #         pickle.dump(batch_new_data_list, f)
            #     trunk_count+=1
            #     size_count.append(len(batch_new_data_list))
            #     batch_new_data=0
            #     train_add_dataset=0
            #     batch_new_data_list = []
            #     print("Trunk end")


            # 给标记成合适的数据格式
            # 数据筛选(三个维度应采用一个指标筛选进来)

    log_dict = {
                "use pseudo number": pseudo_count,
                "pseudo acc":np.mean(results),
               "time": time.time() - t_ep,
    }
    logger.log(log_dict)
    pprint(log_dict)
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
    train_add_dataset = train_dataloader.dataset + MyDataset(batch_new_data_list)
    train_dataloader = DataLoader(
        train_add_dataset,  # The training samples.
        batch_size=hyper_parameters["batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=False,
        collate_fn=data_maker.generate_batch,
    )
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
    batch_new_data_list=[]
    torch.cuda.empty_cache()
    ## valid
    if total_epoch==TOTAL_EPOCHS-1:
        modelf1.eval()
        t_ep = time.time()
        total_ent_sample_acc, total_head_rel_sample_acc, total_tail_rel_sample_acc = 0., 0., 0.
        total_rel_correct_num, total_rel_pred_num, total_rel_gold_num = 0, 0, 0
        for batch_ind, batch_valid_data in enumerate(tqdm(valid_dataloader, desc="Validating")):
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
                    ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs = modelf1(
                        batch_input_ids,
                        batch_attention_mask,
                        batch_token_type_ids,
                    )
                elif config["encoder"] in {
                    "BiLSTM",
                }:
                    ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs = modelf1(
                        batch_input_ids)

            ent_sample_acc = metrics.get_sample_accuracy(ent_shaking_outputs,
                                                         batch_ent_shaking_tag)
            head_rel_sample_acc = metrics.get_sample_accuracy(
                head_rel_shaking_outputs, batch_head_rel_shaking_tag)
            tail_rel_sample_acc = metrics.get_sample_accuracy(
                tail_rel_shaking_outputs, batch_tail_rel_shaking_tag)

            rel_cpg = metrics.get_rel_cpg(sample_list, tok2char_span_list,
                                          ent_shaking_outputs,
                                          head_rel_shaking_outputs,
                                          tail_rel_shaking_outputs,
                                          hyper_parameters["match_pattern"])

            ent_sample_acc, head_rel_sample_acc, tail_rel_sample_acc, rel_cpg =  ent_sample_acc.item(), head_rel_sample_acc.item(
            ), tail_rel_sample_acc.item(), rel_cpg


            total_ent_sample_acc += ent_sample_acc
            total_head_rel_sample_acc += head_rel_sample_acc
            total_tail_rel_sample_acc += tail_rel_sample_acc

            total_rel_correct_num += rel_cpg[0]
            total_rel_pred_num += rel_cpg[1]
            total_rel_gold_num += rel_cpg[2]

        avg_ent_sample_acc = total_ent_sample_acc / len(valid_dataloader)
        avg_head_rel_sample_acc = total_head_rel_sample_acc / len(valid_dataloader)
        avg_tail_rel_sample_acc = total_tail_rel_sample_acc / len(valid_dataloader)

        rel_prf = metrics.get_prf_scores(total_rel_correct_num,
                                         total_rel_pred_num,
                                         total_rel_gold_num)

        log_dict = {
            "val_ent_seq_acc": avg_ent_sample_acc,
            "val_head_rel_acc": avg_head_rel_sample_acc,
            "val_tail_rel_acc": avg_tail_rel_sample_acc,
            "val_prec": rel_prf[0],
            "val_recall": rel_prf[1],
            "val_f1": rel_prf[2],
            "time": time.time() - t_ep,
        }
        logger.log(log_dict)
        pprint(log_dict)

        valid_f1 = rel_prf[2]

# ----------------------training complete-----------------------

print("Training complete!")
# print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))