

## iterative update
def split_sample(dataset,n_part):
    """
    Given pytorch dataset
    return index of n_part
    """
    factor=len(dataset)//n_part
    res=len(dataset)%n_part
    return [factor + int((factor * (i + 1) + res) // len(dataset)) * res for i in range(n_part)]


def stratified_sample(dataset, ratio, display=False):
    """
    Given pytorch dataset and ratio of stratified.
    return subset dataset
    """
    import collections
    from torch.utils.data import  Subset
    import random
    data_dict = collections.defaultdict(list)
    for i in range(len(dataset)):
        j=dataset[i][-1][1]#每个例子中每个类加入一个#every sentence multi relations
        if len(j) != 0:
            rel_record=[]
            for every_rel_ins in j:
                rel_id=every_rel_ins[0]
                if rel_id not in rel_record:
                    data_dict[rel_id].append(i)# data_dict[label]+=train_index
                    rel_record.append(rel_id)
        else:
            data_dict[-1].append(i)# 对应无tuple那种关系

    sampled_indices = []
    rest_indices = []


    if display:
        plot_samples=[]
        plot_rest=[]
        for rel_id,indices in data_dict.items():
            random.shuffle(indices)
            index = int(len(indices) * ratio + 1)
            sampled_indices += indices[:index]
            plot_samples.append(len(indices[:index]))
            rest_indices += indices[index:]
            plot_rest.append(len(indices[index:]))
        import matplotlib.pyplot as plt
        plt.subplot(212)
        plt.bar(data_dict.keys(), [len(x) for x in data_dict.values()])
        plt.title("origin training data")
        plt.subplot(221)
        plt.bar(data_dict.keys(), plot_samples)
        plt.title("labeled data")
        plt.subplot(222)
        plt.bar(data_dict.keys(), plot_rest)
        plt.title("unlabeled data")
        plt.show()
    else:
        for indices in data_dict.values():
            random.shuffle(indices)
            index = int(len(indices) * ratio + 1)
            sampled_indices += indices[:index]
            rest_indices += indices[index:]
    return [Subset(dataset, sampled_indices), Subset(dataset, rest_indices)]

def filter_duplicates(rel_list):
    rel_memory_set = set()
    filtered_rel_list = []
    for rel in rel_list:
        rel_memory = "{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0],
                                                                 rel["subj_tok_span"][1],
                                                                 rel["predicate"],
                                                                 rel["obj_tok_span"][0],
                                                                 rel["obj_tok_span"][1])
        if rel_memory not in rel_memory_set:
            filtered_rel_list.append(rel)
            rel_memory_set.add(rel_memory)
    return filtered_rel_list


## utilis
# Takes a time in seconds and returns a string hh:mm:ss
def format_time(elapsed):
    import datetime
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

## utils
def save_model(current,last,save_path,current_model):
    import torch
    if current > last:
        last = current
        torch.save(
                current_model.state_dict(),
                save_path)
        print("Best model save with: {} in save path:{}".format(last,save_path))
    return last

def batch2dataset(*args):
    """
    将batch数据解析成需要的格式
    (sample,input_ids,attention_mask,token_type_ids,tok2char_span,spots_tuple,)
    """
    a = []
    for i in zip(*args):
        a.append(i)
    return a