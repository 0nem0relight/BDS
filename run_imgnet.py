import os
import faiss
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances
import numpy as np
from util.args_loader import get_args
from util import metrics
import torch
import scipy.special
from OOD_detector import *


args = get_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
load_tensorboard = False

ood_dataset_size = {
    'inat': 10000,
    'sun50': 10000,
    'places50': 10000,
    'dtd': 5640,
}
class_num = 1000
normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
prob_normalizer=lambda x: x/ np.sum(x,axis=1,keepdims=True)
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def kl_divergence(p, q):
    return np.sum(p * np.log(p / q),axis=1)

id_train_size = 1281167
id_val_size = 50000

FORCE_RUN = False
norm_cache = f"cache/{args.in_dataset}_train_{args.name}_in/feat_norm.mmap"
cache_dir = f"cache/{args.in_dataset}_train_{args.name}_in"
if not FORCE_RUN and os.path.exists(norm_cache):
    norm_feat_train = np.memmap(norm_cache, dtype=float, mode='r', shape=(id_train_size, 2048))
    logit_train = np.memmap(f"{cache_dir}/score.mmap", dtype=float, mode='r', shape=(id_train_size, class_num))
    label_train = np.memmap(f"{cache_dir}/label.mmap", dtype=float, mode='r', shape=(id_train_size,))
else:
    norm_feat_train = np.memmap(norm_cache, dtype=float, mode='w+', shape=(id_train_size, 2048))
    feat_train = np.memmap(f"{cache_dir}/feat.mmap", dtype=float, mode='r', shape=(id_train_size, 2048))
    norm_feat_train[:] = normalizer(feat_train)

pred_train = np.argmax(logit_train, axis=1)
index = np.where(label_train == pred_train)
filtered_feat = norm_feat_train[index]
filtered_pred = pred_train[index]
filtered_feat, filtered_pred = np.array(filtered_feat), np.array(filtered_pred)
cat_centre,class_feat = [None for _ in range(class_num)],[None for _ in range(class_num)]
for cls in range(class_num):
    index=(filtered_pred == cls)
    class_feat[cls] = filtered_feat[index]
    cat_centre[cls]=class_feat[cls].mean(0)
cat_centre=np.array(cat_centre)
del norm_feat_train

norm_cache = f"cache/{args.in_dataset}_val_{args.name}_in/feat_norm.mmap"
cache_dir = f"cache/{args.in_dataset}_val_{args.name}_in"
if not FORCE_RUN and os.path.exists(norm_cache):
    norm_feat_id = np.memmap(norm_cache, dtype=float, mode='r', shape=(id_val_size, 2048))
    logit_id = np.memmap(f"{cache_dir}/score.mmap", dtype=float, mode='r', shape=(id_val_size, class_num))
else:
    norm_feat_id = np.memmap(norm_cache, dtype=float, mode='w+', shape=(id_val_size, 2048))
    feat_id = np.memmap(f"{cache_dir}/feat.mmap", dtype=float, mode='r', shape=(id_val_size, 2048))
    norm_feat_id[:] = normalizer(feat_id)

feat_ood_all,logit_ood_all = {},{}
for ood_dataset in args.out_datasets:
    ood_feat_log = np.memmap(f"cache/{ood_dataset}vs{args.in_dataset}_{args.name}_out/feat.mmap", dtype=float,
                             mode='r', shape=(ood_dataset_size[ood_dataset], 2048))
    ood_logit_log = np.memmap(f"cache/{ood_dataset}vs{args.in_dataset}_{args.name}_out/score.mmap", dtype=float,
                              mode='r', shape=(ood_dataset_size[ood_dataset], class_num))
    feat_ood_all[ood_dataset] = normalizer(ood_feat_log)
    logit_ood_all[ood_dataset]=ood_logit_log
print("done!!!")


softmax_id = scipy.special.softmax(logit_id, axis=-1)
softmax_oods = {name:scipy.special.softmax(score, axis=-1) for name,score in logit_ood_all.items()}

def BDS(p):
    dis_prob_results, kl_dis_results, all_results, temp_results = [], [], [], []
    energy_results = []
    uniform_arr = np.ones(class_num) / class_num
    energy_id = Energy_Detector(logit_id)
    distance_id = Euc_Detector(cat_centre, norm_feat_id, class_num=1000,p=p)
    temp_id = 1 / (1 + distance_id)
    dis_prob_id = softmax(temp_id)
    kl_id = kl_divergence(uniform_arr, dis_prob_id)
    score_id = np.multiply(energy_id,kl_id)
    for name, feature_ood in feat_ood_all.items():
        energy_ood =Energy_Detector(logit_ood_all[name])
        distance_ood = Euc_Detector(cat_centre, feature_ood, class_num=1000,p=p)
        temp_ood = 1 / (1 + distance_ood)
        dis_prob_ood = softmax(temp_ood)
        kl_ood = kl_divergence(uniform_arr, dis_prob_ood)
        score_ood = np.multiply(energy_ood,kl_ood)
        results = metrics.cal_metric(energy_id, energy_ood)
        energy_results.append(results)
        results = metrics.cal_metric(np.max(temp_id, axis=1), np.max(temp_ood, axis=1))
        temp_results.append(results)
        results = metrics.cal_metric(np.max(dis_prob_id, axis=1), np.max(dis_prob_ood, axis=1))
        dis_prob_results.append(results)
        results = metrics.cal_metric(kl_id, kl_ood)
        kl_dis_results.append(results)
        results = metrics.cal_metric(score_id, score_ood)
        all_results.append(results)

    metrics.print_all_results(energy_results, args.out_datasets, f'energy')
    metrics.print_all_results(temp_results, args.out_datasets, f'direct')
    metrics.print_all_results(dis_prob_results, args.out_datasets, f'dis_prob')
    metrics.print_all_results(kl_dis_results, args.out_datasets, f'kl_dis_prob')
    metrics.print_all_results(all_results, args.out_datasets, f'all')


if __name__ == "__main__":
    for x in [2]:
        print(f"p={x}")
        BDS(x)
