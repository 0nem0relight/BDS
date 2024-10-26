import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from util import metrics
from util.args_loader import get_args
from sklearn.metrics import pairwise_distances
import torch
import scipy.special
from OOD_detector import *



args = get_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ['KMP_DUPLICATE_LIB_OK']='True'
load_tensorboard=False

normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
prob_normalizer=lambda x: x/ np.sum(x,axis=1,keepdims=True)
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def kl_divergence(p, q):
    return np.sum(p * np.log(p / q),axis=1)


cache_name = f"./cache/{args.in_dataset}_train_{args.name}_in_alllayers.npz"
log = np.load(cache_name, allow_pickle=True)
feat_log, logit_train, label_train = log["feat_log"].astype(np.float32),log["score_log"].astype(np.float32),log["label_log"].astype(np.float32)
norm_feat_train=normalizer(feat_log[:, range(448, 960)])
pred_train=np.argmax(logit_train,axis=1)
class_num = logit_train.shape[1]

cache_name = f"./cache/{args.in_dataset}_val_{args.name}_in_alllayers.npz"
log = np.load(cache_name, allow_pickle=True)
feat_log_val, logit_id, label_id = log["feat_log"].astype(np.float32),log["score_log"].astype(np.float32),log["label_log"].astype(np.float32)
max_log_val=np.argmax(logit_id,axis=1)
norm_feat_id=normalizer(feat_log_val[:, range(448, 960)])

feat_ood_all,logit_ood_all = {},{}
for ood_dataset in args.out_datasets:
    cache_name = f"cache/{ood_dataset}vs{args.in_dataset}_{args.name}_out_alllayers.npz"
    log = np.load(cache_name, allow_pickle=True)
    ood_feat_log, ood_logit_log = log["ood_feat_log"], log["ood_score_log"]
    feat_ood_all[ood_dataset] = ood_feat_log[:, range(448, 960)]
    logit_ood_all[ood_dataset]=ood_logit_log

softmax_id = scipy.special.softmax(logit_id, axis=-1)
softmax_oods = {name:scipy.special.softmax(score, axis=-1) for name,score in logit_ood_all.items()}


stdD_dis_results,dis_results,stdD_results=[],[],[]
dis_prob_results,kl_dis_results,all_results,temp_results=[],[],[],[]
energy_results=[]
stdD_dict= {}
index = np.where(label_train == pred_train)
filtered_feat = norm_feat_train[index]
filtered_pred = pred_train[index]
filtered_feat, filtered_pred = np.array(filtered_feat), np.array(filtered_pred)
cat_centre = [None for _ in range(class_num)]
class_feat=[None for _ in range(class_num)]
for cls in range(class_num):
    index=(filtered_pred == cls)
    class_feat[cls] = filtered_feat[index]
    cat_centre[cls]=class_feat[cls].mean(0)
cat_centre=np.array(cat_centre)


uniform_arr=np.ones(class_num)/class_num
energy_id = DML_Dectector(feat_log_val[:, range(448, 960)],2)#Energy_Detector(logit_id)
distance_id =Euc_Detector(cat_centre,norm_feat_id,class_num=10,p=2)
temp_id=1/(1+distance_id)#norm.pdf(distance_id, loc=0, scale=sigma)
dis_prob_id=softmax(temp_id)
kl_id=kl_divergence(uniform_arr,dis_prob_id)
score_id = np.multiply(energy_id, kl_id)

for name, feature_ood  in feat_ood_all.items():
    energy_ood =DML_Dectector(feature_ood,2)#Energy_Detector(logit_ood_all[name])
    feature_ood=normalizer(feature_ood)
    distance_ood=Euc_Detector(cat_centre,feature_ood,class_num=10,p=2)
    temp_ood =1 / (1 + distance_ood)#norm.pdf(distance_ood, loc=0, scale=sigma)
    dis_prob_ood = softmax(temp_ood)
    kl_ood = kl_divergence(uniform_arr, dis_prob_ood)
    score_ood = np.multiply(energy_ood, kl_ood)

    results=metrics.cal_metric(energy_id,energy_ood)
    energy_results.append(results)
    results = metrics.cal_metric(np.max(temp_id,axis=1), np.max(temp_ood,axis=1))
    temp_results.append(results)
    results = metrics.cal_metric(np.max(dis_prob_id,axis=1), np.max(dis_prob_ood,axis=1))
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



