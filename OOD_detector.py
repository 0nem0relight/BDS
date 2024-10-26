import os
import torch
from util.args_loader import get_args
import numpy as np
import faiss
from numpy.linalg import pinv
from scipy.special import logsumexp, softmax
from sklearn.covariance import EmpiricalCovariance
from util.model_loader import get_model, get_classifier
import scipy.special

args = get_args()


def MSP_Detector(prob):
    return prob.max(axis=-1)

def MaxLogits_Detector(logits):
    return logits.max(axis=-1)

def Energy_Detector(logits):
    return scipy.special.logsumexp(logits, axis=-1)

def KNN_Detector(norm_train_list,norm_input,K=50):
    class_num=len(norm_train_list)
    distance=np.zeros([norm_input.shape[0],class_num])
    for i in range(class_num):
        index = faiss.IndexFlatL2(norm_train_list[i].shape[1])
        index.add(norm_train_list[i])
        D, _ = index.search(norm_input, K)
        temp=np.mean(D,axis=1)
        temp=temp.reshape(-1, 1)
        distance[:,i]=np.squeeze(temp)
    return distance

def VIM_Detector(feat_train,feat_input,logits_input):
    if args.model_arch == 'resnet18-supcon':
        classifier = get_classifier(args, num_classes=10)
        classifier.eval()
        fclayer = classifier.fc
    elif args.model_arch == 'resnet18':
        model = get_model(args, 10, load_ckpt=True)
        model.eval()
        fclayer = model.fc
    elif args.model_arch == 'resnet50-supcon':
        print("classifier")
        classifier = get_classifier(args, num_classes=1000)
        classifier.eval()
        fclayer = classifier.fc
    elif args.model_arch == 'resnet50':
        print("fclayer")
        model = get_model(args, 1000, load_ckpt=True)
        model.eval()
        fclayer = model.fc
    w = fclayer.weight.cpu().detach().numpy()
    b = fclayer.bias.cpu().detach().numpy()

    if feat_input.shape[-1] >= 2048:
        DIM = 1000
    elif feat_input.shape[-1] >= 768:
        DIM = 512
    else:
        DIM = feat_input.shape[-1] // 2


    u = -np.matmul(pinv(w), b)
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(feat_train - u)
    eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
    NS = np.ascontiguousarray(
        (eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)

    vlogit_id_train = np.linalg.norm(np.matmul(feat_train - u, NS), axis=-1)
    alpha = feat_train.max(axis=-1).mean() / vlogit_id_train.mean()

    vlogit_id_val = np.linalg.norm(np.matmul(feat_input - u, NS), axis=-1) * alpha
    energy_id_val = logsumexp(logits_input, axis=-1)
    score_id = -vlogit_id_val + energy_id_val
    return score_id

def Mah_Detector(norm_feat_train,label_train,norm_feat_input,class_num=10):
    mean_feat = norm_feat_train.mean(0)
    std_feat = norm_feat_train.std(0)
    prepos_feat_ssd = lambda x: (x - mean_feat) / (std_feat + 1e-10)

    ssd_train,ssd_input=prepos_feat_ssd(norm_feat_train),prepos_feat_ssd(norm_feat_input)


    inv_sigma_cls = [None for _ in range(class_num)]
    mean_cls = [None for _ in range(class_num)]
    cov = lambda x: np.cov(x.T, bias=True)
    for cls in range(class_num):
        mean_cls[cls] = ssd_train[label_train == cls].mean(0)
        feat_cls_center = ssd_train[label_train == cls] - mean_cls[cls]

        inv_sigma_cls[cls] = np.linalg.pinv(cov(feat_cls_center))

    def maha_score(X):
        score_cls = np.zeros((class_num, len(X)))
        for cls in range(class_num):
            inv_sigma = inv_sigma_cls[cls]
            mean = mean_cls[cls]
            z = X - mean
            score_cls[cls] = np.sum(z * (inv_sigma.dot(z.T)).T, axis=-1)
        return score_cls.T

    Mah_input = maha_score(ssd_input)
    return Mah_input

def p_norm(x, p):
    return np.power(np.sum(np.power(np.abs(x), p)), 1 / p)
def Euc_Detector(cat_centre,norm_feat_input,class_num,p):

    distance_input = []
    for index, item in enumerate(norm_feat_input):
        distance_input.append([])
        for j in range(class_num):
            distance_input[index].append(p_norm(item - cat_centre[j],p=p))
    distance_input = np.array(distance_input)
    return distance_input

def GradNorm_Detector(feat_input,class_num):
    if args.model_arch == 'resnet18-supcon':
        classifier = get_classifier(args, num_classes=10)
        classifier.eval()
        fclayer = classifier.fc
    elif args.model_arch == 'resnet18':
        model = get_model(args, 10, load_ckpt=True)
        model.eval()
        fclayer = model.fc
    elif args.model_arch == 'resnet50-supcon':
        print("classifier")
        classifier = get_classifier(args, num_classes=1000)
        classifier.eval()
        fclayer = classifier.fc
    elif args.model_arch == 'resnet50':
        print("fclayer")
        model = get_model(args, 1000, load_ckpt=True)
        model.eval()
        fclayer = model.fc
    w = fclayer.weight.cpu().detach().numpy()
    b = fclayer.bias.cpu().detach().numpy()

    fc = torch.nn.Linear(*w.shape[::-1])
    fc.weight.data[...] = torch.from_numpy(w)
    fc.bias.data[...] = torch.from_numpy(b)
    fc.cuda()

    x = torch.tensor(feat_input.copy()).float().cuda()
    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()

    confs = []

    for i in x:
        targets = torch.ones((1, class_num)).cuda()
        fc.zero_grad()
        loss = torch.mean(
            torch.sum(-targets * logsoftmax(fc(i[None])), dim=-1))
        loss.backward()
        layer_grad_norm = torch.sum(torch.abs(
            fc.weight.grad.data)).cpu().numpy()
        confs.append(layer_grad_norm)
    score_input=np.array(confs)

    return score_input
def generalized_entropy(softmax_id_val, gamma=0.1, M=10):
    probs = softmax_id_val
    probs_sorted = np.sort(probs, axis=1)[:, -M:]
    scores = np.sum(probs_sorted ** gamma * (1 - probs_sorted) ** (gamma), axis=1)
    return scores

def DML_Dectector(feat_input,lamda):
    if args.model_arch == 'resnet18-supcon':
        classifier = get_classifier(args, num_classes=10)
        classifier.eval()
        fclayer = classifier.fc
    elif args.model_arch == 'resnet18':
        model = get_model(args, 10, load_ckpt=True)
        model.eval()
        fclayer = model.fc
    elif args.model_arch == 'resnet50-supcon':
        classifier = get_classifier(args, num_classes=1000)
        classifier.eval()
        fclayer = classifier.fc
    elif args.model_arch == 'resnet50':
        model = get_model(args, 1000, load_ckpt=True)
        model.eval()
        fclayer = model.fc
    w = fclayer.weight.cpu().detach().numpy()
    b = fclayer.bias.cpu().detach().numpy()

    feature_norm = np.linalg.norm(feat_input, axis=1, keepdims=True)  # 特征向量的模长，形状为 (batch_size, 1)
    weight_norm = np.linalg.norm(w, axis=1, keepdims=True)

    dot_product = np.dot(feat_input, w.T)  # (batch_size, output_dim)
    cosine_similarity = dot_product / (feature_norm * weight_norm.T)

    MaxCos = np.max(cosine_similarity, axis=1)  # 最大余弦相似度，形状为 (batch_size,)
    max_index = np.argmax(cosine_similarity, axis=1)

    maxfeature=feat_input[max_index]
    MaxNorm =np.linalg.norm(maxfeature, ord=2, axis=-1)
    DML=lamda*MaxCos+MaxNorm
    return DML
