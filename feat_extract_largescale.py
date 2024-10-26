import torch
import os
from util.args_loader import get_args
from util.data_loader import get_loader_in, get_loader_out
from util.model_loader import get_model,get_classifier
import numpy as np
import torch.nn.functional as F

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = get_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


loader_in_dict = get_loader_in(args, config_type="eval", split=('train', 'val'))
trainloaderIn, testloaderIn, num_classes = loader_in_dict.train_loader, loader_in_dict.val_loader, loader_in_dict.num_classes
model = get_model(args, num_classes, load_ckpt=True).to(device)


batch_size = args.batch_size
featdim = {
    'resnet50': 2048,
    'resnet50-supcon': 2048,
}[args.model_arch]

FORCE_RUN = True
ID_RUN = True
OOD_RUN = True
if args.model_arch == 'resnet50-supcon':
    Classifier_RUN=True
    classifier = get_classifier(args, num_classes=num_classes)
else:
    Classifier_RUN = False

if ID_RUN:
    for split, in_loader in [ ('train', trainloaderIn),('val', testloaderIn)]: # ('train', trainloaderIn),
        cache_dir = f"cache/{args.in_dataset}_{split}_{args.name}_in"
        if FORCE_RUN or not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            feat_log = np.memmap(f"{cache_dir}/feat.mmap", dtype=float, mode='w+', shape=(len(in_loader.dataset), featdim))
            logit_log = np.memmap(f"{cache_dir}/logit.mmap", dtype=float, mode='w+', shape=(len(in_loader.dataset), num_classes))
            label_log = np.memmap(f"{cache_dir}/label.mmap", dtype=float, mode='w+', shape=(len(in_loader.dataset),))

            model.eval()
            if Classifier_RUN:
                classifier.eval()
            for batch_idx, (inputs, targets) in enumerate(in_loader):

                inputs, targets = inputs.to(device), targets.to(device)
                start_ind = batch_idx * batch_size
                end_ind = min((batch_idx + 1) * batch_size, len(in_loader.dataset))

                if args.model_arch == 'resnet50-supcon':
                    out = model.encoder(inputs)
                else:
                    out = model.features(inputs)
                if len(out.shape) > 2:
                    out = F.adaptive_avg_pool2d(out, 1)
                    out = out.view(out.size(0), -1)

                logit = model.fc(out)
                if Classifier_RUN:
                    logit = classifier(out)
                maxlog=torch.argmax(logit,dim=1)
                feat_log[start_ind:end_ind, :] = out.data.cpu().numpy()
                label_log[start_ind:end_ind] = targets.data.cpu().numpy()
                logit_log[start_ind:end_ind] = logit.data.cpu().numpy()
                feat_log.flush()
                label_log.flush()
                logit_log.flush()
                if batch_idx % 100 == 0:
                    print(f"{batch_idx}/{len(in_loader)}")
        else:
            feat_log = np.memmap(f"{cache_dir}/feat.mmap", dtype=float, mode='r', shape=(len(in_loader.dataset), featdim))
            logit_log = np.memmap(f"{cache_dir}/logit.mmap", dtype=float, mode='r', shape=(len(in_loader.dataset), num_classes))
            label_log = np.memmap(f"{cache_dir}/label.mmap", dtype=float, mode='r', shape=(len(in_loader.dataset),))

if OOD_RUN:

    for ood_dataset in args.out_datasets:
        loader_test_dict = get_loader_out(args, dataset=(None, ood_dataset), split=('val'))
        out_loader = loader_test_dict.val_ood_loader

        cache_dir = f"cache/{ood_dataset}vs{args.in_dataset}_{args.name}_out"
        if FORCE_RUN or not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            ood_feat_log = np.memmap(f"{cache_dir}/feat.mmap", dtype=float, mode='w+', shape=(len(out_loader.dataset), featdim))
            ood_logit_log = np.memmap(f"{cache_dir}/logit.mmap", dtype=float, mode='w+', shape=(len(out_loader.dataset), num_classes))
            model.eval()
            if Classifier_RUN:
                classifier.eval()
            for batch_idx, (inputs, _) in enumerate(out_loader):
                inputs = inputs.to(device)
                start_ind = batch_idx * batch_size
                end_ind = min((batch_idx + 1) * batch_size, len(out_loader.dataset))

                if args.model_arch == 'resnet50-supcon':
                    out = model.encoder(inputs)
                else:
                    out = model.features(inputs)
                if len(out.shape) > 2:
                    out = F.adaptive_avg_pool2d(out, 1)
                    out = out.view(out.size(0), -1)

                logit = model.fc(out)
                if Classifier_RUN:
                    logit = classifier(out)
                maxlog=torch.argmax(logit,dim=1)
                ood_feat_log[start_ind:end_ind, :] = out.data.cpu().numpy()
                ood_logit_log[start_ind:end_ind] = logit.data.cpu().numpy()
                ood_feat_log.flush()
                ood_logit_log.flush()
                if batch_idx % 100 == 0:
                    print(f"{batch_idx}/{len(out_loader)}")
        else:
            ood_feat_log = np.memmap(f"{cache_dir}/feat.mmap", dtype=float, mode='r', shape=(len(out_loader.dataset), featdim))
            ood_logit_log = np.memmap(f"{cache_dir}/logit.mmap", dtype=float, mode='r', shape=(len(out_loader.dataset), num_classes))
