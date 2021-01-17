from datetime import datetime
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import hflip
from data.data_pipe import de_preprocess
import io
from torchvision import transforms as trans
import torch
from models.mobilenet import l2_norm
from sklearn.model_selection import KFold

def pair_euc_score(x1, x2):
    x1, x2 = np.array(x1), np.array(x2)
    dist = np.sum(np.square(x1 - x2), axis=1)
    return -dist


def pair_MLS_score(x1, x2, sigma_sq1=None, sigma_sq2=None):
    if sigma_sq1 is None:
        x1, x2 = np.array(x1), np.array(x2)
        assert sigma_sq2 is None, 'either pass in concated features, or mu, sigma_sq for both!'
        D = int(x1.shape[1] / 2)
        mu1, sigma_sq1 = x1[:, :D], x1[:, D:]
        mu2, sigma_sq2 = x2[:, :D], x2[:, D:]
    else:
        x1, x2 = np.array(x1), np.array(x2)
        sigma_sq1, sigma_sq2 = np.array(sigma_sq1), np.array(sigma_sq2)
        mu1, mu2 = x1, x2
    sigma_sq_mutual = sigma_sq1 + sigma_sq2
    dist = np.sum(np.square(mu1 - mu2) / sigma_sq_mutual + np.log(sigma_sq_mutual), axis=1)
    return dist



def separate_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])
    return paras_only_bn, paras_wo_bn

hflip = trans.Compose([
            de_preprocess,
            trans.ToPILImage(),
            trans.functional.hflip,
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
def hflip_batch(imgs_tensor):
    hfliped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        hfliped_imgs[i] = hflip(img_ten)
    return hfliped_imgs


def get_time():
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')


def gen_plot(fpr, tpr):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve", fontsize=14)
    plot = plt.plot(fpr, tpr, linewidth=2)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    plt.close()
    return buf

def evaluate(embeddings, actual_issame, nrof_folds=10, pca=0):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy, best_thresholds = calculate_roc(thresholds, embeddings1, embeddings2,
                                       np.asarray(actual_issame), nrof_folds=nrof_folds, pca=pca)
    return tpr, fpr, accuracy, best_thresholds

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, pca=0):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    best_thresholds = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)
    # print('pca', pca)

    if pca == 0:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
#         print('best_threshold_index', best_threshold_index, acc_train[best_threshold_index])
        best_thresholds[fold_idx] = thresholds[best_threshold_index]
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 dist[test_set],
                                                                                                 actual_issame[
                                                                                                     test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set],
                                                      actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy, best_thresholds

def get_evaluate(carray, issame,model, nrof_folds = 5, tta = False):
    device = torch.device("cpu")
    model.eval()
    idx = 0
    embeddings = np.zeros([len(carray), 512])
    with torch.no_grad():
        while idx + 200 <= len(carray):
            batch = torch.tensor(carray[idx:idx + 200])
            if tta:
                fliped = hflip_batch(batch)
                emb_batch = model(batch.to(device))[0] + model(fliped.to(device))[0]
                embeddings[idx:idx + 200] = l2_norm(emb_batch)
            else:
                embeddings[idx:idx + 200] = model(batch.to(device))[0].cpu()
            idx += 200
        if idx < len(carray):
            batch = torch.tensor(carray[idx:])
            if tta:
                fliped = hflip_batch(batch)
                emb_batch = model(batch.to(device))[0] + model(fliped.to(device))[0]
                embeddings[idx:] = l2_norm(emb_batch)
            else:
                embeddings[idx:] = model(batch.to(device)).cpu()
    tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
    buf = gen_plot(fpr, tpr)
    roc_curve = Image.open(buf)
    roc_curve_tensor = trans.ToTensor()(roc_curve)
    return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor

def pfe_evaluate(mu_embeddings, sig_sq_embeddings, actual_issame, nrof_folds=5, pca=0):
    # Calculate evaluation metrics
    mu_embeddings1, sig_sq_embeddings1 = mu_embeddings[0::2], sig_sq_embeddings[0::2]
    mu_embeddings2, sig_sq_embeddings2 = mu_embeddings[1::2], sig_sq_embeddings[1::2]

    tpr, fpr, accuracy, best_thresholds = pfe_calculate_roc(mu_embeddings1, mu_embeddings2,
                                                sig_sq_embeddings1, sig_sq_embeddings2,
                                       np.asarray(actual_issame), nrof_folds=nrof_folds, pca=pca)
#     thresholds = np.arange(0, 4, 0.001)
#     val, val_std, far = calculate_val(thresholds, embeddings1, embeddings2,
#                                       np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds)
#     return tpr, fpr, accuracy, best_thresholds, val, val_std, far
    return tpr, fpr, accuracy, best_thresholds

def pfe_calculate_roc(mu_embeddings1, mu_embeddings2,
                                                sig_sq_embeddings1, sig_sq_embeddings2,
                      actual_issame, nrof_folds=10, pca=0):
    assert (mu_embeddings1.shape[0] == mu_embeddings2.shape[0])
    assert (mu_embeddings1.shape[1] == mu_embeddings2.shape[1])
    assert (sig_sq_embeddings1.shape[0] == sig_sq_embeddings2.shape[0])
    assert (sig_sq_embeddings1.shape[1] == sig_sq_embeddings2.shape[1])


    dist = pair_MLS_score(mu_embeddings1,mu_embeddings2,sig_sq_embeddings1,sig_sq_embeddings2)
    thresholds = np.arange(np.min(dist), np.max(dist), 0.5)
    nrof_pairs = min(len(actual_issame), mu_embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    best_thresholds = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)
    # print('pca', pca)




    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
#         print('best_threshold_index', best_threshold_index, acc_train[best_threshold_index])
        best_thresholds[fold_idx] = thresholds[best_threshold_index]
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 dist[test_set],
                                                                                                 actual_issame[
                                                                                                     test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set],
                                                      actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy, best_thresholds


def get_pfe_evaluate(carray, issame, model, pfe_model, nrof_folds = 5, tta = False):
    device = torch.device("cpu")
    model.eval()
    pfe_model.eval()
    idx = 0
    embeddings = np.zeros([len(carray), 512])
    embeddings_2 = np.zeros([len(carray), 512])
    with torch.no_grad():
        while idx + 200 <= len(carray):
            batch = torch.tensor(carray[idx:idx + 200])
            if tta:
                fliped = hflip_batch(batch)
                mu_batch = model(batch.to(device))[0]
                conv_final_batch = model(batch.to(device))[1]
                sg_sq_batch = torch.exp(pfe_model(conv_final_batch.to(device)))
                embeddings[idx:idx + 200] = mu_batch
                embeddings_2[idx:idx + 200] = sg_sq_batch
            idx += 200
        if idx < len(carray):
            batch = torch.tensor(carray[idx:])
            if tta:
                fliped = hflip_batch(batch)
                mu_batch = model(batch.to(device))[0]
                conv_final_batch = model(batch.to(device))[1]
                sg_sq_batch = torch.exp(pfe_model(conv_final_batch.to(device)))
                embeddings[idx:] = mu_batch
                embeddings_2[idx:] = sg_sq_batch

    tpr, fpr, accuracy, best_thresholds = pfe_evaluate(embeddings,embeddings_2, issame, nrof_folds)
    buf = gen_plot(fpr, tpr)
    roc_curve = Image.open(buf)
    roc_curve_tensor = trans.ToTensor()(roc_curve)
    return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor
