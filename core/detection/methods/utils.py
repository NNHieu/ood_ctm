from pathlib import Path
from typing import Callable
import torch
import torch.nn.functional as F
from tqdm import tqdm

from sklearn.decomposition import IncrementalPCA

def list_feature_by_class(features, labels, preds=None):
    # print(preds)
    num_class = labels.max().item() + 1
    features_by_class = {}
    for c in range(num_class):
        if preds is None:
            inds = (labels == c).squeeze()
        else:
            inds = ((labels == c) * (preds == c)).squeeze()
            # print(inds.sum())
        features_by_class[c] = features[inds]
    return features_by_class

def mean_and_normalize_list_features(features_by_class):
    class_means = torch.cat([f.mean(dim=0, keepdim=True) for f in features_by_class])
    class_dir = F.normalize(class_means, dim=-1)
    return class_means, class_dir

class ForwardPass(Callable):
    def forward(self, batch, return_feature_list=False, penultimate_feature=False):
        '''
        Params:
            batch: (X, y)
            return_feature_list: default=False
            penultimate_feature: default=False
        '''
        raise NotImplementedError()
    
    def __call__(self, batch, return_feature_list=False, penultimate_feature=False):
        return self.forward(batch, return_feature_list=False, penultimate_feature=False)

class PreProcessFeature(Callable):
    def __init__(self, layer_index=0) -> None:
        super().__init__()
        self.layer_index = layer_index

    def _forward_collect(self, forward_fn, data):
        logits, features_list = forward_fn(data, return_feature_list=True)
        probs = F.softmax(logits, dim=1)
        features = features_list[self.layer_index]
        preds = probs.argmax(-1).squeeze()
        return features, probs, preds

    def adapt(self, forward_fn, train_loader, **kwargs):
        with torch.no_grad():
            outputs = []
            for batch in tqdm(train_loader, desc="Collect ref feats"):
                b_features, probs, preds = self._forward_collect(forward_fn, batch)
                labels = batch[1].to(b_features.device)
                outputs.append((probs, b_features, labels))
            probs, b_features, labels = zip(*outputs)
            probs = torch.vstack(probs)
            features = torch.vstack(b_features)
            labels = torch.cat(labels)

        Z = features
        Cov = torch.matmul(Z.t(), Z) * (1 / Z.shape[0])
        r = torch.linalg.matrix_rank(Cov)
        U, S, Vh = torch.linalg.svd(Cov)

        self._proj_matrix = U[:, :-120] @ Vh[:-120, :]

    def __call__(self, feats):
        return feats @ self._proj_matrix.T

class CalMeanClass:
    def __init__(self, layer_index=0) -> None:
        super().__init__()
        self.layer_index = layer_index

    def _forward_collect(self, forward_fn, data):
        logits, features_list = forward_fn(data, return_feature_list=True)
        probs = F.softmax(logits, dim=1)
        features = features_list[self.layer_index]
        preds = probs.argmax(-1).squeeze()
        return features, probs, preds

    def _adapt(self, forward_fn, train_loader):
        b_features, probs, _ = self._forward_collect(forward_fn, next(iter(train_loader)))
        feat_size = b_features.shape[-1]
        num_classes = probs.shape[-1]
        device = b_features.device

        class_means = torch.zeros(num_classes, feat_size).to(device)
        class_counts = torch.zeros(num_classes).to(device)
        for batch in tqdm(train_loader, desc="Collect ref feats"):
            b_features, probs, preds = self._forward_collect(forward_fn, batch)
            # b_features = b_features.cpu()
            labels = batch[1].squeeze().to(device)
            # outputs.append((probs, b_features, labels))
            correct_inds = (preds == labels)
            b_features = b_features[correct_inds]
            labels = labels[correct_inds]
            # print(b_features.shape, labels.shape)
            # labels = preds
            class_means.index_add_(0, labels, b_features)
            class_counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float))
        global_mean = class_means.sum(dim=0, keepdim=True) / class_counts.sum()
        class_means = class_means / class_counts.unsqueeze(-1)
        return class_means, global_mean

    def adapt(self, forward_fn, train_loader, *, cache_feats_path=None, cache_mean_path=None, normalize=False, **kwargs):
        with torch.no_grad():
            self.class_means_, self.global_mean_ = self._adapt(forward_fn, train_loader)
        if cache_mean_path is not None:
            torch.save((self.class_means_, self.global_mean_), cache_mean_path)
        return self.class_means_, self.global_mean_

    def _get_state(self):
        '''
        Return a dict of state includes:
            class_means: tensor of class means
            global_mean: tensor of global mean
        '''
        return {
            'layer_index': self.layer_index,
            'class_means': self.class_means_,
            'global_mean': self.global_mean_,
        }

    def _set_state(self, state_dict):
        self.layer_index = 0 # state_dict['layer_index']
        self.class_means_ = state_dict[0] # state_dict['class_means']
        self.global_mean_ = state_dict[1] # state_dict['global_mean']

    def save_cached_state(self, path):
        torch.save(self._get_state(), path)
    
    def load_cached_state(self, path):
        state_dict = torch.load(path)
        self._set_state(state_dict)
    
class CalCovClassWise:
    '''
    Calculate covariance matrix for each class using partial PCA
    '''
    
    def __init__(self, layer_index=0) -> None:
        super().__init__()
        self.layer_index = layer_index
        self._state = None

    def _forward_collect(self, forward_fn, data):
        logits, features_list = forward_fn(data, return_feature_list=True)
        probs = F.softmax(logits, dim=1)
        features = features_list[self.layer_index]
        preds = probs.argmax(-1).squeeze()
        return features, probs, preds

    def _gen_state(self, forward_fn, train_loader):
        # Get a sample batch to get the feature size
        with torch.no_grad():
            b_features, probs, preds = self._forward_collect(forward_fn, next(iter(train_loader)))
            batch_size, feat_size = b_features.shape
            num_classes = probs.shape[-1]
            device = b_features.device

        Covs = [torch.zeros(feat_size, feat_size).to(device) for _ in range(num_classes)]
        counts = [0 for _ in range(num_classes)]
        class_means = torch.zeros(num_classes, feat_size).to(device)
        for batch in tqdm(train_loader, desc="Collect ref feats"):
            b_features, probs, preds = self._forward_collect(forward_fn, batch)
            preds = preds.to(device)
            b_features = b_features.to(device)
            labels = batch[1].to(device)
            # b_features = F.normalize(b_features, dim=1) 

            # b_features = b_features.cpu().numpy()
            # labels = labels.cpu().numpy()
            class_features_list = list_feature_by_class(b_features, preds)
            for c in class_features_list.keys():
                num_new = class_features_list[c].shape[0]
                # Cov = Cov * (counts[c]/(counts[c] + num_new)) + (class_features_list[c].T @ class_features_list[c]) * (1/(counts[c] + num_new))
                # class_means[c] = class_means[c] * (counts[c]/(counts[c] + num_new)) + class_features_list[c].mean(axis=0) * (1/(counts[c] + num_new))
                Covs[c] = Covs[c] + (class_features_list[c].T @ class_features_list[c])
                class_means[c] += class_features_list[c].sum(axis=0)
                counts[c] += num_new
        counts = torch.tensor(counts).to(device)
        
        self._state = {
            'layer_index': self.layer_index,
            'class_sums': class_means,
            'Cov_sums': Covs,
            'counts': counts,
            'feat_size': feat_size,
            'num_classes': num_classes,
        }

    def _runtime_adapt(self, forward_fn, train_loader, *, normalize=False, centering=False):
        class_sums = self._state['class_sums']
        Cov_sums = self._state['Cov_sums']
        counts = self._state['counts']
        num_classes = self._state['num_classes']
        # device = 'cpu'

        global_mean = class_sums.sum(dim=0, keepdim=True) / counts.sum()
        class_means = class_sums / counts.unsqueeze(-1)

        # shift all features to global mean
        if centering:
            # class_means -= counts.unsqueeze(-1) * global_mean
            class_means -= global_mean
            for c in range(num_classes):
                Cov_sums[c] -= counts[c] * global_mean.T @ global_mean

        class_mean_dirs = F.normalize(class_means, dim=-1)
        
        components = []
        singular_values = []
        ranks = []
        # for c in range(num_classes):
        #     Cov[c] = Covs[c] / counts[c]
        # Covs = torch.stack(Covs)
        # Cov = torch.sum(Covs, dim=0) / counts.sum()

        for c in range(num_classes):
            Cov = Cov_sums[c] / counts[c]
            class_mean_dir = class_mean_dirs[c]
            u_u_T = class_mean_dir.unsqueeze(-1) @ class_mean_dir.unsqueeze(0)
            processed_cov = Cov - Cov @ u_u_T - u_u_T @ Cov + u_u_T @ Cov @ u_u_T

            U, S, Vh = torch.linalg.svd(processed_cov)
            rank = torch.linalg.matrix_rank(processed_cov)
            components.append(U)
            singular_values.append(S)
            ranks.append(rank.item())
        
        self.components_ = components
        self.singular_values_ = singular_values
        self.class_means_ = class_means
        self.class_mean_dirs_ = class_mean_dirs
        self.global_mean_ = global_mean
        self.ranks_ = ranks

    def _get_state(self):
        return self._state

    def _set_state(self, state_dict):
        self._state = state_dict

    def save_cached_state(self, path):
        torch.save(self._get_state(), path)
    
    def load_cached_state(self, path):
        state_dict = torch.load(path)
        self._set_state(state_dict)

    def adapt(self, forward_fn, train_loader, *, normalize=False, **kwargs):
        if self._state is None:
            self._gen_state(forward_fn, train_loader)
        self._runtime_adapt(forward_fn, train_loader, normalize=normalize, **kwargs)