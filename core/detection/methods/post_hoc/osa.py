from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from .base_detector import BaseDetector
from ..utils import CalCovClassWise

from time import time
import logging

from tqdm import tqdm

logger = logging.getLogger(__name__)

def list_feature_by_class(features, labels):
    num_class = labels.max().item() + 1
    features_by_class = []
    for c in range(num_class):
        inds = (labels == c).squeeze()
        features_by_class.append(features[inds])
    return features_by_class

def mean_and_normalize_list_features(features_by_class):
    class_means = torch.cat([f.mean(dim=0, keepdim=True) for f in features_by_class])
    class_dir = F.normalize(class_means, dim=-1)
    return class_means, class_dir


class OSA(BaseDetector):
    def __init__(self, *, layer_index, rank_rtol, class_dir='mean', add_bg_score: bool, weight_type="inv", remove_mean_dir=True, cache=None, device=None, **kwargs) -> None:
        super().__init__("OSA")
        self.layer_index = layer_index
        self.rank_rtol = rank_rtol
        self.add_bg_score = add_bg_score
        self.weight_type = weight_type
        self.remove_mean_dir = remove_mean_dir
        self.class_dir = class_dir
        if cache is not None:
            self.cache = Path(cache)
            self.cache.parent.mkdir(parents=True, exist_ok=True)
        self.device = device

    def _forward_collect(self, forward_fn, data, device=None):
        logits, features_list = forward_fn(data, return_feature_list=True)
        # logging.info("Forwarded")
        probs = F.softmax(logits, dim=1)
        features = features_list[self.layer_index]
        preds = probs.argmax(-1).squeeze()
        if device is not None:
            features = features.to(device)
            probs = probs.to(device)
            preds = preds.to(device)
        return features, probs, preds

    @torch.no_grad()
    def adapt(self, forward_fn, train_loader, *, net, **kwargs):
        loaded_cache = False
        adapter = CalCovClassWise(layer_index=self.layer_index)
        if self.cache is not None and self.cache.exists():
            print(f"Loading cached class means from {self.cache}")
            adapter.load_cached_state(self.cache)
            loaded_cache = True
        adapter.adapt(forward_fn, 
                      train_loader, 
                      centering=False)
        
        if not loaded_cache and self.cache is not None:
            adapter.save_cached_state(self.cache)
        
        print(adapter.ranks_)

        inv_pc_weights = []
        for i in range(len(adapter.singular_values_)):
            class_weights = (1 / torch.sqrt(adapter.singular_values_[i]))
            # class_weights[adapter.ranks_[i]:] = 0
            # class_weights = class_weights / 
            inv_pc_weights.append(class_weights.unsqueeze(0))
            # class_weights = class_weights / class_weights.sum()
            # class_weights = adapter.singular_values_[i].clone()
            # class_weights = torch.ones_like(adapter.singular_values_[i])

            # class_weights[adapter.ranks_[i]:] = 0
            # class_weights[100:] = 0

            # print("Top 10 singular values:", adapter.singular_values_[i][:10])
            # print("10 Smallest singular value:", adapter.singular_values_[i][-10:])
            # pc_weights.append(torch.ones_like(adapter.singular_values_[i]).unsqueeze(0))


        # pc_weights = []
        # if self.weight_type == "inv":
            
        #     # print(pc_weights)

        # elif self.weight_type == "var":
        #     for i in range(len(self.svd_collect)):
        #         pc_weights, Vh, r = self.svd_collect[i][:3]
        #         # Vh = Vh[:800]
        #         # r = 800
        #         if isinstance(self.rank_rtol, str) and ('p' in self.rank_rtol or 's' in self.rank_rtol):
        #             percent = float(self.rank_rtol[1:]) / 100
        #             # logger.info(f"Using {percent} of PCs: {Vh.shape}")
        #             r = int(Vh.shape[0] * percent)
        #         full_weights = torch.zeros(1, Vh.shape[0])
        #         full_weights[0, :r] = pc_weights[0, :r]
        #         self.svd_collect[i] = (full_weights, Vh, r)
        # elif self.weight_type == "none":
        #     for i in range(len(self.svd_collect)):
        #         pc_weights, Vh, r = self.svd_collect[i][:3]
        #         # Vh = Vh[:800]
        #         if isinstance(self.rank_rtol, str):
        #             if ('p' in self.rank_rtol or 's' in self.rank_rtol):
        #                 percent = float(self.rank_rtol[1:]) / 100
        #                 # logger.info(f"Using {percent} of PCs: {Vh.shape}")
        #                 r = int(Vh.shape[0] * percent)
        #             if 'e' in self.rank_rtol:
        #                 explained_variance_ratio = torch.sqrt(pc_weights[0]) / torch.sqrt(pc_weights[0]).sum()
        #                 cumsum = explained_variance_ratio.cumsum(dim=-1)
        #                 r = (cumsum <= (float(self.rank_rtol[1:]) / 100)).sum()
        #             Vh = Vh[:r]
        #         pc_weights = torch.ones(1, Vh.shape[0])
        #         # pc_weights[0, 0] = 0
        #         if isinstance(self.rank_rtol, str) and 's' in self.rank_rtol:
        #             pc_weights[0,:r] = 0
        #         else:
        #             pc_weights[0,r:] = 0
        #         # pc_weights[0, 200:] = 0
        #         # pc_weights[0, -1] = 0
        #         # pc_weights = (pc_weights.shape[1] / r) * pc_weights
        #         self.svd_collect[i] = (pc_weights, Vh, r)
        # else:
        #     raise ValueError(f"Unknown weight_type {self.weight_type}")

        self._cated_components = torch.cat(adapter.components_, dim=1) #shape=(d, C*d)
        self._cated_weights = torch.cat(inv_pc_weights, dim=1) # 1 x d*c
        self._num_classes = len(adapter.components_)
        self._ranks = torch.tensor(adapter.ranks_).unsqueeze(0).to(self._cated_weights.device)
        self._adapter = adapter

        # logger.info(self._r_list)

        # self._Vs = self._Vs.cuda()
        # self._weights = self._weights.cuda()
        # self._r_list = self._r_list.cuda()

    
    @torch.no_grad()
    def score_batch(self, forward_fn, data):
        features, probs, preds = self._forward_collect(forward_fn, data)
        # features -= self._adapter.global_mean_

        preds = torch.argmax(probs, dim=1)


        features = F.normalize(features, dim=-1)
        
        feat_mean = torch.index_select(self._adapter.class_mean_dirs_, 0, preds)
        scalings = (features * feat_mean).sum(dim=-1)
        features -= feat_mean * scalings.unsqueeze(-1)

        projs = ((features @ self._cated_components)**2)
        score_dict = {}
        # OSA-inv
        scores = (projs * self._cated_weights).view(features.shape[0], self._num_classes, -1).sum(dim=-1)
        # scores = scores[torch.arange(preds.shape[0]), preds]
        scores = scores.min(dim=-1)[0]
        score_dict["osa-inv"] = -scores.cpu()

        # OSA-full
        scores = projs.view(features.shape[0], self._num_classes, -1).sum(dim=-1)
        # scores = scores[torch.arange(preds.shape[0]), preds]
        scores = scores.min(dim=-1)[0]
        score_dict["osa-full"] = -scores.cpu()

        
        # weighted_cossim = weighted_cossim / (1 + weighted_cossim)
        # weighted_cossim = weighted_cossim / self._ranks
        # scores = weighted_cossim[torch.arange(preds.shape[0]), preds]
        # scores = weighted_cossim.max(dim=-1)[0]
        # scores = scores.cpu()
        return {
            "score_dict": score_dict,
            "preds": preds,
            "probs": probs,
        }

    def __str__(self) -> str:
        cldir = f"_cldir={self.class_dir}" if self.remove_mean_dir else ""
        bg = "_bg" if self.add_bg_score else ""
        weight = f"_we={self.weight_type}"
        return f"{self.name}_layer={self.layer_index}"
