from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .base_detector import BaseDetector

normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10


class KNNPostprocessor(BaseDetector):
    def __init__(self, *, K, layer_index=0, cache=None, **kwargs):
        super().__init__('knn')
        self.K = K
        self.activation_log = None
        self.layer_index = layer_index
        self.cache = None
        if cache is not None:
            self.cache = Path(cache)
            self.cache.parent.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def adapt(self, forward_fn, ref_id_dataloader, **kwargs):
        if self.cache is not None and self.cache.exists():
            self.activation_log = np.load(self.cache)
        else:
            activation_log = []
            for batch in tqdm(ref_id_dataloader,
                                desc='Building KNN index table: ',
                                position=0,
                                leave=True):
                _, features = forward_fn(batch, return_feature_list=True)
                feature = features[self.layer_index]
                activation_log.append(normalizer(feature.data.cpu().numpy()))

            self.activation_log = np.concatenate(activation_log, axis=0)
            if self.cache is not None:
                np.save(self.cache, self.activation_log)
        self.index = faiss.IndexFlatL2(self.activation_log.shape[1])
        # res = faiss.StandardGpuResources() 
        # self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        self.index.add(self.activation_log)

    
    @property
    def require_adapt(self):
        return True
    
    @torch.no_grad()    
    def score_batch(self, forward_fn, data: Any):
        output, features = forward_fn(data, return_feature_list=True)
        feature = features[self.layer_index]
        dim = feature.shape[1]
        feature_normed = normalizer(feature.data.cpu().numpy())
        D, _ = self.index.search(
            feature_normed,
            # self.K,
            1000,
        )
        score_dict = {}
        for k in [2, 3, 5, 10, 20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000]:
            kth_dist = D[:, k - 1]
            score_dict[f"knn_{k}"] = -torch.from_numpy(kth_dist)
        
        _, pred = torch.max(torch.softmax(output, dim=1), dim=1)
        return {
            "score_dict": score_dict,
            "preds": pred
        }

    def set_hyperparam(self, hyperparam: list):
        self.K = hyperparam[0]

    def get_hyperparam(self):
        return self.K
    
    def __str__(self) -> str:
        return f"{self.name}_K={self.K}_layer-index={self.layer_index}"