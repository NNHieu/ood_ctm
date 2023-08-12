from pathlib import Path
import torch
import torch.nn.functional as F
from core.detection.methods.utils import CalMeanClass
from  .base_detector import BaseDetector
from .ash import ash_s

energy_score = lambda output, T: -T*torch.logsumexp(output / T, dim=1)


class CTMDetector(BaseDetector):
    def __init__(self, *, ref_dir_type='class_mean', shift_type=None, cache_class_means=None, cache_feats=None, layer_index=0, **kwargs) -> None:
        super().__init__('ctm')
        self.cache_class_means = None
        if cache_class_means is not None:
            self.cache_class_means = Path(cache_class_means)
            self.cache_class_means.parent.mkdir(parents=True, exist_ok=True)
        self.cache_feats = cache_feats
        self.layer_index = layer_index
        self.normalize_feature = True
        self.ref_dir_type = ref_dir_type
        self.shift_type = shift_type
    
    @torch.no_grad()
    def adapt(self, forward_fn, ref_id_dataloader, *, net, **kwargs):
        calmean = CalMeanClass(layer_index = self.layer_index)
        if self.cache_class_means is not None and self.cache_class_means.exists():
            print(f"Loading cached class means from {self.cache_class_means}")
            calmean.load_cached_state(self.cache_class_means)
        else:
            calmean.adapt(forward_fn, ref_id_dataloader, cache_feats_path=self.cache_feats, cache_mean_path=self.cache_class_means, normalize=False)
            if self.cache_class_means is not None:
                calmean.save_cached_state(self.cache_class_means)
        
        class_means = calmean.class_means_
        global_mean = calmean.global_mean_

        self._shift =  -global_mean
        device = net.fc.weight.data.device
        ref_dir ={
            "class_mean": F.normalize(class_means, dim=-1).to(device),
            # "class_mean": class_means.to(device),

            "last_weight": F.normalize(net.fc.weight.data, dim=-1).to(device),
            "shifted_class_mean": F.normalize(class_means + self._shift, dim=-1).to(device),
            "shifted_last_weight": F.normalize(net.fc.weight.data, dim=-1).to(device),
            "global_mean": F.normalize(global_mean, dim=-1).to(device),
        }
        self._last_weight_norms = torch.norm(net.fc.weight.data, dim=-1)
        self._ref_dir = ref_dir
        # self._ref_dir = F.normalize(net.fc.weight.data, dim=-1)
    
    @torch.no_grad()
    def score_batch(self, forward_fn, data):
        logits, feat_list = forward_fn(data, return_feature_list=True)
        preds = torch.argmax(logits, dim=1)
        results = {'preds': preds,}


        feats = feat_list[self.layer_index]
        shifted_feats = feats + self._shift
        
        results['feat_norm'] = torch.norm(feats, dim=-1)
        results['shifted_feat_norm'] = torch.norm(shifted_feats, dim=-1)
        probs = F.softmax(logits, dim=-1)
        results['probs'] = probs.cpu()

        # feats = ash_s(feats.view(feats.shape[0], -1, 1, 1)).view_as(feats)
        if self.normalize_feature:
            feats = F.normalize(feats, dim=-1)
            shifted_feats = F.normalize(shifted_feats, dim=-1)
        

        scores = {}
        for key, ref_dir in self._ref_dir.items():
            if 'shifted' not in key:
                logits = (feats @ ref_dir.T)
                scores[key] = logits.cpu().max(dim=-1).values
                # scores[key+'_energy'] = -energy_score(logits, 1).cpu()
                if key == "class_mean":
                    probs = F.softmax(feats @ self._ref_dir["last_weight"].T, dim=-1)
                    C = probs.shape[1]
                    probs_norms = torch.norm(probs, dim=-1, keepdim=True)
                    # if_probs = (probs - 1/C) / torch.sqrt(((1 - 1/C) * (probs_norms**2 - 1/C) ))
                    weights_norms = self._last_weight_norms
                    if_scores = (weights_norms.unsqueeze(0) * logits)
                    if_scores = if_scores[torch.arange(preds.shape[0]), preds]
                    # if_scores = if_scores.max(dim=-1).values
                    scores['influence'] = if_scores.cpu()
            else: 
                logits = (shifted_feats @ ref_dir.T)
                scores[key] = logits.cpu().max(dim=-1).values
                # scores[key+'_energy'] = -energy_score(logits, 1).cpu()
        
        scores['global_mean'] = -scores['global_mean']
        
        results['score_dict'] = scores
        # results.update(scores)
        # if self.shift_type is not None:
        #     if self.ref_dir_type == "class_mean":
        #         results['scores'] = scores['shifted_class_mean']
        #     elif self.ref_dir_type == "last_weight":
        #         results['scores'] = scores['shifted_last_weight']
        # else:
        #     results['scores'] = scores[self.ref_dir_type]

        return results
    
    def __str__(self) -> str:
        # name = f"{self.name}_layer={self.layer_index}_ref={self.ref_dir_type}" 
        # if self.shift_type is not None:
        #     name += f"_shift={self.shift_type}"
        name = f"{self.name}_layer={self.layer_index}" 
        return name
