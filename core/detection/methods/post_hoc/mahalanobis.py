from __future__ import print_function
import copy
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import numpy as np
from  .base_detector import BaseDetector

import logging

logger = logging.getLogger(__name__)

noise_magninutes = [0, 0.0005, 0.001, 0.0014, 0.002, 0.0024, 0.005, 0.01, 0.05, 0.1, 0.2]

class MahalanobisDetector(BaseDetector):
    def __init__(self, *, noise_magnitude, layer_index=0, DATASET_MEAN, DATASET_STD, cache=None, normalize=False, **kwargs) -> None:
        super().__init__('mahalanobis')
        self.magnitude = noise_magnitude
        self.layer_index = layer_index
        self.sample_mean = None
        self.precision = None
        self.DATASET_MEAN = DATASET_MEAN
        self.DATASET_STD = DATASET_STD
        self.normalize = normalize
        self.cache = None
        if cache is not None:
            self.cache = Path(cache)
            self.cache.parent.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def adapt(self, forward_fn, ref_id_dataloader, num_classes, **kwargs):
        '''
        Compute the mean and correlation matrix for ref id data 
        '''
        self.num_classes=num_classes
        if self.cache is not None and self.cache.exists():
            self.sample_est = torch.load(self.cache, map_location="cuda")
        else:
            self.sample_est = sample_estimator(forward_fn, 
                                                            self.num_classes, 
                                                            ref_id_dataloader, normalize=self.normalize)
            if self.cache is not None:
                torch.save(self.sample_est, self.cache)
        # for key, val in self.sample_est.items():
        #     self.sample_est[key] = [t.cuda() for t in val]
                                                
    @property
    def require_adapt(self):
        return True

    def _get_feature(self, forward_fn, batch):
        logits, feature_list = forward_fn(batch, return_feature_list=True)
        out_features = feature_list[self.layer_index]
        # if self.normalize:
        #     return F.normalize(out_features, dim=-1)
        return out_features
    
    def _get_est(self, normalize=False):
        if normalize:
            return self.sample_est["normalized"]
        else:
            return self.sample_est["normal"]

    # @torch.no_grad()
    def _score_batch(self, forward_fn, batch, normalize=False):
        layer_index = self.layer_index
        original_data = copy.deepcopy(batch[0])
        data: torch.Tensor = batch[0].cuda()
        data = data.requires_grad_(True) # Set requires_grad to True for input data
        batch[0] = data
        layer_index = self.layer_index
        out_features = self._get_feature(forward_fn, batch)
        if normalize:
            out_features = F.normalize(out_features, dim=-1)
        # compute Mahalanobis score
        with torch.no_grad():
            gaussian_scores = self._gaussian_score(out_features, layer_index, normalize=normalize)
            md_preds = gaussian_scores.argmax(1)
        score_dict = {"normalized" if normalize else "normal": gaussian_scores.max(1).values}
        md = self._gaussian_score(out_features, layer_index, normalize=normalize, preds=md_preds)
        loss = torch.mean(md)
        loss.backward()
        
        gradient = torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / self.DATASET_STD[0])
        gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / self.DATASET_STD[1])
        gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / self.DATASET_STD[2])
        
        # score_dict = {}
        with torch.no_grad():
            original_data = original_data.cuda()
            for magnitude in noise_magninutes:
                tempInputs = original_data - magnitude*gradient
                # noise_out_features = forward_fn((tempInputs,), return_feature_list=True)[1][layer_index]
                noise_out_features = self._get_feature(forward_fn, (tempInputs, batch[1]))
                if normalize:
                    noise_out_features = F.normalize(noise_out_features, dim=-1)
                noise_gaussian_scores = self._gaussian_score(noise_out_features, layer_index, normalize=normalize)
                noise_gaussian_scores = torch.max(noise_gaussian_scores, dim=1).values
                score_dict[f"noise_mag={magnitude}" + ("_normed" if normalize else "")] = noise_gaussian_scores
        return score_dict, md_preds
    
    def score_batch(self, forward_fn, batch):
        score_dict, md_preds = self._score_batch(forward_fn, copy.deepcopy(batch), normalize=False)
        normalized_score_dict, _ = self._score_batch(forward_fn, batch, normalize=True)
        score_dict.update(normalized_score_dict)

        # tempInputs = torch.add(data.data, gradient, alpha=-self.magnitude)
        # with torch.no_grad():
        #     noise_out_features = forward_fn((tempInputs,), return_feature_list=True)[1][layer_index]
        #     noise_gaussian_scores = self._gaussian_score(noise_out_features, layer_index)
        #     noise_gaussian_scores = torch.max(noise_gaussian_scores, dim=1).values
        return {
            "score_dict": score_dict,
            "preds": md_preds
        }
    
    def _gaussian_score(self, out_features, layer_index, normalize=False, preds=None):
        sample_mean, precision = self._get_est(normalize=normalize)
        if preds is None:
            gaussian_scores = []
            for i in range(self.num_classes):
                centered_feats = out_features - sample_mean[layer_index][i]
                term_gau = -0.5*torch.mm(torch.mm(centered_feats, precision[layer_index]), centered_feats.t()).diag()
                gaussian_scores.append(term_gau.view(-1,1))
            return torch.cat(gaussian_scores, 1)
        else:
            batch_sample_mean = sample_mean[layer_index][preds] # Get the mean of the predicted class
            centered_feats = out_features - batch_sample_mean
            md = 0.5*((centered_feats @ precision[layer_index]) @ centered_feats.T).diag()
            return md
    
    def __str__(self) -> str:
        return f"{self.name}_layerindex={self.layer_index}"


@torch.no_grad()
def sample_estimator(forward_fn, num_classes, train_loader, normalize=False):
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of class mean
             precision: list of precisions
    """
    sample_input = next(iter(train_loader))
    feature_list = forward_fn(sample_input, return_feature_list=True)[1]
    feature_size_list = np.array([out.size(1) for out in feature_list])
    device = "cpu"
    logger.info(f'Get sample mean and covariance. Num feature layers: {len(feature_size_list)}')
    list_features = []
    labels = []


    correct, total = 0, 0
    for batch in train_loader:
        target = batch[1]
        total += target.size(0)
        logits, out_features_list = forward_fn(batch, return_feature_list=True)
        pred = logits.cpu().argmax(1)
        correct += (pred == target).sum()
        out_features_list = [feats.cpu() for feats in out_features_list]
        list_features.append(out_features_list)
        labels.append(target)
    labels = torch.cat(labels).cpu()
    logger.info('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))
    
    def _estimate(list_features):
        sample_class_mean_list = []
        precision_list = []
        for k in range(len(list_features)):
            class_means = torch.Tensor(num_classes, feature_size_list[k]).to(device)
            for c in range(num_classes):
                c_inds = (labels == c)
                if c_inds.sum() > 0:
                    class_means[c] = torch.mean(list_features[k][c_inds], dim=0)
                    list_features[k][c_inds] -= class_means[c] # center the data by subtracting class mean

            sample_class_mean_list.append(class_means)
            # X = list_features[k].cpu().numpy()
            # group_lasso.fit(X)
            # temp_precision = group_lasso.precision_
            # temp_precision = torch.from_numpy(temp_precision).float().to(device)
            cov_k = (list_features[k].T @ list_features[k]) / list_features[k].shape[0] # get the covariance matrix
            precision_k = torch.linalg.pinv(cov_k, hermitian=True) # get the precision (inverse covariance) matrix
            precision_list.append(precision_k)
            return sample_class_mean_list, precision_list

    # list_features = [torch.cat(feats) for feats in zip(*list_features)]
    results = {}
    list_features = [torch.cat(feats) for feats in zip(*list_features)]
    results["normal"] = _estimate(copy.deepcopy(list_features))
    list_features = [F.normalize(feats, dim=-1) for feats in list_features]
    results["normalized"] = _estimate(list_features)

    return results
