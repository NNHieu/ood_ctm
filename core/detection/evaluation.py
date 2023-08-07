import torch
from torch.utils.data import Subset
import numpy as np
import pandas as pd
from ood_metrics import calc_metrics
from tqdm import tqdm
from .methods.post_hoc.base_detector import BaseDetector
import logging
logger = logging.getLogger(__name__)

RECALL_LEVEL_DEFAULT = 0.95

def compute_metrics(_pos, _neg, recall_level=RECALL_LEVEL_DEFAULT):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    metric_dict = calc_metrics(examples, labels)
    auroc = metric_dict['auroc']
    aupr = metric_dict['aupr_in']
    fpr = metric_dict['fpr_at_95_tpr']

    return auroc, aupr, fpr


def score_loop(forward_fn, detector: BaseDetector, loader, in_dist=False, leave_tqdm=True):
    score_dict = {}
    aux = {}
    for batch in tqdm(loader, leave=leave_tqdm):
        # logger.info("Loaded data")
        result_dict = detector.score_batch(forward_fn, batch)
        # logger.info("Computed scores")
        for k,v in result_dict['score_dict'].items():
            score_dict[k] = score_dict.get(k, []) + [v]
        del result_dict['score_dict']
        for k, v in result_dict.items():
            aux[k] = aux.get(k, []) + [v]
    # scores = torch.cat(scores, dim=0).cpu()
    score_dict = {k: torch.cat(v, dim=0).cpu() for k, v in score_dict.items()}
    aux = {k: torch.cat(v, dim=0).cpu() for k, v in aux.items()}
    return score_dict, aux


def _random_subset_loader(dataset, num_examples, bs, num_workers=2):
    '''
    Create a DataLoader from a random sub-dataset
    '''
    if num_examples is not None:
        indices = torch.randperm(len(dataset))[:num_examples].tolist()
        # ood_data_indices = np.arange(ood_num_examples)
        sub_data = Subset(dataset, indices)
    else:
        sub_data = dataset
    return torch.utils.data.DataLoader(sub_data, batch_size=bs,
                                       shuffle=False,
                                       num_workers=num_workers)


# def get_results(forward_fn, detector, get_measures, ood_loader, num_runs=5):
#     measures = []
#     for _ in range(num_runs):
#         out_score, _ = score_loop(forward_fn, detector, ood_loader, in_dist=False)
#         out_score = out_score.numpy()
#         measures.append(get_measures(out_score))
#     measures = zip(*measures)
#     return measures


class Evaluator():
    '''
    Run experiments that fit in one table
    '''
    
    def __init__(self, ood_num_examples, test_bs, num_to_avg: int) -> None:
        self.in_score_dict = {}
        
        self.ood_num_examples = ood_num_examples
        self.test_bs = test_bs
        self.num_to_avg = num_to_avg
        self.reset()
    
    @torch.no_grad()
    def eval_in_acc(self, forward_fn, id_name, id_set, num_workers=2):
        in_loader = torch.utils.data.DataLoader(id_set, 
                                                batch_size=self.test_bs, shuffle=False,
                                                num_workers=num_workers)
        num_corrects = 0
        for batch in tqdm(in_loader):
            logits = forward_fn(batch).cpu()
            preds = logits.argmax(dim=1)
            # print(preds, end=',')
            num_corrects += (preds == batch[1]).sum()
        # print(f"In-Dist Acc [{id_name}]: {(num_corrects / len(id_set))*100:0.4f}%")
        return num_corrects / len(id_set)

    def compute_in_score(self, forward_fn, detector, id_name, id_set, num_workers=2):
        in_loader = torch.utils.data.DataLoader(id_set, 
                                                batch_size=self.test_bs, shuffle=False,
                                                num_workers=num_workers)
        in_score_dict, ind_aux = score_loop(forward_fn,
                                detector,
                                in_loader,
                                in_dist=True)
        self.in_score_dict[id_name] = in_score_dict
        self.id_aux[id_name] = ind_aux
    
    def compute_ood_score(self, forward_fn, detector, ood_data):
        ood_loader = _random_subset_loader(ood_data, len(ood_data), self.test_bs) # Get a different subset
        out_scores_dict, _ = score_loop(forward_fn, detector, ood_loader, in_dist=False, leave_tqdm=False)
        # out_scores = out_scores.numpy()
        return out_scores_dict

    def eval_ood(self, forward_fn, detector, id_name, ood_name, ood_data, verbose=False, is_near=False):
        assert self.in_score_dict is not None and id_name in self.in_score_dict
        in_scores_dict = self.in_score_dict[id_name]
        
        run_measures = {}
        self.ood_score_dict[ood_name] = []
        self.ood_aux[ood_name] = []
        num_to_avg = self.num_to_avg
        if self.ood_num_examples[ood_name] is None or len(ood_data) <= self.ood_num_examples[ood_name]:
            num_to_avg = 1
        for _ in tqdm(range(num_to_avg), desc=f"Multiple Run on ood subset from {ood_name}"):
            ood_loader = _random_subset_loader(ood_data, self.ood_num_examples[ood_name], self.test_bs) # Get a different subset
            out_scores_dict, ood_aux = score_loop(forward_fn, detector, ood_loader, in_dist=False, leave_tqdm=False)
            # out_scores = out_scores.numpy()
            self.ood_score_dict[ood_name].append(out_scores_dict)
            self.ood_aux[ood_name].append(ood_aux)
            # run_measures.append(compute_metrics(-in_scores, -out_scores)) # ID as positive
            for k,id_scores in in_scores_dict.items():
                run_measures[k] = run_measures.get(k, []) + [compute_metrics(id_scores, out_scores_dict[k])]
            # run_measures.append(compute_metrics(in_scores, out_scores)) # ID as positive
        for k in run_measures.keys():
            run_measures[k] = list(zip(*run_measures[k]))

            run_measures[k] = np.array(run_measures[k]) * 100
            auroc, aupr, fpr = run_measures[k].mean(axis=-1).tolist()
            auroc_std, aupr_std, fpr_std = run_measures[k].std(axis=-1).tolist()

            record = [id_name, ood_name, is_near,
                    str(detector), str(k), 
                    num_to_avg, 
                    auroc, aupr, fpr,
                    auroc_std, aupr_std, fpr_std]
            self.df.loc[len(self.df)] = record
        if True:
            print(self.df.loc[len(self.df) - len(run_measures)])

    def reset(self):
        self.df = pd.DataFrame(columns=("id_data", "ood_data", "is_near",
                                        "detector", "version",
                                        "num_runs",
                                        "auroc", "aupr", "fpr",
                                        "auroc_std", "aupr_std", "fpr_std"))
        self.ood_score_dict = {}
        self.in_score_dict = {}
        self.ood_aux = {}
        self.id_aux = {}