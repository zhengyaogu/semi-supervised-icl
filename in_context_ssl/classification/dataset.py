from datasets import load_dataset, load_from_disk
from in_context_ssl.hendrycks_math.utils import process_docs
from in_context_ssl.classification.template import *
from collections import defaultdict
import random
import numpy as np
from itertools import chain
import torch


class ClassificationDataset:

    template = ClassificationTemplate()
    instance_template = ClassificationInstanceTemplate()

    def __init__(self, task, seed=42):
        self.ds = load_from_disk("./in_context_ssl/classification/data/{}_train.hf".format(task))
        self.test_ds = load_from_disk("./in_context_ssl/classification/data/{}_test.hf".format(task))
        self.labels = np.unique(self.ds["label"]).tolist()
    
    def even_sample_by_type(self, ds, k, threshold, quantile, topk, seed):
        
        if quantile:
            confidences = [inst["confidence"] for inst in ds]
            threshold = np.quantile(confidences, quantile)
        
        if topk:
            confidences = sorted([inst["confidence"] for inst in ds], reverse=True)
            threshold = confidences[k]
        print(threshold)
        
        type_groups = defaultdict(list)
        for idx, example in enumerate(ds):
            if threshold is not None:
                if example["confidence"] >= threshold:
                    type_groups[example["label"]].append(idx)
            else:
                type_groups[example["label"]].append(idx)
            
        assert k <= sum([len(v) for v in type_groups.values()]), "not enough available data"

        random.seed(seed)

        k_by_type = defaultdict(int)
        rem_k_by_type = {k: len(type_groups[k]) for k in type_groups}
        rem_k = k
        while rem_k > 0 and rem_k_by_type:
            base_k = rem_k // len(rem_k_by_type.keys())
            if base_k == 0: # if the number of remaining k cannot be evenly spread over the groups, randomly sample
                            # rem_k keys and distribute the rem_k
                available_keys = [k for k in rem_k_by_type.keys() if rem_k_by_type[k] > 0]
                sample_k = min(rem_k, len(available_keys))
                lucky_keys = random.sample(available_keys, k=sample_k)
                for k in lucky_keys:
                    k_by_type[k] += 1
                break
            min_k = min(base_k, min(rem_k_by_type.values()))
            for k in rem_k_by_type:
                k_by_type[k] += min_k
                rem_k_by_type[k] -= min_k
                rem_k -= min_k
            rem_k_by_type = {k: v for k, v in rem_k_by_type.items() if v > 0}
        
        sampled_idx = []
        for k in type_groups:
            sampled_idx.extend(random.sample(type_groups[k], k=k_by_type[k]))
        return sampled_idx

    def get_demonstrations(self, demo_file, k, k_gt, data_selection="random", answer=True, threshold=None, quantile=None, topk=None, seed=42):
        assert data_selection in ["random", "eps-knn"]
        def take_only_relevant_att(d):
            out_d = {
                "input": d["input"],
            }
            if answer:
                out_d["label"] = d["label"]
            return out_d
        
        ds = load_from_disk(demo_file)
        random.seed(seed)
        if not answer:
            template = self.instance_template_unsupervised
        else:
            template = self.instance_template
        
        if data_selection == "random":
            sampled_idx = self.even_sample_by_type(ds, k, threshold, quantile, topk, seed)
        elif data_selection == "":
            sampled_idx = None
        sampled_ds = ds.select(sampled_idx).shuffle()

        sampled_idx_gt = self.even_sample_by_type(self.ds, k_gt, threshold=None, quantile=None, topk=None, seed=seed)
        sampled_ds_gt = self.ds.select(sampled_idx_gt).shuffle()

        demonstrations = template.connector.join([
            template.format(take_only_relevant_att(d)) for d in chain(sampled_ds_gt, sampled_ds)
        ])

        self.demonstrations = demonstrations
        return demonstrations
    
    def get_demonstrations_iterative(self, ds, k, labeled_indices: set, eps):
        print("len labeled idx", len(labeled_indices))
        if not labeled_indices:
            random_idx = random.sample(range(len(ds)), k=min(k, len(ds)))
            return random_idx

        rem_idx = [i for i in range(len(ds)) if i not in labeled_indices]

        k = min(len(rem_idx), k)
        
        k_random = int(eps * k)
        k_nn = k - k_random

        labeled_embeddings = torch.Tensor([
            inst["embedding"] for inst in ds.select(labeled_indices)
        ]) # N1 * D

        rem_idx = torch.tensor(rem_idx)
        
        ds_rem = ds.select(rem_idx.tolist())
        embeddings = torch.Tensor([
            inst["embedding"] for inst in ds_rem
        ]) # N2 * D

        labeled_norm = torch.norm(labeled_embeddings, dim=1)
        unlabeled_norm = torch.norm(embeddings, dim=1)
        normalizer = torch.outer(labeled_norm, unlabeled_norm) # N1 * N2
        dists = (labeled_embeddings @ embeddings.t()) / normalizer # N1 * N2
        nearest_dists, _ = dists.min(dim=0) # N2
        _, nearest_idx = torch.topk(nearest_dists, k_nn, largest=False, sorted=False)
        nearest_idx = nearest_idx.tolist()
        nearest_idx = rem_idx[nearest_idx].tolist()
        
        nearest_idx_set = set(nearest_idx)
        random_idx = random.sample([
            i for i in rem_idx.tolist() if i not in nearest_idx_set
        ], k=k_random)

        assert len(random_idx + nearest_idx) == len(set(random_idx + nearest_idx))

        return random_idx + nearest_idx
                
        
    
    def train_iter(self, demo_file, k, answer=True, seed=42):

        demonstrations = self.get_demonstrations(demo_file, k=0, k_gt=k, answer=answer, data_selection="random", seed=seed)

        for inst in self.ds:
            q_d = {
                "demonstrations": demonstrations,
                "query": inst["input"],
                "labels": self.labels
            }
            query = self.template.format(q_d)
            yield {
                "query": query,
                **inst
            }

    def __len__(self):
        return len(self.test_ds)
    
    def __getitem__(self, i):
        inst = self.test_ds[i]
        q_d = {
            "demonstrations": self.demonstrations,
            "query": inst["input"],
            "labels": self.labels
        }
        query = self.template.format(q_d)
        return {
            "query": query,
            **inst
        }