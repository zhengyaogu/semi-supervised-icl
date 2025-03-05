from datasets import load_dataset, load_from_disk
from in_context_ssl.hendrycks_math.utils import process_docs
from in_context_ssl.reasoning.template import *
from collections import defaultdict
import random
import numpy as np
from itertools import chain


##############################################################################################################
# ReasoningDataset: an abstraction for datasets used in reasoning tasks
##############################################################################################################

class ReasoningDataset:

    def __init__(self, seed=42):
        self.ds = load_from_disk("./in_context_ssl/bigbench/data/geometric_shapes_train.hf")

        self.test_ds = load_from_disk("./in_context_ssl/bigbench/data/geometric_shapes_test.hf")
    
    def even_sample_by_type(self, ds, k, style, threshold, quantile, topk, seed):
        assert style in ["ricl", "psl"]

        if style == "ricl":
            threshold = None
        
        if quantile:
            confidences = [inst["confidence"] for inst in ds]
            threshold = np.quantile(confidences, quantile)
        
        if topk:
            confidences = sorted([inst["confidence"] for inst in ds], reverse=True)
            threshold = confidences[k]
        
        type_groups = defaultdict(list)
        for idx, example in enumerate(ds):
            if threshold is not None:
                if example["confidence"] >= threshold:
                    type_groups[example["group"]].append(idx)
            else:
                if style == "ricl":
                    if example["answer"] == example["pred"]:
                        type_groups[example["group"]].append(idx)
                else:
                    type_groups[example["group"]].append(idx)
            
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

    def get_demonstrations(self, demo_file, k, k_gt, style="psl", answer=True, rationale=True, threshold=None, quantile=None, topk=None, seed=42):
        def take_only_relevant_att(d):
            out_d = {
                "question": d["question"],
            }
            if answer:
                out_d["answer"] = d["answer"]
            if rationale:
                out_d["rationale"] = d["rationale"]
            return out_d

        ds = load_from_disk(demo_file)
        random.seed(seed)
        if not answer:
            template = self.instance_template_unsupervised
        else:
            template = self.instance_template
        
        sampled_idx = self.even_sample_by_type(ds, k, style, threshold, quantile, topk, seed)
        sampled_ds = ds.select(sampled_idx).shuffle()

        sampled_idx_gt = self.even_sample_by_type(self.ds, k_gt, style, threshold=None, quantile=None, topk=None, seed=seed)
        sampled_ds_gt = self.ds.select(sampled_idx_gt).shuffle()

        demonstrations = template.connector.join([
            template.format(take_only_relevant_att(d)) for d in chain(sampled_ds_gt, sampled_ds)
        ])

        self.demonstrations = demonstrations
        return demonstrations


    def __len__(self):
        return len(self.test_ds)
    
    def __getitem__(self, i):
        inst = self.test_ds[i]
        q_d = {
            "demonstrations": self.demonstrations,
            "query": inst["question"]
        }
        query = self.template.format(q_d)
        return {
            "query": query,
            **inst
        }
    
    def dynamic_data_selection_iter(self, demo_file, k, answer=True, rationale=True, threshold=None, quantile=None, seed=42):
        def take_only_relevant_att(d):
            out_d = {
                "question": d["question"],
            }
            if answer:
                out_d["answer"] = d["answer"]
            if rationale:
                out_d["rationale"] = d["rationale"]
            return out_d
        demo_ds = load_from_disk(demo_file)
        embeddings = np.array([inst['embedding'] for inst in demo_ds])

        template = self.instance_template

        if quantile:
            confidences = [inst["confidence"] for inst in demo_ds]
            threshold = np.quantile(confidences, quantile)
        
        for inst in self.test_ds:
            test_embedding = inst['embedding']
            distances = np.dot(embeddings, test_embedding)
            top_k_idx = np.argsort(distances)[:k].tolist()
            if threshold:
                top_k_idx = [i for i in top_k_idx if demo_ds[i]["confidence"] > threshold]
            sampled_ds = demo_ds.select(top_k_idx).shuffle()
            demonstrations = template.connector.join([template.format(take_only_relevant_att(d)) for d in sampled_ds])

            q_d = {
                "demonstrations": demonstrations,
                "query": inst["question"]
            }
            query = self.template.format(q_d)
            yield {
                "query": query,
                **inst
            }
            

    def train_iter(self, demo_file, k, answer=True, rationale=True, seed=42):

        demonstrations = self.get_demonstrations(demo_file, 0, k, answer=answer, rationale=rationale, seed=seed)

        for inst in self.ds:
            q_d = {
                "demonstrations": demonstrations,
                "query": inst["question"]
            }
            query = self.template.format(q_d)
            yield {
                "query": query,
                **inst
            }


##############################################################################################################
# LiveBench MATH
##############################################################################################################
class GeometricShapesDataset(ReasoningDataset):
    
    template = BigBenchTemplate()
    instance_template = BigBenchInstanceTemplate()
    instance_template_with_rationale = BigBenchInstanceTemplateWithRationale()
    instance_template_unsupervised = BigBenchInstanceTemplateUnsupervised()

    def __init__(self, seed=42):
        self.ds = load_from_disk("./in_context_ssl/reasoning/data/geometric_shapes_train.hf")

        self.test_ds = load_from_disk("./in_context_ssl/reasoning/data/geometric_shapes_test.hf")

class LiveBenchMathDataset(ReasoningDataset):

    template = LiveBenchMathTemplate()
    instance_template = BigBenchInstanceTemplate()
    instance_template_with_rationale = BigBenchInstanceTemplateWithRationale()
    instance_template_unsupervised = BigBenchInstanceTemplateUnsupervised()

    def __init__(self, seed=42):
        self.ds = load_from_disk("./in_context_ssl/reasoning/data/livebench_math_train.hf")

        self.test_ds = load_from_disk("./in_context_ssl/reasoning/data/livebench_math_test.hf")

class GPQADataset(ReasoningDataset):

    template = LiveBenchMathTemplate()
    instance_template = BigBenchInstanceTemplate()
    instance_template_with_rationale = BigBenchInstanceTemplateWithRationale()
    instance_template_unsupervised = BigBenchInstanceTemplateUnsupervised()

    def __init__(self, seed=42):
        self.ds = load_from_disk("./in_context_ssl/reasoning/data/gpqa_train.hf")

        self.test_ds = load_from_disk("./in_context_ssl/reasoning/data/gpqa_test.hf")

class LiveBenchReasoningDataset(ReasoningDataset):

    template = LiveBenchMathTemplate()
    instance_template = BigBenchInstanceTemplate()
    instance_template_with_rationale = BigBenchInstanceTemplateWithRationale()
    instance_template_unsupervised = BigBenchInstanceTemplateUnsupervised()

    def __init__(self, seed=42):
        self.ds = load_from_disk("./in_context_ssl/reasoning/data/livebench_reasoning_train.hf")

        self.test_ds = load_from_disk("./in_context_ssl/reasoning/data/livebench_reasoning_test.hf")


class Tracking7Dataset(ReasoningDataset):

    template = LiveBenchMathTemplate()
    instance_template = BigBenchInstanceTemplate()
    instance_template_with_rationale = BigBenchInstanceTemplateWithRationale()
    instance_template_unsupervised = BigBenchInstanceTemplateUnsupervised()

    def __init__(self, seed=42):
        self.ds = load_from_disk("./in_context_ssl/reasoning/data/tracking7_train.hf")

        self.test_ds = load_from_disk("./in_context_ssl/reasoning/data/tracking7_test.hf")

class Tracking7Dataset(ReasoningDataset):

    template = LiveBenchMathTemplate()
    instance_template = BigBenchInstanceTemplate()
    instance_template_with_rationale = BigBenchInstanceTemplateWithRationale()
    instance_template_unsupervised = BigBenchInstanceTemplateUnsupervised()

    def __init__(self, seed=42):
        self.ds = load_from_disk("./in_context_ssl/reasoning/data/tracking7_train.hf")

        self.test_ds = load_from_disk("./in_context_ssl/reasoning/data/tracking7_test.hf")

class DateDataset(ReasoningDataset):

    template = LiveBenchMathTemplate()
    instance_template = BigBenchInstanceTemplate()
    instance_template_with_rationale = BigBenchInstanceTemplateWithRationale()
    instance_template_unsupervised = BigBenchInstanceTemplateUnsupervised()

    def __init__(self, seed=42):
        self.ds = load_from_disk("./in_context_ssl/reasoning/data/date_train.hf")

        self.test_ds = load_from_disk("./in_context_ssl/reasoning/data/date_test.hf")


##############################################################################################################
# Translation
##############################################################################################################



class TranslationDatasetBem(ReasoningDataset):

    template = TranslationTemplate("English", "Bemba")
    instance_template = TranslationInstanceTemplate("English", "Bemba")
    instance_template_with_rationale = BigBenchInstanceTemplateWithRationale() #placeholder template
    instance_template_unsupervised = TranslationInstanceTemplateUnsupervised("English")

    def __init__(self, seed=42):
        self.ds = load_from_disk("./in_context_ssl/reasoning/data/flores_bem_train.hf")

        self.test_ds = load_from_disk("./in_context_ssl/reasoning/data/flores_bem_test.hf")

["bem", "kmr", "ewe", "spa", "fra", "deu"]

class TranslationDatasetKmr(ReasoningDataset):

    template = TranslationTemplate("English", "Kurmanji")
    instance_template = TranslationInstanceTemplate("English", "Kurmanji")
    instance_template_with_rationale = BigBenchInstanceTemplateWithRationale() #placeholder template
    instance_template_unsupervised = TranslationInstanceTemplateUnsupervised("English")

    def __init__(self, seed=42):
        self.ds = load_from_disk("./in_context_ssl/reasoning/data/flores_kmr_train.hf")

        self.test_ds = load_from_disk("./in_context_ssl/reasoning/data/flores_kmr_test.hf")

class TranslationDatasetEwe(ReasoningDataset):

    template = TranslationTemplate("English", "Ewe")
    instance_template = TranslationInstanceTemplate("English", "Ewe")
    instance_template_with_rationale = BigBenchInstanceTemplateWithRationale() #placeholder template
    instance_template_unsupervised = TranslationInstanceTemplateUnsupervised("English")

    def __init__(self, seed=42):
        self.ds = load_from_disk("./in_context_ssl/reasoning/data/flores_ewe_train.hf")

        self.test_ds = load_from_disk("./in_context_ssl/reasoning/data/flores_ewe_test.hf")

class TranslationDatasetSpa(ReasoningDataset):

    template = TranslationTemplate("English", "Spanish")
    instance_template = TranslationInstanceTemplate("English", "Spanish")
    instance_template_with_rationale = BigBenchInstanceTemplateWithRationale() #placeholder template
    instance_template_unsupervised = TranslationInstanceTemplateUnsupervised("English")

    def __init__(self, seed=42):
        self.ds = load_from_disk("./in_context_ssl/reasoning/data/flores_spa_train.hf")

        self.test_ds = load_from_disk("./in_context_ssl/reasoning/data/flores_spa_test.hf")

class TranslationDatasetDeu(ReasoningDataset):

    template = TranslationTemplate("English", "German")
    instance_template = TranslationInstanceTemplate("English", "German")
    instance_template_with_rationale = BigBenchInstanceTemplateWithRationale() #placeholder template
    instance_template_unsupervised = TranslationInstanceTemplateUnsupervised("English")

    def __init__(self, seed=42):
        self.ds = load_from_disk("./in_context_ssl/reasoning/data/flores_deu_train.hf")

        self.test_ds = load_from_disk("./in_context_ssl/reasoning/data/flores_deu_test.hf")


class TranslationDatasetFra(ReasoningDataset):

    template = TranslationTemplate("English", "French")
    instance_template = TranslationInstanceTemplate("English", "French")
    instance_template_with_rationale = BigBenchInstanceTemplateWithRationale() #placeholder template
    instance_template_unsupervised = TranslationInstanceTemplateUnsupervised("English")

    def __init__(self, seed=42):
        self.ds = load_from_disk("./in_context_ssl/reasoning/data/flores_fra_train.hf")

        self.test_ds = load_from_disk("./in_context_ssl/reasoning/data/flores_fra_test.hf")


class TranslationDatasetFij(ReasoningDataset):

    template = TranslationTemplate("English", "Fijian")
    instance_template = TranslationInstanceTemplate("English", "Fijian")
    instance_template_with_rationale = BigBenchInstanceTemplateWithRationale() #placeholder template
    instance_template_unsupervised = TranslationInstanceTemplateUnsupervised("English")

    def __init__(self, seed=42):
        self.ds = load_from_disk("./in_context_ssl/reasoning/data/flores_fij_train.hf")

        self.test_ds = load_from_disk("./in_context_ssl/reasoning/data/flores_fij_test_new.hf")

class TranslationDatasetFao(ReasoningDataset):

    template = TranslationTemplate("English", "Faroese")
    instance_template = TranslationInstanceTemplate("English", "Faroese")
    instance_template_with_rationale = BigBenchInstanceTemplateWithRationale() #placeholder template
    instance_template_unsupervised = TranslationInstanceTemplateUnsupervised("English")

    def __init__(self, seed=42):
        self.ds = load_from_disk("./in_context_ssl/reasoning/data/flores_fao_train.hf")

        self.test_ds = load_from_disk("./in_context_ssl/reasoning/data/flores_fao_test_new.hf")

class TranslationDatasetFon(ReasoningDataset):

    template = TranslationTemplate("English", "Fon")
    instance_template = TranslationInstanceTemplate("English", "Fon")
    instance_template_with_rationale = BigBenchInstanceTemplateWithRationale() #placeholder template
    instance_template_unsupervised = TranslationInstanceTemplateUnsupervised("English")

    def __init__(self, seed=42):
        self.ds = load_from_disk("./in_context_ssl/reasoning/data/flores_fon_train.hf")

        self.test_ds = load_from_disk("./in_context_ssl/reasoning/data/flores_fon_test_new.hf")

class TranslationDatasetFuv(ReasoningDataset):

    template = TranslationTemplate("English", "Nigerian Fulfulde")
    instance_template = TranslationInstanceTemplate("English", "Nigerian Fulfulde")
    instance_template_with_rationale = BigBenchInstanceTemplateWithRationale() #placeholder template
    instance_template_unsupervised = TranslationInstanceTemplateUnsupervised("English")

    def __init__(self, seed=42):
        self.ds = load_from_disk("./in_context_ssl/reasoning/data/flores_fuv_train.hf")

        self.test_ds = load_from_disk("./in_context_ssl/reasoning/data/flores_fuv_test_new.hf")

class TranslationDatasetCrh(ReasoningDataset):

    template = TranslationTemplate("English", "Crimean Tatar")
    instance_template = TranslationInstanceTemplate("English", "Crimean Tatar")
    instance_template_with_rationale = BigBenchInstanceTemplateWithRationale() #placeholder template
    instance_template_unsupervised = TranslationInstanceTemplateUnsupervised("English")

    def __init__(self, seed=42):
        self.ds = load_from_disk("./in_context_ssl/reasoning/data/flores_crh_train.hf")

        self.test_ds = load_from_disk("./in_context_ssl/reasoning/data/flores_crh_test_new.hf")

class TranslationDatasetHat(ReasoningDataset):

    template = TranslationTemplate("English", "Haitian Creole")
    instance_template = TranslationInstanceTemplate("English", "Haitian Creole")
    instance_template_with_rationale = BigBenchInstanceTemplateWithRationale() #placeholder template
    instance_template_unsupervised = TranslationInstanceTemplateUnsupervised("English")

    def __init__(self, seed=42):
        self.ds = load_from_disk("./in_context_ssl/reasoning/data/flores_hat_train.hf")

        self.test_ds = load_from_disk("./in_context_ssl/reasoning/data/flores_hat_test_new.hf")

class TranslationDatasetVec(ReasoningDataset):

    template = TranslationTemplate("English", "Venetian")
    instance_template = TranslationInstanceTemplate("English", "Venetian")
    instance_template_with_rationale = BigBenchInstanceTemplateWithRationale() #placeholder template
    instance_template_unsupervised = TranslationInstanceTemplateUnsupervised("English")

    def __init__(self, seed=42):
        self.ds = load_from_disk("./in_context_ssl/reasoning/data/flores_vec_train.hf")

        self.test_ds = load_from_disk("./in_context_ssl/reasoning/data/flores_vec_test_new.hf")

class TranslationDatasetTyv(ReasoningDataset):

    template = TranslationTemplate("English", "Tuvan")
    instance_template = TranslationInstanceTemplate("English", "Tuvan")
    instance_template_with_rationale = BigBenchInstanceTemplateWithRationale() #placeholder template
    instance_template_unsupervised = TranslationInstanceTemplateUnsupervised("English")

    def __init__(self, seed=42):
        self.ds = load_from_disk("./in_context_ssl/reasoning/data/flores_tyv_train.hf")

        self.test_ds = load_from_disk("./in_context_ssl/reasoning/data/flores_tyv_test_new.hf")

class TranslationDatasetSrd(ReasoningDataset):

    template = TranslationTemplate("English", "Sardinian")
    instance_template = TranslationInstanceTemplate("English", "Sardinian")
    instance_template_with_rationale = BigBenchInstanceTemplateWithRationale() #placeholder template
    instance_template_unsupervised = TranslationInstanceTemplateUnsupervised("English")

    def __init__(self, seed=42):
        self.ds = load_from_disk("./in_context_ssl/reasoning/data/flores_srd_train.hf")

        self.test_ds = load_from_disk("./in_context_ssl/reasoning/data/flores_srd_test_new.hf")

class BackTranslationABC(ReasoningDataset):
    
    def __init__(self, data_file):
        self.test_ds = load_from_disk(data_file)
    
    def __getitem__(self, i):
        inst = self.test_ds[i]
        q_d = {
            "demonstrations": self.demonstrations,
            "query": inst["answer"]
        }
        query = self.template.format(q_d)
        return {
            "query": query,
            **inst
        }
        

class BackTranslationDatasetBem(BackTranslationABC):

    template = TranslationTemplate("Bemba", "English")
    instance_template = BackTranslationInstanceTemplate("Bemba", "English")
    instance_template_with_rationale = BigBenchInstanceTemplateWithRationale() #placeholder template
    instance_template_unsupervised = TranslationInstanceTemplateUnsupervised("English")
    
    def __init__(self, data_file):
        super().__init__(data_file)
        self.ds = load_from_disk("./in_context_ssl/reasoning/data/flores_bem_train.hf")


class BackTranslationDatasetFij(BackTranslationABC):

    template = TranslationTemplate("Fijian", "English")
    instance_template = BackTranslationInstanceTemplate("Fijian", "English")
    instance_template_with_rationale = BigBenchInstanceTemplateWithRationale() #placeholder template
    instance_template_unsupervised = TranslationInstanceTemplateUnsupervised("English")
    
    def __init__(self, data_file):
        super().__init__(data_file)
        self.ds = load_from_disk("./in_context_ssl/reasoning/data/flores_fij_train.hf")

