import datasets
from datasets import load_dataset, load_from_disk
from in_context_ssl.reasoning.template import *
import os
import openai
from openai import OpenAI
from tqdm import tqdm
import numpy as np
from pydantic import BaseModel, Field
import json
from in_context_ssl.classification.dataset import *
import re
import pandas as pd
import torchmetrics
import matplotlib.pyplot as plt
from in_context_ssl.classification.constant import *
from in_context_ssl.classification.utils import *
import os

# set your API key here
os.environ["OPENAI_API_KEY"] = "<API Key>"
client = OpenAI()

def inference():
  k_total = 500
  k_gt = 0
  
  ds = ClassificationDataset(task="fp")
  print(ds.get_demonstrations(
      #"in_context_ssl/classification/data/banking_train.hf", 
      # uncomment line above to run inference on original data
      "in_context_ssl/classification/data/fp_psl_k=0_verbalized.hf".format(k_gt),
      #runs inference using pseudo-demos
      k=k_total-k_gt, k_gt=k_gt, data_selection="random", answer=True,
      quantile=0.9
  ))
  
  preds = []
  gold = []
  
  for inst in tqdm(ds):
      query = inst["query"]
      choices = query_openai(client, query, "gpt-4o-mini", n=1, structured_output=True, confidence=False, logprobs=True)
      
      preds.append(choices[0].message.parsed.label)
      gold.append(inst["label"])
  
  preds = np.array(preds)
  gold = np.array(gold)
  acc = (gold == preds).mean().item()

  return acc


def pseudo_demo_generation():
  ds = ClassificationDataset(task="banking")
  
  new_ds_verbalized = []
  
  for inst in tqdm(ds.train_iter(
      "in_context_ssl/classification/data/banking_train.hf",
      k=64, answer=True
  )):
      query = inst["query"]
      choices = query_openai(client, query, "gpt-4o-mini", n=1, structured_output=True, confidence=True, logprobs=True)
      
      o_verbalized = extract_response_classification(choices, confidence="verbalized")
  
      o_verbalized["input"] = inst["input"]
      o_verbalized["gold"] = inst["label"]
  
  
      new_ds_verbalized.append(o_verbalized)
  
  # save the generated dataset
  datasets.Dataset.from_pandas(pd.DataFrame(new_ds_verbalized)).save_to_disk("in_context_ssl/classification/data/{your file name}.hf")

def iter_psd():
  k_gt = 16
  chunk_size = 500
  demo_size_cap = 1000
  
  ds = ClassificationDataset(task="banking")
  demo_gt = ds.get_demonstrations(
      "in_context_ssl/classification/data/banking_train.hf", 
      k=0, k_gt=16, answer=True, data_selection="random", seed=42
  )
  demo_psl = ""
  labeled_indices = set()
  psl_ds = None
  
  labeled_ds = None
  
  while len(labeled_indices) < len(ds.ds):
      idx_curr = ds.get_demonstrations_iterative(ds.ds, chunk_size, labeled_indices, eps=0.5)
      #print("overlapped idx", len([i for i in idx_curr if i in labeled_indices]))
      ds_curr = ds.ds.select(idx_curr)
  
      new_ds = []
      preds = []
      gold = []
      preds_filtered = []
      gold_filtered = []
  
      demo_curr = demo_gt + ds.instance_template.connector + demo_psl if len(demo_psl) > 0 else demo_gt
      #print(demo_curr)
      for i, inst in tqdm(zip(idx_curr, ds_curr)):
          q_d = {
              "demonstrations": demo_curr,
              "query": inst["input"],
              "labels": ds.labels
          }
          query = ds.template.format(q_d)
          query_d = {
              "query": query,
              **inst
          }
  
          labeled_indices.add(i)
      
          choices = query_openai(client, query, "gpt-4o-mini", n=1, structured_output=True, confidence=True, logprobs=True)
          o = extract_response_classification(choices, confidence="verbalized")
          o["input"] = inst["input"]
          new_ds.append(o)
          preds.append(o["label"])
          gold.append(inst["label"])
          if o["confidence"] > 0.9:
              preds_filtered.append(o["label"])
              gold_filtered.append(inst["label"])
  
      
      new_ds = datasets.Dataset.from_pandas(pd.DataFrame(new_ds))
      if labeled_ds is None:
          labeled_ds = new_ds
      else:
          labeled_ds = datasets.concatenate_datasets([labeled_ds, new_ds])
          
      new_ds = new_ds.filter(lambda x: x["confidence"] > 0.9)
  
      if psl_ds is None:
          psl_ds = new_ds
      else:
          psl_ds = datasets.concatenate_datasets([psl_ds, new_ds])
  
      def take_only_relevant_att(d):
          out_d = {
              "input": d["input"],
          }
          out_d["label"] = d["label"]
          return out_d
      
      if len(psl_ds) > demo_size_cap:
          sampled_idx = ds.even_sample_by_type(psl_ds, k=demo_size_cap - k_gt, threshold=None, quantile=None, topk=demo_size_cap - k_gt, seed=42)
          sampled_demo_ds = psl_ds.select(sampled_idx)
          demo_psl = ds.instance_template.connector.join([
              ds.instance_template.format(take_only_relevant_att(d)) for d in sampled_demo_ds
          ])
      else:
          demo_psl = ds.instance_template.connector.join([
              ds.instance_template.format(take_only_relevant_att(d)) for d in psl_ds
          ])
      
      preds = np.array(preds)
      gold = np.array(gold)
      print("chunk acc: ", (preds == gold).mean().item())
      preds_filtered = np.array(preds_filtered)
      gold_filtered = np.array(gold_filtered)
      print("filtered chunk acc: ", (preds_filtered == gold_filtered).mean().item())
      print("len labeled_indices", len(labeled_indices))
  
      datasets.Dataset.from_pandas(pd.DataFrame(new_ds)).save_to_disk("in_context_ssl/classification/data/{your file name}.hf")
