import datasets
from datasets import load_dataset, load_from_disk
from in_context_ssl.reasoning.template import *
import os
import openai
from openai import OpenAI
from tqdm import tqdm
import numpy as np
import json
from in_context_ssl.reasoning.utils import *
from in_context_ssl.reasoning.dataset import *
import re
import pandas as pd
from in_context_ssl.grading.grader import grade_answer
import matplotlib.pyplot as plt

def process_doc(doc: dict):
    out_doc = {
        "question": doc["turns"][0],
        "answer": doc["ground_truth"],
        "group": doc["task"]
    }
    return out_doc

def preprocess_data():
    ds = load_dataset("livebench/math")["test"]

    train_datasets = []
    test_datasets = []

    subtasks = ["math_comp", "AMPS_Hard", "olympiad"]
    remove_columns = [k for k in ds.features.keys() if k not in ["turns", "ground_truth"]]
    for subtask in subtasks:
        print(len(ds))
        ds_curr = ds.filter(lambda x: x["task"] == subtask)
        cutoff = int(len(ds_curr) * 0.75)
        print(cutoff)
        ds_curr = ds_curr.shuffle()
        ds_train = ds_curr.select(range(cutoff)).map(process_doc, remove_columns=remove_columns)
        ds_test = ds_curr.select(range(cutoff, len(ds_curr))).map(process_doc, remove_columns=remove_columns)
        train_datasets.append(ds_train)
        test_datasets.append(ds_test)

    ds_train = datasets.concatenate_datasets(train_datasets)
    ds_test = datasets.concatenate_datasets(test_datasets)

    ds_train.save_to_disk("in_context_ssl/reasoning/data/livebench_math_train.hf")
    ds_test.save_to_disk("in_context_ssl/reasoning/data/livebench_math_test.hf")

def add_embedding(doc: dict):
    out_doc = {
        "embedding": client.embeddings.create(
            input = [doc["question"]], model="text-embedding-3-large"
        ).data[0].embedding
    }
    return out_doc
ds = load_from_disk("in_context_ssl/reasoning/data/livebench_math_test.hf")
ds = ds.map(add_embedding)
ds.save_to_disk("in_context_ssl/reasoning/data/livebench_math_test_new.hf")

def inference():
  preds = []
  gold = []
  rationales = []
  messages = []
  
  ds = LiveBenchMathDataset()
  print(ds.get_demonstrations(
      "in_context_ssl/reasoning/data/livebench_math_psl_sc.hf",
      k=3, k_gt=0, style="psl", answer=True, rationale=True, 
      quantile=0.9, seed=42
  ))
  
  for inst in tqdm(ds):
      choice = query_openai(client, inst["query"], model="gpt-4o-mini", structured_output=False, confidence=False)[0]
      
      o = parse_output_livebench_math(choice.message.content)
  
      preds.append(o["answer"])
      gold.append(inst["answer"])
      rationales.append(o["rationale"])
      messages.append(choice.message.content)
  
  correct = np.array([grade_answer(p, g) for p, g in zip(preds, gold)]).astype(float)
  correct.mean()

def Naive_Semi_ICL_annotation():
  ds = LiveBenchMathDataset()
  
  preds = []
  gold = []
  rationales = []
  messages = []
  
  for inst in tqdm(ds.mot_iter(
      "in_context_ssl/reasoning/data/livebench_math_psl_verbalized.hf",
      k=4, answer=True, rationale=True,
      threshold=0.9, seed=42
  )):
      choice = query_openai(client, inst["query"], model="gpt-4o-mini", structured_output=False, confidence=False, logprobs=False)[0]
  
      o = parse_output_livebench_math(choice.message.content)
  
      preds.append(o["answer"])
      gold.append(inst["answer"])
      rationales.append(o["rationale"])
      messages.append(choice.message.content)



if __name__ == "__main__":
  #add API key
  api_key = ""
  os.environ["OPENAI_API_KEY"] = api_key
  client = OpenAI()
