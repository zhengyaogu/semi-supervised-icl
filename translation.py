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
from in_context_ssl.reasoning.utils import *
from in_context_ssl.reasoning.dataset import *
import re
import pandas as pd
from in_context_ssl.reasoning.utils import *
import torchmetrics
import matplotlib.pyplot as plt

def create_translation_dataset(target_lang, stage):
    split = "dev" if stage == "train" else "devtest"
    ds = load_dataset("openlanguagedata/flores_plus")[split]
    df  = ds.to_pandas()

    def add_embedding(doc):
        out_doc = {
            "embedding": client.embeddings.create(
                input=[doc["question"]],
                model="text-embedding-3-large"
            ).data[0].embedding
        }
        return out_doc

    df_source = df[df["iso_639_3"] == "eng"]
    df_target = df[df["iso_639_3"] == target_lang]
    joined_df = pd.merge(df_source, df_target, on="id", how="inner")[[
        "text_x", "text_y", "topic_x"
    ]]
    joined_df = joined_df.rename(columns={
        "text_x": "question",
        "text_y": "answer",
        "topic_x": "group"
    })

    ds = datasets.Dataset.from_pandas(joined_df)
    ds = ds.map(add_embedding)
    ds.save_to_disk("in_context_ssl/reasoning/data/flores_{}_{}.hf".format(target_lang, stage))
    return ds

def inference():
  k_total = 100
  k_gt = 0
  
  ds = TranslationDatasetSrd()
  print(ds.get_demonstrations(
      "in_context_ssl/reasoning/data/flores_srd_psl_k={}_verbalized.hf".format(k_gt),
      k=k_total-k_gt, k_gt=k_gt, 
      style="psl", answer=True, rationale=False, quantile=0.9, topk=False, seed=42
  ))
  preds = []
  gold = []
  messages = []
  
  for inst in tqdm(ds):
      choice = query_openai(client, inst["query"], model="gpt-4o-mini", n=1, structured_output=False, confidence=False, logprobs=True)[0]
      o = parse_output_translation("Sardinian", choice.message.content)
      messages.append(choice.message.content)
      preds.append(o["answer"])
      gold.append(inst["answer"])
  
  chrf = torchmetrics.CHRFScore(return_sentence_level_score=True)
  chrf(preds, gold)
  score = chrf.compute()[1].mean()

  return score

def naive_semi_ICL_annotation():
  preds = []
  gold = []
  confidences = []
  messages = []
  
  new_ds_verbalized = []
  new_ds_entropy = []
  
  ds = TranslationDatasetBem()
  
  k=0
  for inst in tqdm(ds.train_iter(
      "in_context_ssl/reasoning/data/flores_bem_train.hf",
      k=k, answer=True, rationale=False, seed=42
  )):
      choices = query_openai(client, inst["query"], n=1, model="gpt-4o-mini", structured_output=False, confidence=True, logprobs=True)
  
      o_verbalized = aggregate(choices, parser=lambda x: parse_output_translation("Bemba", x), confidence="verbalized", rationale=False)
      o_entropy = aggregate(choices, parser=lambda x: parse_output_translation("Bemba", x), confidence="entropy", rationale=False)
  
      d_verbalized = {
          "question": inst["question"],
          "answer": o_verbalized["answer"],
          "group": inst["group"],
          "confidence": o_verbalized["confidence"],
      }
      d_entropy = dict(d_verbalized)
      d_entropy["confidence"] = o_entropy["confidence"]
      new_ds_verbalized.append(d_verbalized)
      new_ds_entropy.append(d_entropy)
  
  datasets.Dataset.from_pandas(pd.DataFrame(
      new_ds_verbalized
  )).save_to_disk("in_context_ssl/reasoning/data/flores_bem_psl_k={}_verbalized.hf".format(k))
  datasets.Dataset.from_pandas(pd.DataFrame(
      new_ds_entropy
  )).save_to_disk("in_context_ssl/reasoning/data/flores_bem_psl_k={}_entropy.hf".format(k))


if __name__ == "__main__":
  # set your API here
  os.environ["OPENAI_API_KEY"] = ""
  client = OpenAI()
