from pydantic import BaseModel, Field
from typing import List, Any, Callable
import re
from collections import defaultdict
from itertools import combinations
from in_context_ssl.grading.grader import grade_answer
from torchmetrics import CHRFScore
import torch
from openai import OpenAI
from torch.nn.functional import cosine_similarity
import math

class Step(BaseModel):
    step: str

class Output(BaseModel):
    rationale: str
    answer: str

class OutputWithConfidence(BaseModel):
    rationale: str
    answer: str
    confidence: float

def extract_answer_geometric_shape(answer):
    answer = answer[:3]

    if len(answer) == 1:
        return "({})".format(answer)
    else:
        return answer

def query_openai(client, prompt, model, n=1, structured_output=False, confidence=False, logprobs=True):
    if model in ["gpt-4o-mini", "gpt-4o"]:
        if confidence:
            response_format = OutputWithConfidence
        else:
            response_format = Output
    else:
        response_format = None
    
    if not structured_output:
        response_format = None
    
    if response_format is not None:
        completion = client.beta.chat.completions.parse(
            model= model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            n=n,              # Number of responses to generate
            # max_tokens=5000,     # Set a lower max_tokens value to limit response length and avoid timeout,
            response_format=response_format,
            logprobs=logprobs
        )
    else:
        completion = client.beta.chat.completions.parse(
            model= model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            n=n,              # Number of responses to generate
            # max_tokens=5000,     # Set a lower max_tokens value to limit response length and avoid timeout,
            logprobs=logprobs
        )
    return completion.choices


def parse_output_geometric_shape_structured(choice, confidence):
    answer = choice.message.parsed.answer
    rationale = choice.message.parsed.rationale
    out_d = {
        "answer": answer,
        "rationale": rationale
    }
    if confidence:
        out_d["confidence"] = choice.message.parsed.confidence
    return out_d

def parse_output_tracking7(text):
    answer_match = re.search(
        r"(\([A-G]\))",
        text
    )
    confidence_match = re.search(
        r"\*\*Confidence\*\*:\s+([+-]?(?:[0-9]*[.])?[0-9]+)",
        text
    )
    if answer_match:
        answer = answer_match.group(1)
    else:
        answer = None
    if confidence_match:
        confidence = float(confidence_match.group(1))
    else:
        confidence = 0.0
    return {
        "answer": answer,
        "rationale": text,
        "confidence": confidence
    }

parse_output_logical_deduction7 = parse_output_tracking7
parse_output_date = parse_output_tracking7


def parse_output_livebench_math(text):
    math_comp_match = re.search(
        r"[A-Z]{5}",
        text
    )
    amps_hard_match = re.search(
        r"\\boxed\{(.*)\}",
        text
    )
    olympiad_matches = re.findall(
        r"\d+(?:\s*,\s*\d+)*",
        text
    )
    confidence_match = re.search(
        r"\*\*Confidence\*\*:\s+([+-]?(?:[0-9]*[.])?[0-9]+)",
        text
    )

    if math_comp_match:
        answer = math_comp_match.group()[0]
    elif amps_hard_match:
        answer = amps_hard_match.group(1)
    elif olympiad_matches:
        answer = max(olympiad_matches, key=len)
    else:
        answer = None
    
    if confidence_match:
        confidence = float(confidence_match.group(1))
    else:
        confidence = 0.0
    return {
        "answer": answer,
        "rationale": text,
        "confidence": confidence
    }

def parse_output_gpqa(text):
    match = re.search(
        r"[0-9]{5}",
        text
    )
    alt_match = re.search(
        r"(?:\*\*)?Answer(?:\*\*)?:(?:\*\*)?\s+([0-9])",
        text
    )
    confidence_match = re.search(
        r"\*\*Confidence\*\*:\s+([+-]?(?:[0-9]*[.])?[0-9]+)",
        text
    )
    if match:
        answer = int(match.group()[0])
    elif alt_match:
        answer = int(alt_match.group(1))
    else:
        answer = None
    if confidence_match:
        confidence = float(confidence_match.group(1))
    else:
        confidence = 0.0
    return {
        "answer": answer,
        "rationale": text,
        "confidence": confidence
    }

def parse_output_geometric_shapes_unstructured(choice):
    text = choice.message.content

    answer_match = re.search(
        r"(?:\*\*)?Answer(?:\*\*)?:(?:\*\*)?\s+(\([A-Z]\)).*",
        text
    )
    #r"(?:\*\*)?Rationale(?:\*\*)?:\s*(.+?)(?=\n\s*\*\*Answer:\*\*|\Z)"
    rationale_match = re.search(
        r"(?:\*\*)?Rationale(?:\*\*)?:(?:\*\*)?\s*(.+?)\s*(?=(?:(?:\*\*)?Answer(?:\*\*)?:(?:\*\*)?)|$)",
        text,
        re.DOTALL
    )
    answer = None if not answer_match else answer_match.group(1).strip()
    rationale = None if not rationale_match else rationale_match.group(1).strip()
    
    return {
        "answer": answer,
        "rationale": rationale
    }

def parse_output_geometric_shapes_structured(choice, confidence):
    answer = choice.message.parsed.answer
    rationale = choice.message.parsed.rationale
    out_d = {
        "answer": answer,
        "rationale": rationale
    }
    if confidence == "verbalized":
        out_d["confidence"] = choice.message.parsed.confidence
    elif confidence == "entropy":
        log_probs = [token.logprob for token in choice.logprobs.content]
        confidence = sum(log_probs) / len(log_probs)
        out_d["confidence"] = confidence
    else:
        out_d["confidence"] = None
    return out_d

def extract_answer_geometric_shapes(answer):
    answer = answer[:3]

    if len(answer) == 1:
        return "({})".format(answer)
    else:
        return answer

def aggregate(choices: List[Any], parser: Callable, confidence: str, rationale: bool, correct_answer=None) -> dict[str, Any]:
    assert confidence in ["verbalized", "self_consistency", "entropy", "back_translation_chrf", "back_translation_lm"]
    if parser == parse_output_geometric_shapes_structured:
        parsed_outputs = [parser(choice, confidence="") for choice in choices]
        for inst in parsed_outputs:
            inst["answer"] = extract_answer_geometric_shapes(inst["answer"])
    else:
        parsed_outputs = [parser(choice.message.content) for choice in choices]

    if confidence == "verbalized":
        assert len(parsed_outputs) == 1, "verbalized confidence can only be used with one choice"
        if rationale:
            return {
                "answer": parsed_outputs[0]["answer"],
                "rationale": parsed_outputs[0]["rationale"],
                "confidence": parsed_outputs[0]["confidence"]
            }
        else:
            return {
                "answer": parsed_outputs[0]["answer"],
                "confidence": parsed_outputs[0]["confidence"]
            }
    elif confidence == "self_consistency":
        preds = [p["answer"] for p in parsed_outputs]
        rationales = [p["rationale"] for p in parsed_outputs]
        n = len(preds)
        
        def count_equivalence_classes(pred: List[Any]) -> dict[str, int]:
            equivalence_classes = defaultdict(list)

            for answer in pred:
                found = False
                for key in equivalence_classes:
                    if grade_answer(key, answer):
                        equivalence_classes[key].append(answer)
                        found = True
                        break
                if not found:
                    equivalence_classes[answer].append(answer)

            result = {key: len(equivalence_classes[key]) / n for key in equivalence_classes}
            return result
        
        counter = count_equivalence_classes(preds)
        majority = max(counter, key=counter.get)
        for p, r in zip(preds, rationales):
            if p == majority:
                rationale = r
                break
        confidence = counter[majority]
        if rationale:
            return {
                "answer": majority,
                "rationale": rationale,
                "confidence": confidence
            }
        else:
            return {
                "answer": majority,
                "confidence": confidence
            }
    elif confidence == "entropy":
        assert len(parsed_outputs) == 1, "entropy confidence can only be used with one choice"
        choice = choices[0]
        log_probs = [token.logprob for token in choice.logprobs.content]
        confidence = sum(log_probs) / len(log_probs)
        if rationale:
            return {
                "answer": parsed_outputs[0]["answer"],
                "rationale": parsed_outputs[0]["rationale"],
                "confidence": confidence
            }
        else:
            return {
                "answer": parsed_outputs[0]["answer"],
                "confidence": confidence
            }
    elif confidence.startswith("back_translation"):
        assert correct_answer is not None
        m = CHRFScore(return_sentence_level_score=True)
        client = OpenAI()

        def chrf_score(pred1, pred2):
            return (m(pred1, pred2)[1] ** math.exp(3)).item()
        
        @torch.no_grad()
        def lm_score(pred1, pred2):
            emb2 = client.embeddings.create(
                input=[pred2],
                model="text-embedding-3-large"
            ).data[0].embedding
            return torch.clamp(
                cosine_similarity(
                    torch.tensor(correct_answer_emb),
                    torch.Tensor(emb2),
                    dim=0
                ),
                min=0.
            ).item()

        if confidence.endswith("chrf"):
            scorer = chrf_score
        elif confidence.endswith("lm"):
            scorer = lm_score
            correct_answer_emb = client.embeddings.create(
                input=[correct_answer],
                model="text-embedding-3-large"
            ).data[0].embedding
        
        preds = [inst["answer"] for inst in parsed_outputs]
        scores = torch.zeros((len(preds), ))
        for i, pred in enumerate(preds):
            score = scorer(correct_answer, pred)
            scores[i] = score
        
        score = scores.mean(dim=0).item()
        
        if rationale:
            return {
                "answer": parsed_outputs[0]["answer"],
                "rationale": parsed_outputs[0]["rationale"],
                "confidence": score
            }
        else:
            return {
                "answer": parsed_outputs[0]["answer"],
                "confidence": score
            }
        
        
def parse_output_translation(target_lang, text):
    answer_match = re.search(
        fr'(?:{target_lang}:\s*)?([\"\']?)(.+)\1',
        text
    )
    confidence_match = re.search(
        r"\*\*Confidence\*\*:\s+([+-]?(?:[0-9]*[.])?[0-9]+)",
        text
    )
    answer = None if not answer_match else answer_match.group(2)
    confidence = None if not confidence_match else confidence_match.group(1)
    return {
        "answer": answer,
        "confidence": float(confidence) if confidence is not None else 0.
    }

    