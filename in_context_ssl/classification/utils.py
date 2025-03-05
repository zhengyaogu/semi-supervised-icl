from pydantic import BaseModel, Field
from collections import Counter

def query_openai(client, prompt, model, n=1, structured_output=False, confidence=False, logprobs=True):
    if model in ["gpt-4o-mini", "gpt-4o"]:
        if confidence:
            response_format = ClassificationResponseWithConfidence
        else:
            response_format = ClassificationResponse
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

class ClassificationResponse(BaseModel):
    label: str = Field(description="Label to the Input")

class ClassificationResponseWithConfidence(BaseModel):
    label: str = Field(description="Label to the Input")
    confidence: float = Field(description="The classification confidence of the predicted Label")


def extract_response_classification(choices, confidence):
    if confidence == "verbalized":  
        result = {
            "label": choices[0].message.parsed.label,
            "confidence": choices[0].message.parsed.confidence
        }
    elif confidence == "entropy":
        choice = choices[0]
        log_probs = [token.logprob for token in choice.logprobs.content]
        confidence = sum(log_probs) / len(log_probs)
        result = {
            "label": choice.message.parsed.label,
            "confidence": confidence
        }
    elif confidence == "self_consistency":
        preds = [choice.message.parsed.label for choice in choices]
        preds_d = Counter(preds)
        for p in preds_d:
            preds_d[p] = preds_d[p] / len(preds)
        majority = max(preds_d, key=lambda k: preds_d[k])
        confidence = preds_d[majority]

        result = {
            "label": majority,
            "confidence": confidence
        }
    
    return result