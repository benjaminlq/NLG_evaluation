import numpy as np

from typing import List, Dict
from .prompt import EVALUATION_PROMPT_TEMPLATE
from openai import OpenAI

def extract_token_probs(
    token_list: List
) -> Dict[str, float]:
    token_probs_dict = {}
    for token in token_list:
        token_str = token.token
        token_logprobs = token.logprob
        token_prob = np.exp(token_logprobs)
        token_probs_dict[token_str] = token_prob
    return token_probs_dict

def aggregate_token_scores(
    token_probs: Dict[str, float],
    scores_pool: List[int],
):
    prob_sum = 0
    score_sum = 0
    for score in scores_pool:
        if str(score) in token_probs:
            score_prob = token_probs[str(score)]
            prob_sum += score_prob
            score_sum += (score * score_prob)
        else:
            print(f"Warning: {score} is not present in the top tokens")

    return score_sum / prob_sum

def get_geval_score(
    client: OpenAI, criteria: str, steps: str, document: str, summary: str, metric_name: str,
    scores_pool: List[int] = [1, 2, 3, 4, 5], aggregate_tokens: bool = True
):
    prompt = EVALUATION_PROMPT_TEMPLATE.format(
        criteria=criteria,
        steps=steps,
        metric_name=metric_name,
        document=document,
        summary=summary,
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1,
        frequency_penalty=0,
        presence_penalty=0,
        logprobs=True,
        top_logprobs=20
    )

    if aggregate_tokens:
        token_probs = extract_token_probs(response.choices[0].logprobs.content[0].top_logprobs)
        normalized_score = aggregate_token_scores(token_probs, scores_pool)
    else:
        normalized_score = int(response.choices[0].message.content.strip())

    return normalized_score