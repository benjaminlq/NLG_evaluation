from .prompt import GPTSCORE_GENERAL_PROMPT
from openai import OpenAI
from openai.types import Completion

def calculate_logprobs(
    response: Completion, prompt: str, generated_summary: str
) -> float:
    out = response.choices[0]

    i = out.logprobs.text_offset.index(len(prompt) - len(generated_summary))
    print('eval text', out.logprobs.tokens[i: -1])

    score = sum(out.logprobs.token_logprobs[i:-1])
    return score

def get_gptscore(
    client: OpenAI, instruction: str, document: str, summary: str, 
) -> float:
    prompt = GPTSCORE_GENERAL_PROMPT.format(
        instruction=instruction,
        document=document,
        summary=summary
    )

    response = client.completions.create(
        model="davinci-002",
        prompt=prompt,
        temperature=0,
        max_tokens=0,
        logprobs=0,
        echo=True,
        n=None,
        stop="\n"
    )

    gpt_score = calculate_logprobs(response, prompt, summary)

    return gpt_score