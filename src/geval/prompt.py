EVALUATION_PROMPT_TEMPLATE = """
You will be given one summary written for an article. Your task is to rate the summary on one metric.
Please make sure you read and understand these instructions very carefully.
Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:

{criteria}

Evaluation Steps:

{steps}

Source Text:

{document}

Summary:

{summary}

Evaluation Form (ONLY output a single INTEGER score):

- {metric_name}
"""

RELEVANCY_SCORE = {
    "criteria": (
        "Relevance(1-5) - selection of important content from the source. "
        "The summary should include only important information from the source document. "
        "Annotators were instructed to penalize summaries which contained redundancies and excess information."
        ),
    "steps": (
        "1. Read the summary and the source document carefully. \n"
        "2. Compare the summary to the source document and identify the main points of the article. \n"
        "3. Assess how well the summary covers the main points of the article, and how much irrelevant or redundant information it contains. \n"
        "4. Assign a relevance score from 1 to 5. \n"
        ),
    "score_pool": [1, 2, 3, 4, 5]
    }

COHERENCE_SCORE = {
    "criteria": (
        "Coherence(1-5) - the collective quality of all sentences. "
        "We align this dimension with the DUC quality question of structure and coherence "
        "The summary should not just be a heap of related information, but should build from sentence to a "
        "coherent body of information about a topic."
        ),
    "steps": (
        "1. Read the article carefully and identify the main topic and key points."
        "2. Read the summary and compare it to the article. Check if the summary covers the main topic and key points of the article, "
        "and if it presents them in a clear and logical order.\n"
        "3. Assign a score for coherence on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria."
        ),
    "score_pool": [1, 2, 3, 4, 5]
}

CONSISTENCY_SCORE = {
    "criteria": (
        "Consistency(1-5) - the factual alignment between the summary and the summarized source. "
        "A factually consistent summary contains only statements that are entailed by the source document. "
        "Annotators were also asked to penalize summaries that contained hallucinated facts."
        ),
    "steps": (
        "1. Read the article carefully and identify the main facts and details it presents.\n"
        "2. Read the summary and compare it to the article. Check if the summary contains any factual errors that are not supported by the article.\n"
        "3. Assign a score for consistency based on the Evaluation Criteria."
        ),
    "score_pool": [1, 2, 3, 4, 5]
}

FLUENCY_SCORE = {
    "criteria": (
        "Fluency(1-3): the quality of the summary in terms of grammar, spelling, punctuation, word choice, and sentence structure.\n"
        "1: Poor. The summary has many errors that make it hard to understand or sound unnatural.\n"
        "2: Fair. The summary has some errors that affect the clarity or smoothness of the text, but the main points are still comprehensible.\n"
        "3: Good. The summary has few or no errors and is easy to read and follow."
        ),
    "steps": "Read the summary and evaluate its fluency based on the given criteria. Assign a fluency score from 1 to 3.",
    "score_pool": [1, 2, 3],
}