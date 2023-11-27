# Model-based evaluation: Requires Ground-truth labels

# I. Embedding Similarity Score
Similarity Score between 2 answers. Can be **Cosine Similarity**, **Euclidean Distance** or **Dot Product**. 
### RAGAS
```
from ragas.metrics import AnswerSimilarity

answer_similarity = AnswerSimilarity()
dataset: Dataset({features: ['answer','ground_truths'], num_rows: 25})
results = answer_similarity.score(dataset)
```

### LlamaIndex
```
from llama_index.evaluation import SemanticSimilarityEvaluator
from llama_index import ServiceContext
from llama_index.embeddings import SimilarityMode, OpenAIEmbedding

service_context = ServiceContext.from_defaults(embed_model=OpenAIEmbedding())
evaluator = SemanticSimilarityEvaluator(
    service_context=service_context,
    similarity_mode=SimilarityMode.DEFAULT, # Cosine Similarity
    similarity_threshold=0.6,
)

result = evaluator.aevaluate(response=response, reference=reference)
print(f"Score: {result.score}, Passing: {result.passing}")
```


# II. BERT Score
**Reference**: [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/abs/1904.09675)

# III. Mover Score
**Reference**: [MoverScore: Text Generation Evaluating with Contextualized Embeddings and Earth Mover Distance](https://arxiv.org/abs/1909.02622)

# IV. BART Score
**Reference**: [BARTScore: Evaluating Generated Text as Text Generation](https://arxiv.org/abs/2106.11520)

# V. FactCC
**Reference**: [Evaluating the Factual Consistency of Abstractive Text Summarization](https://arxiv.org/abs/1910.12840)

# VI. QAGS
**Reference**: [Asking and Answering Questions to Evaluate the Factual Consistency of Summaries](https://arxiv.org/abs/2004.04228)

# VII. USR
**Reference**: [USR: An Unsupervised and Reference Free Evaluation Metric for Dialog Generation](https://arxiv.org/abs/2005.00456)

# VIII. UniEval
**Reference**: [Towards a Unified Multi-Dimensional Evaluator for Text Generation](https://arxiv.org/abs/2210.07197)

# IX. GPTScore
**Reference**: [GPTScore: Evaluate as You Desire](https://arxiv.org/abs/2302.04166)