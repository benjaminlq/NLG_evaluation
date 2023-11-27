# Metric-based evaluation: Requires Ground-truth labels

References:

1. [Zuzanna Deutschman - Recommender Systems: Machine Learning Metrics and Business Metrics](https://neptune.ai/blog/recommender-systems-metrics#:~:text=Average%20precision%20(AP)&text=Precision%405%20equals%20%E2%85%95%20because,top%20ranking%20the%20correct%20recommendations)
2. [Benjamin Wang - Ranking Evaluation Metrics for Recommender Systems](https://towardsdatascience.com/ranking-evaluation-metrics-for-recommender-systems-263d0a66ef54)

<br>

# I. End-to-end Metrics
## Natural Language Understanding (NLU)
### 1. Exact Match (EM)
### 2. F1
### 3. Precision
### 4. Recall
<br>

## Natural Language Generation (NLG)
### 1. BLEU

### 2. ROUGE

### 3. METEOR
<br><br>

# II. Retrieval Metrics
## 1. Hit Rate (HR)

Measure the number of queries in which the correct contexts are included inside the retrieved contexts

$$ Hit\ rate = {{Queries\ containing\ correct\ contexts} \over {Total\ number\ of\ queries}} $$

## 2. Mean Average Precision (MAP)
Mean Average Precision measures the relevancy of the retrieved docuements with respect to a query question.
- `Precision`: Fraction of relevant documents in all retrieved items
- `Racall`: Fraction of all relevant docuemnts which have been retrieved by the retrieval system.
- `Precision@k`: Fraction of relevant items in the top k retrieved items.

$$Precision@k = \frac{\sum_{i=1}^kRel(i)}{k}$$

where Rel(i) = 1 if i is a relevant document, 0 if i is an irrelevant document.

- `Average Precision`: Area under the Precision-Recall curve.

$$Average Precision AP@N = \sum_{i=1}^{N}Precision@i\ * \ Rel(i)$$

where N is the number of number of documents to be retrieved, 

- `Mean Averrag Precision (MAP)`: Mean Average Precision (AP) across all queries
  
$$MAP = \frac{1}{Q} \sum_{i=1}^{Q}AP(q_i) $$

where Q is the number of queries and q_i is the ith query.

## 3. Mean Reciprocal Rank (MRR)
- `Reciprocal Rank (RR)`: Sum of the relevance score of the top items weighted by reciprocal rank.
  - For implicit dataset, the relevance score is quantified by binary score: Relevant (1), Irrelevant (0)
  - $$ RR@k = \sum_{i=1}^{k}\frac{relevance_i}{i} $$
    where k is the total number of retrieved items

- `Mean Reciprocal Rank (MRR)`: Mean of reciprocal ranks across all queries.

$$MRR = \frac{1}{Q} \sum_{i=1}^{Q}RR(q_i) $$

where Q is the number of queries and q_i is the ith query.

## 4. Normalized Discounted Cumulative Gain (NDCR)
- `Discounted Cumulative Gain (DCG)`: Cumulative Gain (Relevancy Score) weighted by the position inside the recommendation list.
  $$ DCG@k = \sum_{i=1}^{k}\frac{Rel(i)}{log_{2}(i+1)} $$
- `Ideal Discounted Cumulative Gain (IDCF)`: DCG for the list whereas the recommendation list is ideal.
  $$ IDCG@k = \sum_{i=1}^{I(k)}\frac{Rel(i)}{log_{2}(i+1)} $$
- `Normalized Discounted Cumulative Gain (NDCG):`
  $$ NDCG@k = \frac{DCG@k}{IDCG@k} $$

## 5. LLamaIndex Evaluation



  