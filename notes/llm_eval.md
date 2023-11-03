
# Main Evaluation Criteria for Retrieval Augmented Generation (RAG) Pipeline

`Retrieval`: Quality of the retrieval system

- Are the retrieved contexts relevant to the queries (Precision)?
- Does the retriever retrieve enough contexts for LLM to generate correct responses (Recall)? 

`Generation`: Quality of the generated response

- Does the response match the given context (or is LLM hallucinating)
- Does the response match the query (or is LLM hallucinating)
- Is the response correct or following certain output requirements (guidelines)<br><br>


# I. LlamaIndex LLM-based Evaluation

References: From [LLamaIndex Evaluation](https://gpt-index.readthedocs.io/en/stable/optimizing/evaluation/evaluation.html).
- [Faithfulness](https://gpt-index.readthedocs.io/en/stable/examples/evaluation/faithfulness_eval.html)
- [Relevancy](https://gpt-index.readthedocs.io/en/stable/examples/evaluation/relevancy_eval.html)
- []

## **Summary**
| Metric | | Query | Response | Context | Reference | 
| - | - | - | - | - | - |
| Faithfulness | | | | |
| Relevancy | | | | | |
| Correctness | | | | | |
| Guideline Adherence: | | | | | |
| Correctness | | | | | |

## 0. Base Evaluator Class:
To implement a custom Evaluator Class, create a subclass and overide **aevaluate/aevaluate_response** methods.
```
class BaseEvaluator(ABC):

    @abstractmethod
    async def aevaluate(
        self,
        query: Optional[str] = None,
        response: Optional[str] = None,
        contexts: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Run evaluation with query string, retrieved contexts,
        and generated response string.
        """
        raise NotImplementedError

    async def aevaluate_response(
        self,
        query: Optional[str] = None,
        response: Optional[Response] = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Run evaluation with query string and generated Response object.
        """
        if response is not None:
            response_str = response.response
            contexts = [node.get_content() for node in response.source_nodes]

        return await self.aevaluate(
            query=query, response=response_str, contexts=contexts, **kwargs
        )
```
```
class EvaluationResult(BaseModel):
    """Evaluation result.

    Output of an BaseEvaluator.
    """

    query: Optional[str] = Field(None, description="Query string")
    contexts: Optional[Sequence[str]] = Field(None, description="Context strings")
    response: Optional[str] = Field(None, description="Response string")
    passing: Optional[bool] = Field(None, description="Binary evaluation result (passing or not)")
    feedback: Optional[str] = Field(None, description="Feedback or reasoning for the response")
    score: Optional[float] = Field(None, description="Score for the response")
```
## 1. Relevancy Evaluator
## 2. Faithfulness Evaluator
## 3. Correctness
## 4. Guideline Adherence
## 5. Sentence Similarity
