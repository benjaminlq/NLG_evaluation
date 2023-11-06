# Synthetic Dataset Generation

## I. Evaluation Dataset Generator
### Prompt
```
from llama_index.prompts import PromptTemplate

question_generation_prompt = """\
Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge.
generate only questions based on the below query.
{query_str}
"""

DEFAULT_QUESTION_GENERATION_PROMPT = PromptTemplate(question_generation_prompt)

num_questions_per_chunk = 10
QUESTION_GENERATION_QUERY = (
    f"You are a Teacher/Professor. Your task is to setup 
    {num_questions_per_chunk} questions for an upcoming 
    quiz/examination. The questions should be diverse in nature 
    across the document. Restrict the questions to the 
    context information provided."
    )
```
### Usage

```
from llama_index.evaluation import DatasetGenerator
from llama_index import ServiceContext
from llama_index.llms import OpenAI

service_context = ServiceContext(
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=512),
    chunk_size = 512
)

data_gen = DatasetGenerator.from_documents(
    documents,
    service_context=service_context,
    num_questions_per_chunk=10,
    text_question_template=DEFAULT_QUESTION_GENERATION_PROMPT,
    text_qa_template=text_qa_template,
    question_gen_query=QUESTION_GENERATION_QUERY,
    required_keywords=None,
    exclude_keywords=None,
    show_progress=True
)
```
```
# To generate questions only:
eval_questions = data_gen.generate_questions_from_nodes(num=100)

# To generate questions and answers:
eval_dataset = data_gen.generate_dataset_from_nodes(num=100)

-> class QueryResponseDataset(BaseModel):
    queries: Dict[str, str] = Field(
        default_factory=dict, description="Query id -> query"
    )
    responses: Dict[str, str] = Field(
        default_factory=dict, description="Query id -> response"
    )

```

## 2. Embedding Dataset Generation

