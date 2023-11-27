# Synthetic Dataset Generation

# LlamaIndex

References:
- [LlamaIndex Questions Generation](https://docs.llamaindex.ai/en/stable/examples/evaluation/QuestionGeneration.html)
- [RAGAS Dataset Generation](https://docs.ragas.io/en/latest/concepts/testset_generation.html)

## 1. Evaluation Dataset Generator
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

### Prompt
```
DEFAULT_QA_GENERATE_PROMPT_TMPL = """\
Context information is below.

---------------------
{context_str}
---------------------

Given the context information and not prior knowledge.
generate only questions based on the below query.

You are a Teacher/ Professor. Your task is to setup \
{num_questions_per_chunk} questions for an upcoming \
quiz/examination. The questions should be diverse in nature \
across the document. Restrict the questions to the \
context information provided."
"""
```

### Usage
```
from llama_index.node_parser import SimpleNodeParser

node_parser = SimpleNodeParser.from_defaults(chunk_size = 512)
documents = node_parser.get_nodes_from_documents(documents)

from llama_index.evaluation import (
    generate_question_context_pairs,
    EmbeddingQAFinetuneDataset,
)

qa_dataset = generate_question_context_pairs(
    nodes, llm=OpenAI("gpt-3.5-turbo", temperature=0),
    num_questions_per_chunk=2
)
```

Output is an EmbeddingQAFinetuneDataset object:

```
class EmbeddingQAFinetuneDataset(BaseModel):

    queries: Dict[str, str]  # question id -> query
    corpus: Dict[str, str]  # doc id -> doc content
    relevant_docs: Dict[str, List[str]]  # question id -> list of doc ids
```

# RAGAS

**RAGAS** allows question evolution to generate harder questions.

- **Reasoning**: Rewrite the question in a way that enhances the need for reasoning to answer it effectively.

- **Conditioning**: Modify the question to introduce a conditional element, which adds complexity to the question.

- **Multi-Context**: Rephrase the question in a manner that necessitates information from multiple related sections or chunks to formulate an answer.

## Usage
```
from ragas.testset import TestsetGenerator
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from ragas.llms import LangchainLLM

generator_llm = LangchainLLM(llm=ChatOpenAI(model="gpt-3.5-turbo"))
critic_llm = LangchainLLM(llm=ChatOpenAI(model="gpt-3.5-turbo"))
embeddings_model = OpenAIEmbeddings()

testset_distribution = { "simple": 0.25, "reasoning": 0.5, "multi_context": 0.0, "conditional": 0.25,}

# percentage of conversational question

test_generator = TestsetGenerator(
    generator_llm=generator_llm,
    critic_llm=critic_llm,
    embeddings_model=embeddings_model,
    testset_distribution=testset_distribution,
    chunk_size=512
)

testset = test_generator.generate(documents, test_size=20)
```

Output is TestDataset object containing a list of DataRows

```
DataRow = namedtuple("DataRow", ["question", "context", "answer", "question_type"])

class TestDataset:
    test_data: t.List[DataRow]
```

## Prompts
### Seed Questions
```
SEED_QUESTION = HumanMessagePromptTemplate.from_template(
    """\
Your task is to formulate a question from given context satisfying the rules given below:
    1.The question should make sense to humans even when read without the given context.
    2.The question should be fully answered from the given context.
    3.The question should be framed from a part of context that contains important information. It can also be from tables,code,etc.
    4.The answer to the question should not contain any links.
    5.The question should be of moderate difficulty.
    6.The question must be reasonable and must be understood and responded by humans.
    7.Do no use phrases like 'provided context',etc in the question
    8.Avoid framing question using word "and" that can be decomposed into more than one question.
    9.The question should not contain more than 10 words, make of use of abbreviation wherever possible.
    
context:{context}
"""
```

### Reasoning Questions
```
REASONING_QUESTION = HumanMessagePromptTemplate.from_template(
    """\
You are a prompt rewriter. You will be provided with a question and a long context.Your task to is to complicate the given question to improve the difficulty of answering. 
You should do complicate the question by rewriting question into a multi-hop reasoning question based on the provided context. The question should require the reader to make multiple logical connections or inferences using the information available in given context. 
Here are some strategies to create multi-hop questions:

   - Bridge related entities: Identify information that relates specific entities and frame question that can be answered only by analysing information of both entities.
   
   - Use Pronouns: identify (he, she, it, they) that refer to same entity or concepts in the context, and ask questions that would require the reader to figure out what pronouns refer to.

   - Refer to Specific Details: Mention specific details or facts from different parts of the context including tables, code, etc and ask how they are related.

   - Pose Hypothetical Scenarios: Present a hypothetical situation or scenario that requires combining different elements from the context to arrive at an answer.

Rules to follow when rewriting question:
1. Ensure that the rewritten question can be answered entirely from the information present in the contexts.
2. Do not frame questions that contains more than 15 words. Use abbreviation wherever possible.
3. Make sure the question is clear and unambiguous. 
4. phrases like 'based on the provided context','according to the context',etc are not allowed to appear in the question.

question: {question}
CONTEXTS:
{context}

Multi-hop Reasoning Question:
""")
```

### Multi Context Questions
```
MULTICONTEXT_QUESTION = HumanMessagePromptTemplate.from_template(
    """\
You are a prompt rewriter. You will be provided with a question and two set of contexts namely context1 and context2. 
Your task is to complicate the given question in a way that answering it requires information derived from both context1 and context2. 
Follow the rules given below while rewriting the question.
    1. The rewritten question should not be very long. Use abbreviation wherever possible.
    2. The rewritten question must be reasonable and must be understood and responded by humans.
    3. The rewritten question must be fully answerable from information present in context1 and context2. 
    4. Read and understand both contexts and rewrite the question so that answering requires insight from both context1 and context2.
    5. phrases like 'based on the provided context','according to the context?',etc are not allowed to appear in the question.

question:\n{question}
context1:\n{context1}
context2:\n{context2}
""" )
```

### Conditional Questions
```
CONDITIONAL_QUESTION = HumanMessagePromptTemplate.from_template(
    """\
Rewrite the provided question to increase its complexity by introducing a conditional element.
The goal is to make the question more intricate by incorporating a scenario or condition that affects the context of the question.
Follow the rules given below while rewriting the question.
    1. The rewritten question should not be longer than 25 words. Use abbreviation wherever possible.
    2. The rewritten question must be reasonable and must be understood and responded by humans.
    3. The rewritten question must be fully answerable from information present context.
    4. phrases like 'provided context','according to the context?',etc are not allowed to appear in the question.
for example,
question: What are the general principles for designing prompts in LLMs?
Rewritten Question:how to apply prompt designing principles to improve LLMs performance in reasoning tasks

question:{question}
context:\n{context}
Rewritten Question
""")
```

### Conversation Questions:
```
CONVERSATION_QUESTION = HumanMessagePromptTemplate.from_template(
    """\
Reformat the provided question into two separate questions as if it were to be part of a conversation. Each question should focus on a specific aspect or subtopic related to the original question.
question: What are the advantages and disadvantages of remote work?
Reformatted Questions for Conversation: What are the benefits of remote work?\nOn the flip side, what challenges are encountered when working remotely?
question:{question}

Reformatted Questions for Conversation:
""")
```