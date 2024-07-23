from typing import Union, Callable, List
import tiktoken
from llama_index.core.schema import TextNode, NodeWithScore

def count_tokens(
    texts: Union[str, TextNode, NodeWithScore, List],
    tokenizer: Callable = tiktoken.encoding_for_model("gpt-3.5-turbo")
) -> int:
    """Count total number of tokens of documents

    Args:
        texts (Union[str, TextNode, NodeWithScore ,List]): Documents or List of Documents
        tokenizer (Callable, optional): Tokenizer encoding function. Defaults to tiktoken.encoding_for_model("gpt-3.5-turbo").

    Returns:
        int: Number of tokens
    """
    token_counter = 0
    if not isinstance(texts, List):
        texts = [texts]
    for text in texts:
        if isinstance(text, NodeWithScore):
            text_str = text.node.text
        elif isinstance(text, TextNode):
            text_str = text.text
        elif isinstance(text, str):
            text_str = text
        else:
            ValueError("Invalid input texts") 
        token_counter += len(tokenizer.encode(text_str))
    return token_counter