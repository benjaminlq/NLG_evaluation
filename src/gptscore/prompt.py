GPTSCORE_GENERAL_PROMPT = """\
Instruction:
{instruction}

Source Text:
{document}

Summary:
{summary}\
    """
    
SUMMARIZE_FACTUAL_CONSISTENCY_INST = "Generate a relevant summary with consistent facts for the following text:"
REWRITE_FACTUAL_CONSISTENCY_INST = "Rewrite the following text with relevant consistent facts:"
SUMMARIZE_COVERAGE_INST = "Generate a summary with as much semantic coverage as possible for the following text:"
REWRITE_COVERAGE_INST = "Rewrite the following text with the same semantics:"
SUMMARIZE_COHERENCE_INST = "Generate a coherent summary for the following text:"
REWRITE_COHERENCE_INST = "Rewrite the following text into a coherent text:"
SUMMARIZE_FLUENCY_INST = "Generate a fluent and grammatical summary for the following text:"
REWRITE_FLUENCY_INST = "Rewrite the following text into a fluent and grammatical text:"

