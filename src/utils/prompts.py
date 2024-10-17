"""
Define all prompts used in this project.
"""

PLACEHOLDER = "."  # The effect of placeholder

INSTRUCTION = (
    "Based on the following triplets from a knowledge graph, please answer the given question. "
    "Please keep the answers as simple as possible and return all the possible answers as a list.\n\n"
)

RETRIEVAL_CONTEXT = """Triplets:\n"""  # + triplet token-ids

QUESTION = """\n\nQuestion:\n{question}"""

BOS = '<s>'
BOS_INST = '[INST]'
EOS_INST = '[/INST]'
EOS = '</s>'

FORMER = BOS + BOS_INST + INSTRUCTION + RETRIEVAL_CONTEXT
# masked triplet
LATTER = QUESTION + EOS_INST
LABEL = """{label}""" + EOS


# v1 prompt:
# bos + bos_inst + inst + triplet + query + eos_inst + labels + eos.
