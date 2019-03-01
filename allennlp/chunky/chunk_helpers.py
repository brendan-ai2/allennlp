import itertools

import numpy as np

def get_chunks(tags):
    chunks = []
    cur = []

    for tag in tags:
        cur.append(tag)

        if tag.startswith('L-') or tag.startswith('U-'):
            chunks.append(cur)
            cur = []

    return chunks


def make(size, chunk_type):
    if size == 0:
        return []
    elif size == 1:
        return ['U-' + chunk_type]
    else:
        return ['B-' + chunk_type] + ['I-' + chunk_type] * (size - 2) + ['L-' + chunk_type]

def grow(chunk):
    chunk_len = len(chunk)
    assert chunk_len > 0
    return make(chunk_len + 1, chunk[0][2:])

def shrink(chunk):
    chunk_len = len(chunk)
    assert chunk_len > 0
    return make(chunk_len - 1, chunk[0][2:])

def garble(tags):
    chunks = get_chunks(tags)
    if len(chunks) >= 2:
        index = np.random.randint(0, len(chunks))
        direction = np.random.randint(0, 2)
        garble_helper(chunks, index, direction)

    new_tags = list(itertools.chain.from_iterable(chunks))
    new_chunks = get_chunks(new_tags)
    index2 = np.random.randint(0, len(new_chunks))
    switch_tag(new_chunks, index2)

    return list(itertools.chain.from_iterable(new_chunks))

# TO INVESTIGATE VOCAB BUGS:
#In [1]: okay_tags = ["U-O", "B-NP", "L-NP", "I-NP", "U-NP", "U-PP", "U-VP", "B-VP", "L-VP", "I-VP", "U-ADVP", "U-SBAR", "U-ADJP", "U-PRT", "B-ADJP", "L-ADJP", "B-ADVP", "L-ADVP", "B-PP", "L-PP", "I-ADJP", "B-SBAR", "
#   ...: L-SBAR", "B-CONJP", "L-CONJP", "I-ADVP", "U-INTJ", "I-CONJP", "U-LST", "I-PP", "B-INTJ", "L-INTJ", "I-INTJ", "I-UCP", "B-UCP", "L-UCP", "I-SBAR", "U-CONJP", "B-PRT", "I-PRT", "L-PRT"]
#
#In [2]: type_to_bioul = {}
#
#In [3]: for tag in okay_tags:
#   ...:     t = tag[2:]
#   ...:     bioul = tag[0]
#   ...:     if t not in type_to_bioul:
#   ...:         type_to_bioul[t] = set()
#   ...:     type_to_bioul[t].add(bioul)
#   ...:

def garble_helper(chunks, index, direction):
    if index == 0:
        direction = 0
    elif index == len(chunks) - 1:
        direction = 1

    offset = 1 - direction * 2

    grow_type = chunks[index][0][2:]
    shrink_type = chunks[index + offset][0][2:]

    # Never grow the O chunk. B-O, etc. aren't in the vocab.
    # Same with LST.
    # UCP, on the other hand, doesn't have a U tag, so we can't always shrink it.
    if grow_type != 'O' and grow_type != 'LST' and (shrink_type != "UCP" or len(chunks[index + offset]) > 2):
        chunks[index] = grow(chunks[index])
        chunks[index + offset] = shrink(chunks[index + offset])

# Tags that can be all sizes.
SAFE_TAGS = ['NP', 'PP', 'VP', 'ADVP', 'SBAR', 'ADJP', 'PRT', 'CONJP', 'INTJ']
NUM = len(SAFE_TAGS)

def switch_tag(chunks, index):
    chunk = chunks[index]
    type = chunk[0][2:]
    new_type = type

    while new_type == type:
        new_type = SAFE_TAGS[np.random.randint(0, NUM)]

    chunks[index] = make(len(chunk), new_type)

