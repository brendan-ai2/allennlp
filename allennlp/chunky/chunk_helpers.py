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

chunks = ['U-O', 'U-NP', 'U-VP', 'B-NP', 'I-NP', 'L-NP', 'U-PP', 'B-NP', 'L-NP', 'U-ADVP', 'U-SBAR', 'B-NP', 'L-NP', 'U-NP', 'U-VP', 'U-ADJP', 'U-PP', 'U-O', 'U-O']

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

def garble(chunks):
    if len(chunks) < 2:
        return chunks

    index = np.random.randint(0, len(chunks))
    direction = np.random.randint(0, 2)
    return garble_helper(chunks, index, direction)

def garble_helper(chunks, index, direction):
    if index == 0:
        direction = 0
    elif index == len(chunks) - 1:
        direction = 1

    offset = 1 - direction * 2
    chunks[index] = grow(chunks[index])
    chunks[index + offset] = shrink(chunks[index + offset])
