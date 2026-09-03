"""Recursive character chunking (R14: chunk size 1,500, overlap 200)."""

from config import CHUNK_SIZE, CHUNK_OVERLAP

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _split_with_separators(text: str, chunk_size: int, overlap: int) -> list[str]:
    final_chunks: list[str] = []

    def _merge_or_append(fragment: str, chunks: list[str]):
        if not fragment:
            return
        if chunks and len(chunks[-1]) + len(fragment) <= chunk_size:
            chunks[-1] = chunks[-1] + fragment
        else:
            chunks.append(fragment)

    def _cruncher(text: str, separators: list[str]):
        if len(text) <= chunk_size or not separators:
            return [text]
        sep = separators[0]
        final = []
        sep_chunks = text.split(sep)
        current = ""
        for piece in sep_chunks:
            candidate = current + sep + piece if current else piece
            if len(candidate) > chunk_size and current:
                _merge_or_append(current, final)
                current = piece
            else:
                current = candidate
            if len(current) > chunk_size:
                # recurse within this oversized piece using the next separator level
                for sub in _cruncher(current, separators[1:]):
                    _merge_or_append(sub, final)
                current = ""
        if current:
            _merge_or_append(current, final)
        return final

    chunks = _cruncher(text, _SEPARATORS)
    # apply overlap by re-joining chunks' tails
    final_chunks = []
    for i, c in enumerate(chunks):
        if c:
            final_chunks.append(c)
    final_chunks = _apply_overlap(final_chunks, chunk_size, overlap)
    return final_chunks


def _apply_overlap(chunks: list[str], chunk_size: int, overlap: int) -> list[str]:
    if len(chunks) <= 1 or overlap <= 0:
        return chunks
    out = [chunks[0]]
    for i in range(1, len(chunks)):
        tail = out[-1][-(overlap):] if len(out[-1]) >= overlap else out[-1]
        out.append(tail + chunks[i])
    return out


def chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    return _split_with_separators(text, chunk_size, overlap)
