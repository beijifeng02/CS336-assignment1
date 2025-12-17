"""BPE Tokenizer Training Implementation."""

from __future__ import annotations

import os
import re
from collections import defaultdict
from multiprocessing import Pool
from typing import BinaryIO


# GPT-2 style pre-tokenization regex pattern
GPT2_PRETOKENIZE_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
    re.UNICODE
)


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


def pretokenize_chunk(args: tuple[str, int, int, list[str]]) -> dict[bytes, int]:
    """
    Pre-tokenize a chunk of the file and return word counts.
    
    Args:
        args: Tuple of (file_path, start, end, special_tokens)
    
    Returns:
        Dictionary mapping word bytes to their counts
    """
    file_path, start, end, special_tokens = args
    
    word_counts: dict[bytes, int] = defaultdict(int)
    
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    
    # Build regex pattern to split on special tokens
    if special_tokens:
        escaped_tokens = [re.escape(token) for token in special_tokens]
        split_pattern = re.compile("|".join(escaped_tokens))
        # Split text by special tokens, keeping the tokens
        parts = split_pattern.split(chunk)
    else:
        parts = [chunk]
    
    # Pre-tokenize each part using GPT-2 regex
    for part in parts:
        if not part:
            continue
        matches = GPT2_PRETOKENIZE_PATTERN.findall(part)
        for match in matches:
            word_bytes = match.encode("utf-8")
            word_counts[word_bytes] += 1
    
    return dict(word_counts)



class TokenNode:
    """Doubly linked list node for efficient BPE merging."""
    __slots__ = ['token', 'prev', 'next']
    
    def __init__(self, token: bytes):
        self.token = token
        self.prev: TokenNode | None = None
        self.next: TokenNode | None = None


class WordLinkedList:
    """Doubly linked list representing a word as a sequence of tokens."""
    __slots__ = ['head', 'tail', 'length']
    
    def __init__(self, word_bytes: bytes):
        """Initialize linked list from word bytes."""
        self.head: TokenNode | None = None
        self.tail: TokenNode | None = None
        self.length = 0
        
        # Initialize with single-byte tokens
        for byte in word_bytes:
            node = TokenNode(bytes([byte]))
            if self.head is None:
                self.head = node
                self.tail = node
            else:
                node.prev = self.tail
                self.tail.next = node
                self.tail = node
            self.length += 1
    
    def __iter__(self):
        """Iterate over all nodes in the list."""
        node = self.head
        while node is not None:
            yield node
            node = node.next


def build_initial_pair_counts(
    word_lists: dict[bytes, WordLinkedList],
    word_counts: dict[bytes, int]
) -> tuple[dict[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], set[tuple[bytes, TokenNode]]]]:
    """
    Build initial pair counts and location index.
    
    Returns:
        pair_counts: Dict mapping (token1, token2) -> total count
        pair_locations: Dict mapping (token1, token2) -> set of (word_bytes, first_node)
    """
    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_locations: dict[tuple[bytes, bytes], set[tuple[bytes, TokenNode]]] = defaultdict(set)
    
    for word_bytes, linked_list in word_lists.items():
        count = word_counts[word_bytes]
        node = linked_list.head
        while node is not None and node.next is not None:
            pair = (node.token, node.next.token)
            pair_counts[pair] += count
            pair_locations[pair].add((word_bytes, node))
            node = node.next
    
    return dict(pair_counts), pair_locations


def merge_pair_in_word(
    word_bytes: bytes,
    linked_list: WordLinkedList,
    word_count: int,
    pair: tuple[bytes, bytes],
    new_token: bytes,
    pair_counts: dict[tuple[bytes, bytes], int],
    pair_locations: dict[tuple[bytes, bytes], set[tuple[bytes, TokenNode]]]
) -> None:
    """
    Merge all occurrences of a pair in a word's linked list.
    Updates pair_counts and pair_locations incrementally.
    """
    node = linked_list.head
    
    while node is not None and node.next is not None:
        if node.token == pair[0] and node.next.token == pair[1]:
            next_node = node.next
            
            # Remove old pairs involving these nodes from counts
            # (prev, node.token) pair
            if node.prev is not None:
                old_left_pair = (node.prev.token, node.token)
                pair_counts[old_left_pair] -= word_count
                pair_locations[old_left_pair].discard((word_bytes, node.prev))
            
            # (next_node.token, next_node.next.token) pair
            if next_node.next is not None:
                old_right_pair = (next_node.token, next_node.next.token)
                pair_counts[old_right_pair] -= word_count
                pair_locations[old_right_pair].discard((word_bytes, next_node))
            
            # Merge: update node's token and remove next_node
            node.token = new_token
            node.next = next_node.next
            if next_node.next is not None:
                next_node.next.prev = node
            else:
                linked_list.tail = node
            linked_list.length -= 1
            
            # Add new pairs involving the merged token
            # (prev, new_token) pair
            if node.prev is not None:
                new_left_pair = (node.prev.token, new_token)
                pair_counts[new_left_pair] = pair_counts.get(new_left_pair, 0) + word_count
                pair_locations[new_left_pair].add((word_bytes, node.prev))
            
            # (new_token, next) pair
            if node.next is not None:
                new_right_pair = (new_token, node.next.token)
                pair_counts[new_right_pair] = pair_counts.get(new_right_pair, 0) + word_count
                pair_locations[new_right_pair].add((word_bytes, node))
            
            # Don't advance node - check if new token can merge again
        else:
            node = node.next



def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on the input corpus.
    
    Args:
        input_path: Path to the training data text file.
        vocab_size: Target vocabulary size (including special tokens).
        special_tokens: List of special tokens (e.g., ['<|endoftext|>']).
    
    Returns:
        vocab: Dictionary mapping token ID to token bytes.
        merges: List of merge operations in order of creation.
    """
    # Initialize vocab with 256 single-byte tokens
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: list[tuple[bytes, bytes]] = []
    
    # Calculate how many merges we need
    num_special = len(special_tokens)
    num_merges_needed = vocab_size - 256 - num_special
    
    if num_merges_needed <= 0:
        # No merges needed, just add special tokens
        for i, token in enumerate(special_tokens):
            vocab[256 + i] = token.encode("utf-8")
        return vocab, merges
    
    # Step 1: Multi-process pre-tokenization
    num_processes = os.cpu_count() or 1
    
    # Find the first special token to use as chunk boundary
    split_token = special_tokens[0] if special_tokens else None
    split_token_bytes = split_token.encode("utf-8") if split_token else b"\n"
    
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_token_bytes)
    
    # Prepare arguments for parallel processing
    chunk_args = [
        (str(input_path), start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    
    # Run pre-tokenization in parallel
    word_counts: dict[bytes, int] = defaultdict(int)
    
    if len(chunk_args) > 1:
        with Pool(processes=num_processes) as pool:
            results = pool.map(pretokenize_chunk, chunk_args)
        for result in results:
            for word, count in result.items():
                word_counts[word] += count
    else:
        # Single chunk, no need for multiprocessing
        result = pretokenize_chunk(chunk_args[0])
        word_counts.update(result)
    
    if not word_counts:
        # Empty corpus, return base vocab with special tokens
        for i, token in enumerate(special_tokens):
            vocab[256 + i] = token.encode("utf-8")
        return vocab, merges
    
    # Step 2: Build linked lists for each unique word
    word_lists: dict[bytes, WordLinkedList] = {
        word: WordLinkedList(word) for word in word_counts
    }
    
    # Step 3: Build initial pair counts and locations
    pair_counts, pair_locations = build_initial_pair_counts(word_lists, word_counts)
    
    # Step 4: Main BPE loop
    for _ in range(num_merges_needed):
        if not pair_counts:
            break
        
        # Find the best pair (max count, tie-break by lexicographically greater)
        best_pair = None
        best_count = 0
        
        for pair, count in pair_counts.items():
            if count <= 0:
                continue
            if count > best_count or (count == best_count and (best_pair is None or pair > best_pair)):
                best_count = count
                best_pair = pair
        
        if best_pair is None or best_count <= 0:
            break
        
        # Create new merged token
        new_token = best_pair[0] + best_pair[1]
        
        # Record the merge
        merges.append(best_pair)
        vocab[256 + len(merges) - 1] = new_token
        
        # Get all locations where this pair occurs (make a copy since we'll modify)
        locations = list(pair_locations.get(best_pair, set()))
        
        # Remove the merged pair from tracking
        del pair_counts[best_pair]
        if best_pair in pair_locations:
            del pair_locations[best_pair]
        
        # Update all words containing this pair
        affected_words = set(loc[0] for loc in locations)
        for word_bytes in affected_words:
            merge_pair_in_word(
                word_bytes,
                word_lists[word_bytes],
                word_counts[word_bytes],
                best_pair,
                new_token,
                pair_counts,
                pair_locations
            )
        
        # Clean up zero-count pairs
        zero_pairs = [p for p, c in pair_counts.items() if c <= 0]
        for p in zero_pairs:
            del pair_counts[p]
            if p in pair_locations:
                del pair_locations[p]
    
    # Step 5: Add special tokens at the end
    next_id = 256 + len(merges)
    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1
    
    return vocab, merges
