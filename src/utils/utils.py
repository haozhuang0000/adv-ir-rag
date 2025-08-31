"""
This script is written by my colleague: Liu Chang @ AIDF-NUS
"""

from typing import List, Dict, Optional
import re
import httpx
from src.config import EMB_URL
import traceback
from langchain_text_splitters import NLTKTextSplitter

class Utils:

    @staticmethod
    def find_unique_entity(list_of_entity: List[str], list_of_type: List[str], entity_ids: List[int]) -> tuple[
        List[str], List[int]]:
        unique_entity: Dict[str, int] = {}
        pattern = re.compile(r"locationC\w*")
        # For Chinese
        # pattern = re.compile(r'\(公司对象\d+\)')
        for i in range(len(list_of_entity)):
            entity: str = list_of_entity[i]
            type: str = list_of_type[i]
            entity_id = entity_ids[i]
            if len(pattern.findall(type)) > 0 and len(entity) > 0:
                unique_entity[entity] = entity_id

        unique_entity_list = list(unique_entity.keys())
        unique_entity_id_list = list(unique_entity.values())
        return unique_entity_list, unique_entity_id_list

    @staticmethod
    async def a_embed_documents(texts: List[str]) -> Optional[List[List[float]]]:
        data = {'input': texts, 'type': 'documents'}
        timeout = httpx.Timeout(600, connect=10)
        async with httpx.AsyncClient(verify=False, timeout=timeout) as client:
            try:
                print("Sending request")
                response = await client.post(EMB_URL, json=data)
                print("Received response")
                return response.json()
            except Exception as e:
                print(str(e))
                traceback.print_exc()
                return None

    @staticmethod
    async def a_embed_query(texts: str) -> Optional[List[List[float]]]:
        data = {'input': texts, 'type': 'query'}
        timeout = httpx.Timeout(600, connect=10)
        async with httpx.AsyncClient(verify=False, timeout=timeout) as client:
            try:
                print("Sending request")
                response = await client.post(EMB_URL, json=data)
                print("Received response")
                return response.json()["vector"]
            except Exception as e:
                print(str(e))
                traceback.print_exc()
                return None

    @staticmethod
    def split(text_to_split) -> List[str]:
        text_splitter = NLTKTextSplitter(
            chunk_size=1500,
            chunk_overlap=500,
        )
        chunks = text_splitter.split_text(text_to_split)

        # Safety check: ensure no chunk exceeds Milvus limit (2000 chars)
        max_chunk_length = 1600  # Safety margin below 2000
        safe_chunks = []
        rechunked_count = 0

        for chunk in chunks:
            if len(chunk) > max_chunk_length:
                # Re-chunk the oversized chunk instead of truncating
                rechunked_pieces = Utils._rechunk_oversized(chunk, max_chunk_length)
                safe_chunks.extend(rechunked_pieces)
                rechunked_count += 1
            else:
                safe_chunks.append(chunk)

        if rechunked_count > 0:
            original_count = len(chunks)
            final_count = len(safe_chunks)
            print(f"🔄 Re-chunked {rechunked_count} oversized chunks: {original_count} → {final_count} total chunks")

        return safe_chunks

    @staticmethod
    def _rechunk_oversized(text: str, max_length: int) -> List[str]:
        """Re-chunk oversized text into smaller pieces, preserving content"""

        if len(text) <= max_length:
            return [text]

        # Strategy 1: Try splitting at sentence boundaries
        sentences = text.split('.')
        if len(sentences) > 1:
            current_chunk = ""
            chunks = []

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                # Add period back except for last sentence
                sentence_with_period = sentence + "." if sentence != sentences[-1].strip() else sentence

                # Check if adding this sentence would exceed limit
                if len(current_chunk) + len(sentence_with_period) + 1 <= max_length:
                    current_chunk += (" " + sentence_with_period if current_chunk else sentence_with_period)
                else:
                    # Save current chunk and start new one
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sentence_with_period

            # Add the last chunk
            if current_chunk:
                chunks.append(current_chunk)

            # Check if all chunks are now within limit
            all_fit = all(len(chunk) <= max_length for chunk in chunks)
            if all_fit:
                return chunks

        # Strategy 2: If sentence splitting didn't work, split by character count
        chunks = []
        start = 0
        overlap = 100  # Small overlap to maintain context

        while start < len(text):
            end = start + max_length

            if end >= len(text):
                # Last chunk
                chunks.append(text[start:])
                break

            # Try to find a good break point (space, punctuation)
            chunk_text = text[start:end]

            # Look for space in the last 10% of the chunk
            last_space = chunk_text.rfind(' ')
            if last_space > max_length * 0.9:
                end = start + last_space

            chunks.append(text[start:end])
            start = end - overlap  # Move with overlap to maintain context

        return chunks