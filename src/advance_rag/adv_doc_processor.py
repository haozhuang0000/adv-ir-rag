from langchain.prompts import PromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from typing import List
import asyncio

from src.models import ExpandKeywords, ExpandQA
from src.prompts import DocPrompt
from src.utils import llm

class AdvanceDocProcessor:

    def __init__(self):
        self.llm = llm

    async def document_expansion(self, chunks: List[str],
                           expand_headers: str=False,
                           expand_keywords: str=True,
                           expand_qa_session: str=True) -> List[str]:

        if expand_headers:
            #Todo: To be added
            raise NotImplementedError

        if expand_keywords:

            parser = JsonOutputParser(pydantic_object=ExpandKeywords)
            prompt = PromptTemplate(
            template=DocPrompt.KEYWORD_PROMPT,
            input_variables=["chunk_text"],
            partial_variables={"format_instructions": parser.get_format_instructions()})

            chain = prompt | self.llm | parser
            # keywords = []
            # for chunk in chunks:
            #     keyword = chain.invoke({'chunk_text': chunk})
            #     keywords.append(keyword)

            tasks = [chain.ainvoke({'chunk_text': chunk}) for chunk in chunks]
            keywords = await asyncio.gather(*tasks)


        if expand_qa_session:

            parser = JsonOutputParser(pydantic_object=ExpandQA)
            prompt = PromptTemplate(
            template=DocPrompt.QA_GENERATION_PROMPT,
            input_variables=["chunk_text"],
            partial_variables={"format_instructions": parser.get_format_instructions()})

            chain = prompt | self.llm | parser
            # qas = []
            # for chunk in chunks:
            #     qa = chain.invoke({'chunk_text': chunk})
            #     qas.append(qa)

            tasks = [chain.ainvoke({'chunk_text': chunk}) for chunk in chunks]
            qas = await asyncio.gather(*tasks)

        original_chunks = []
        modified_chunks = []
        for i in range(len(chunks)):
            original_chunk = chunks[i]
            if expand_keywords:
                modified_chunk = original_chunk + '\n\n keywords: \n' + keywords[i]['keywords']
            else:
                modified_chunk = original_chunk.copy()

            if expand_qa_session:
                qas_string = '\n'.join(qas[i]['qa_session'])
                modified_chunk = modified_chunk + '\n\n qa session: \n' + qas_string

            original_chunks.append(original_chunk)
            modified_chunks.append(modified_chunk)
            # t_dict = {
            #     'original_chunk': original_chunk, ## For retrieval purpose (store original chunks)
            #     'modified_chunk': modified_chunk, ## For embedding purpose
            #     'chunk_id': i
            # }
            # modified_chunks.append(t_dict)

        return original_chunks, modified_chunks

