from langchain.prompts import PromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from typing import List, Literal
import asyncio

from src.models import RewriteQuery, StepBackQuery, MultiQuery, SubQuery
from src.prompts import QueryPrompt
from src.utils import llm

class AdvanceQueryProcessor:

    def __init__(self):
        self.llm = llm

    def query_transformation(
            self,
            query: str,
            transformation: Literal["rewrite", "stepback", "multiquery", "subquery", None],
    ):
        """
        Apply a specific query transformation.

        :param query: The original query string.
        :param transformation: The type of transformation to apply. Must be one of:
            - "rewrite": Rephrase the query for better clarity.
            - "stepback": Generalize the query to broader knowledge.
            - "multiquery": Generate multiple alternative query variations.
            - "subquery": Break a complex query into smaller sub-queries.
        :return: A corresponding Pydantic model containing the transformed query.
        """
        if transformation == None:
            return query

        elif transformation == "rewrite":

            parser = JsonOutputParser(pydantic_object=RewriteQuery)
            prompt = PromptTemplate(
            template=QueryPrompt.QUERY_REWRITE_PROMPT,
            input_variables=["query"],
            partial_variables={"format_instructions": parser.get_format_instructions()})

            chain = prompt | self.llm | parser
            modified_prompt = chain.invoke({'query': query})
            return modified_prompt

        elif transformation == "stepback":

            parser = JsonOutputParser(pydantic_object=StepBackQuery)
            prompt = PromptTemplate(
                template=QueryPrompt.QUERY_STEP_BACK_PROMPT,
                input_variables=["query"],
                partial_variables={"format_instructions": parser.get_format_instructions()})

            chain = prompt | self.llm | parser
            modified_prompt = chain.invoke({'query': query})
            return modified_prompt

        elif transformation == "multiquery":

            parser = JsonOutputParser(pydantic_object=MultiQuery)
            prompt = PromptTemplate(
                template=QueryPrompt.QUERY_MULTI_GEN_PROMPT,
                input_variables=["query"],
                partial_variables={"format_instructions": parser.get_format_instructions()})

            chain = prompt | self.llm | parser
            modified_prompt = chain.invoke({'query': query})
            return modified_prompt

        elif transformation == "subquery":
            parser = JsonOutputParser(pydantic_object=SubQuery)
            prompt = PromptTemplate(
                template=QueryPrompt.QUERY_SUBQUERY_PROMPT,
                input_variables=["query"],
                partial_variables={"format_instructions": parser.get_format_instructions()})

            chain = prompt | self.llm | parser
            modified_prompt = chain.invoke({'query': query})
            return modified_prompt
