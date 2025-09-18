from pydantic import BaseModel, Field
from typing import List, Dict
# class ContentPageOutput(BaseModel):
#     competitor_name: List[str]
#     summary: List[str]

class ExpandKeywords(BaseModel):

    keywords: str = Field(..., description="Keywords extracted from chunk text")

class ExpandQA(BaseModel):

    qa_session: List[str] = Field(..., description="QA session base on chunk text")

class RewriteQuery(BaseModel):

    prompt: str = Field(..., description="Rewritten Prompt")

class StepBackQuery(BaseModel):

    prompt: str = Field(..., description="Prompt with broader knowledges and more general")

class MultiQuery(BaseModel):

    prompt: List[str] = Field(..., description="Generate multiple version of the queries")

class SubQuery(BaseModel):

    prompt: List[str] = Field(..., description="Convert Complex query into sub-queries")

