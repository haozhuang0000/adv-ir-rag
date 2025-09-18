CONTENT_SEARCHING_PROMPT="""
You are document analyzer, your job is to analyze the document content to find out the respective page for each sections.

You will be giving a markdown

<EXAMPLE STARTS HERE>
# MISSION STATEMENT

Singapore Airlines is a global   
company dedicated to providing air   
transportation services of the highest   
quality and to maximising returns for   
the benefit of its shareholders and employees.

# CONTENTS

# OVERVIEW

2 Chairman’s   
Letter to Shareholders   
4 6 Significant EventsThe SIA Group Portfolio   
7 Three-year Financial Highlights   
8 Statistical Highlights   
10 Board of Directors   
15 SIA's Response to the SQ321 Incident   
16 Our Strategy for the Future

# GOVERNANCE

68 Statement on Risk Management   
71 Corporate Governance Report   
90 Membership and Attendance of Singapore Airlines Limited Board of Directors and Board Committee Members   
91 Further Information on Board of Directors
</EXAMPLE ENDS HERE>

<TASK STARTS HERE>
Your role is find out the content page and extract the content like this in JSON format:
"OVERVIEW": "start": "2", "end": "16", "sections": ["Chairman’s Letter to Shareholders", "Significant Events", "The SIA Group Portfolio", ..., "..."]
"GOVERNANCE": "start": "68", "end": "91", "sections": ["Statement on Risk Management", "Corporate Governance Report", "Membership and Attendance of Singapore Airlines Limited Board of Directors and Board Committee Members", ..., "..."]

- You MUST provide "start" and "end" page!!! You should only leave the empty "end" page for the last session!!
</TASK ENDS HERE>

<INPUT MARKDOWN STARTS HERE>
{input_markdown}
</INPUT MARKDOWN ENDS HERE>

and following this format:
{format_instructions}
"""

class DocPrompt:

    KEYWORD_PROMPT: str = """
    You are an expert keyword extractor, capable of identifying any type of keyword.
    Given a text, your task is to extract meaningful keywords.
    The extracted keywords should reflect domain-specific knowledge, such as finance, technology, or other specialized fields.
    
    <Text>
    {chunk_text}
    </Text>
    
    Please follow this format instruction:
    {format_instructions}
    """

    QA_GENERATION_PROMPT: str = """
    Analyze the input text and generate essential questions that, when answered, 
    capture the main points of the text. Each question should be one line without numbering or prefixes.
    You should generate no more than 8 questions. Each question must less than 30 tokens.
    
    <Text>:
    {chunk_text}
    </Text>
    
    Please follow this format instruction:
    {format_instructions}
    """

class QueryPrompt:


    QUERY_REWRITE_PROMPT = """
    You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. 
    Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.
    
    <Original query>
    {query}
    </Original query>
    
    Please follow this format instruction:
    {format_instructions}
    """

    QUERY_STEP_BACK_PROMPT = """
    You are an AI assistant tasked with generating broader, more general queries to improve context retrieval in a RAG system.
    Given the original query, generate a step-back query that is more general and can help retrieve relevant background information.
    
    <Original query>
    {query}
    </Original query>
    
    Please follow this format instruction:
    {format_instructions}
    """

    QUERY_MULTI_GEN_PROMPT = """
    You are an AI language model assistant. 
    
    Your task is to generate 3 different versions of the given user question to retrieve relevant documents from a vector database.
    
    By generating multiple perspectives on the user question, 
    your goal is to help the user overcome some of the limitations of distance-based similarity search. 
        
    <Original query>
    {query}
    </Original query>
    
    Please follow this format instruction:
    {format_instructions}
    """

    QUERY_SUBQUERY_PROMPT = """
    You are an AI assistant tasked with breaking down complex queries into simpler sub-queries for a RAG system.
    Given the original query, decompose it into 2-4 simpler sub-queries that, when answered together, would provide a comprehensive response to the original query.
    
    <Original query>
    {query}
    </Original query>
    
    <Example>
    Example: What are the impacts of interest rate hikes on the financial system?
    Sub-queries:
        1. How do interest rate hikes affect stock market valuations?
        2. What is the impact of higher interest rates on bond prices and yields?
        3. How do rising interest rates influence bank lending and credit availability?
        4. What are the effects of interest rate hikes on consumer spending and business investment?
    </Example>
    
    Please follow this format instruction:
    {format_instructions}
    """

class PostGenPrompt:

    pass