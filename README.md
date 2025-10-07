# Advanced IR-RAG System

An advanced Information Retrieval and Retrieval-Augmented Generation (RAG) system designed for processing and analyzing financial documents, particularly SEC filings and company reports.

## Overview

This system provides sophisticated document processing capabilities with enhanced RAG techniques including:
- Advanced document chunking and preprocessing
- Multi-modal document parsing (PDF extraction with MinerU)
- Vector database storage with Milvus
- Query expansion and post-generation processing
- Financial data analysis and extraction

## Features

- **Advanced Document Processing**: Intelligent chunking, cleaning, and preprocessing of financial documents
- **Multi-Model Support**: Configurable LLM models for different tasks (extraction, analysis, cleaning)
- **Vector Search**: Milvus integration for efficient similarity search
- **Financial Analysis**: Specialized extraction of financial ratios, company information, and risk factors
- **Batch Processing**: Support for batch API processing with OpenAI
- **Async Processing**: Asynchronous document expansion and keyword generation

## Architecture

```
src/
├── advance_rag/          # Advanced RAG components
│   ├── adv_doc_processor.py     # Document expansion and processing
│   ├── adv_query_processor.py   # Query enhancement and processing
│   └── adv_post_gen_processor.py # Post-generation processing
├── database/             # Database handlers
│   └── milvus_handler.py        # Milvus vector database integration
├── models/              # Data models and schemas
│   └── parser_model.py          # Pydantic models for data validation
├── pdf_extractor/       # PDF processing
│   └── mineru_parser.py         # MinerU-based PDF extraction
├── pdf_processor/       # PDF data preparation
│   └── data_preparation_pdf.py  # PDF preprocessing utilities
├── prompts/            # Prompt templates
│   └── prompt.py               # System prompts for various tasks
└── utils/              # Utility functions
    ├── llm.py                  # LLM configuration and helpers
    └── utils.py                # General utilities
```

## Setup

### Prerequisites

- Python 3.8+
- Access to LiteLLM/OpenAI API
- Milvus database instance
- MySQL database (optional)

### Environment Variables

Create a `.env` file with the following variables:

```env
# LLM Configuration
LITE_LLM_KEY=your_litellm_api_key
EMBEDDING_END_POINT=your_embedding_endpoint

# Database Configuration
MILVUS_URL=your_milvus_url
MILVUS_DB_NAME=your_database_name
MILVUS_PW=your_milvus_password

# MySQL Configuration (optional)
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=your_database

# OpenAI API (for batch processing)
OPENAI_API_KEY=your_openai_api_key
```

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd adv-ir-rag

# Install dependencies
pip install uv
uv pip install -r requirements.txt  # Create requirements.txt based on imports
```

## Example
```bash
## parse the pdf documents (annual report as example)
python -m src.pdf_processor.data_preparation_pdf
```

## Configuration

The system uses a comprehensive configuration system in `src/config.py`:

- **Model Configurations**: Different models for various tasks (notes extraction, table analysis, content cleaning, etc.)
- **Chunking Parameters**: Configurable chunk sizes and overlap settings
- **Financial Data Extraction**: Predefined schemas for financial statements and ratios
- **Document Processing**: Page ranges and patterns for specific document types


## Specialized Features

### Financial Document Analysis

The system includes specialized configurations for analyzing financial documents:

- **SEC Filing Processing**: Predefined item structures for 10-K forms
- **Financial Ratio Extraction**: Automated extraction of key financial metrics
- **Company Information**: Structured extraction of company details, board composition, and risk factors

### Multi-Modal Processing

- Support for various document types (PDF, text)
- Table analysis and extraction
- Image and chart processing capabilities

## Model Configuration

The system supports multiple model configurations optimized for different tasks:

- **Notes Extraction**: DeepSeek R1 model for detailed analysis
- **Table Analysis**: Specialized model for tabular data processing
- **Content Cleaning**: GPT-4 Nano for content refinement
- **Batch Processing**: Cost-effective batch API processing

## Contributing

1. Follow the existing code structure and conventions
2. Add appropriate type hints and documentation
3. Test new features with sample financial documents
4. Update configuration files as needed

## License

[Add your license information here]

## Support

For questions or issues, please refer to the project documentation or open an issue in the repository.