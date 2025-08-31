import os
from dotenv import load_dotenv
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

# LiteLLM Configuration
LITE_LLM_KEY = os.getenv("LITE_LLM_KEY", None)
LITE_LLM_BASE_URL = "https://lightllm.nuscri.org:8443"
EMB_URL = os.getenv("EMBEDDING_END_POINT", None)
MILVUS_URL = os.getenv("MILVUS_URL", None)
MILVUS_DB_NAME = os.getenv("MILVUS_DB_NAME", None)
MILVUS_PW = os.getenv("MILVUS_PW", None)

if LITE_LLM_KEY is None:
    raise ValueError("LITE_LLM_KEY is not set")

CHUNK_CONFIG = {
      "CHUNK_MAX_TOK": 1000,    # Maximum tokens per chunk
      "OVERLAP": 200,           # Overlap tokens between chunks
      "BATCH_SIZE": 10          # Number of chunks per batch
  }

# Model configurations for different tasks
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "notes_extraction": {
        "class": ChatOpenAI,
        "kwargs": {
            "temperature": 0,
            "max_tokens": None,
            "timeout": None,
            "max_retries": 2,
            "model": "ollama/deepseek-r1:8b",
            "api_key": LITE_LLM_KEY,
            "base_url": LITE_LLM_BASE_URL,
        }
    },
    "table_analysis": {
        "class": ChatOpenAI,
        "kwargs": {
            "temperature": 0,
            "max_tokens": None,
            "timeout": None,
            "max_retries": 2,
            "model": "ollama/deepseek-r1:8b",
            "api_key": LITE_LLM_KEY,
            "base_url": LITE_LLM_BASE_URL,
        }
    },
    "content_cleaning": {
        "class": ChatOpenAI,
        "kwargs": {
            "temperature": 0,
            "max_tokens": None,
            "timeout": None,
            "max_retries": 2,
            "model": "openai/gpt-4.1-nano",
            "api_key": LITE_LLM_KEY,
            "base_url": LITE_LLM_BASE_URL,
        }
    },
    "Agents": {
        "class": ChatOpenAI,
        "kwargs": {
            "temperature": 0.1,
            "max_tokens": None,
            "timeout": None,
            "max_retries": 2,
            "model": "openai/gpt-4.1-nano",
            "api_key": LITE_LLM_KEY,
            "base_url": LITE_LLM_BASE_URL,
        }
    },
    "batch_processing": {
        "model": "gpt-4.1-nano",  # OpenAI model for batch API (cost-effective)
        "api_key": os.getenv("OPENAI_API_KEY", LITE_LLM_KEY),  # Direct OpenAI API key
        "base_url": None,  # Use direct OpenAI API for batch processing
        "batch_size": 50,  # Number of requests per batch
        "completion_window": "24h",  # 24-hour processing window
        "poll_interval": 60  # Check status every 60 seconds
    }
}

def get_llm_model(task: str):
    """Get the appropriate LLM model for a given task"""
    if task not in MODEL_CONFIGS:
        raise ValueError(f"Invalid task: {task}")
    return MODEL_CONFIGS[task]["class"](**MODEL_CONFIGS[task]["kwargs"])



# Page ranges for NVIDIA reports
NVIDIA_PAGE_RANGES = {
    "2023": {"start": 59, "end": 87, "notes_section": "Notes to Consolidated Financial Statements"},
    "2024": {"start": 63, "end": 64, "notes_section": "Notes to Consolidated Financial Statements"},
    # "2024": {"start": 55, "end": 83, "notes_section": "Notes to Consolidated Financial Statements"}
}

REMOVE_PATTERNS = [
            r"Table of Contents",
            r"NVIDIA Corporation and Subsidiaries",
            r"Notes to the Consolidated Financial Statements",
            r"\(Continued\)",
            r"=== PAGE \d+ ===",
            r"^\s*$",  # Empty lines
            r"^\s*\d+\s*$",  # Lines with only page numbers
            r"^\s*Note\s+\d+\s*$",  # Lines with only "Note X"
        ]

ITEM_STRUCTURE = {
    # PART I
    "item1":  "Business",
    "item1A": "Risk Factors",
    "item1B": "Unresolved Staff Comments",
    "item1C": "Cybersecurity",                     # added 2023
    "item2":  "Properties",
    "item3":  "Legal Proceedings",
    "item4":  "Mine Safety Disclosures",          # “Not applicable” for most non-mining filers

    # PART II
    "item5":  ("Market for Registrant’s Common Equity, Related Stockholder Matters and "
               "Issuer Purchases of Equity Securities"),
    "item6":  "[Removed and Reserved]",           # Selected Financial Data was eliminated in 2021
    "item7":  ("Management’s Discussion and Analysis of Financial Condition and "
               "Results of Operations"),
    "item7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "item8":  "Financial Statements and Supplementary Data",
    "item9":  ("Changes in and Disagreements with Accountants on Accounting and "
               "Financial Disclosure"),
    "item9A": "Controls and Procedures",
    "item9B": "Other Information",
    "item9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",  # HFCAA

    # PART III
    "item10": "Directors, Executive Officers and Corporate Governance",
    "item11": "Executive Compensation",
    "item12": ("Security Ownership of Certain Beneficial Owners and Management and "
               "Related Stockholder Matters"),
    "item13": ("Certain Relationships and Related Transactions, and Director "
               "Independence"),
    "item14": "Principal Accounting Fees and Services",

    # PART IV
    "item15": "Exhibits and Financial Statement Schedules",
    "item16": "Form 10-K Summary"                 # optional
}

EXTRACTED_RATIOS = {
    "Income Statement": {
        "Revenue": "Revenue",
        "Cost of Goods Sold (COGS)": "Cost of Goods Sold",
        "Gross Profit": "Gross Profit",
        "Operating Expenses": "Operating Expenses",
        "Operating Income": "Operating Income",
        "Net Income": "Net Income"
    },
    "Balance Sheet": {
        "Total Assets": "Total Assets",
        "Current Assets": "Current Assets",
        "Non-Current Assets": "Non-Current Assets",
        "Total Liabilities": "Total Liabilities",
        "Current Liabilities": "Current Liabilities",
        "Non-Current Liabilities": "Non-Current Liabilities",
        "Shareholders' Equity": "Shareholders' Equity",
        "Retained Earnings": "Retained Earnings",
        "Total Equity and Liabilities": "Total Equity and Liabilities"
    },
    "Cash Flow Statement": {
        "Cash from Operating Activities": "Cash from Operating Activities",
        "Cash used in Operating Activities": "Cash used in Operating Activities",
        "Cash from Investing Activities": "Cash from Investing Activities",
        "Cash used in Investing Activities": "Cash used in Investing Activities",
        "Cash from Financing Activities": "Cash from Financing Activities",
        "Cash used in Financing Activities": "Cash used in Financing Activities",
        "Net Cash Flow from Operations": "Net Cash Flow from Operations",
        "Net Cash Flow from Investing": "Net Cash Flow from Investing",
        "Net Cash Flow from Financing": "Net Cash Flow from Financing",
        "Net Increase/Decrease in Cash": "Net Increase/Decrease in Cash"
    }
}

EXTRACTED_FIELDS = {
    "Basic Information": {
        "Company Name": "The official full name of the company",
        "Establishment Date": "The date the company was established",
        "Headquarters Location": "The full name of the location of the company's headquarters"
    },
    "Board Composition": {
        "Name": "The full name of the board member",
        "Position": "The position of the board member",
        "Salary": "The salary of the board member",
        "Bonus": "The bonus of the board member",
        "Stock Awards": "The stock awards of the board member",
        "Stock Options": "The stock options of the board member"
    }
}

CALCULATED_RATIOS = {
    "Key Financial Metrics": {
        "Gross Margin": "Gross Margin",
        "Operating Margin": "Operating Margin",
        "Net Profit Margin": "Net Profit Margin",
        "Current Ratio": "Current Ratio",
        "Quick Ratio": "Quick Ratio",
        "Debt-to-Equity Ratio": "Debt-to-Equity Ratio",
        "Interest Coverage Ratio": "Interest Coverage Ratio",
        "Asset Turnover": "Asset Turnover",
        "Inventory Turnover": "Inventory Turnover",
        "Return on Equity (RoE)": "Return on Equity (RoE)",
        "Return on Assets (RoA)": "Return on Assets (RoA)",
        "Earnings Before Interest & Taxes (EBIT)": "Earnings Before Interest & Taxes (EBIT)",
        "Dividend Payout Ratio": "Dividend Payout Ratio",
        "Earnings Before Interest, Taxes, Depreciation and Amortisation (EBITDA)": "Earnings Before Interest, Taxes, Depreciation and Amortisation (EBITDA)"
    }
}

SUMMARY_SECTIONS = {
    "Company Information": {
        "Company Name": "What is the official full name of the company?",
        "Establishment Date": "when was the date the company was established or incorporated?",
        "Headquartered Location": "where is the company's headquarters located?"
    },
    "Core Competencies": {
        # "Description": "This section provides information about the company’s core competencies, including its innovation strengths, product advantages, brand recognition, and reputation ratings, offering readers insight into the company’s competitive strengths and unique value propositions.",
        "Innovation Advantages": "What are the company's key innovation strengths or differentiators?",
        "Product Advantages": "What product advantages make the company stand out in the market?",
        "Brand Recognition": "How is the company perceived in terms of brand awareness and recognition?",
        "Reputation Ratings": "What are the company's reputation or ESG ratings across sources?"

    },
    "Mission & Vision": {
        # "Description": "This section provides information about the company's purpose and long-term goals, including its mission and vision statements, offering readers a clear understanding of the company's strategic direction and aspirations.",
        "Mission Statement": "What is the company's stated mission?",
        "Vision Statement": "What is the company's long-term vision?",
        "Core Values": "What are the company's core values guiding its operations?"
    },
    "Overall Summary": {
        # "Description": "This section offers an overall summary of the company's financial performance, to help readers get a better understanding of the strengths and weaknesses of the company's financial situation.",
        "Comprehensive financial health": "What is the company’s overall financial health (e.g., liquidity, solvency, coverage)?",
        "Profitability & Earnings Quality": "How profitable is the company and how consistent is its earnings quality?",
        "Operational Efficiency": "How efficient is the company in using its assets and managing costs?",
        "Financial risk identification and early warning": "What financial risks have been identified and what early warnings exist?",
        "Future financial performance projection": "What is the forecast or projection for the company’s financial performance?"
    },
    "Internal Controls": {
        # "Description": "This section summarizes the company's internal control framework over the report.",
        "Risk Assessment Procedures": "What risk assessment procedures does the company have in place?",
        "Control Activities": "What key control activities are implemented by the company?",
        "Monitoring Mechanisms": "How does the company monitor internal controls?",
        "Material Weaknesses": "What material weaknesses or deficiencies have been identified?",
        "Planned Improvements": "What improvements to internal controls are being planned or implemented?",
        "Control Effectiveness": "How effective are the current internal control mechanisms?"
    },
    "Management Discussion and Analysis": {
        # "Description": "This section summarizes management's perspectives on the company's financial and operating results.",
        "Key performance drivers": "What are the primary drivers (e.g., volume, pricing, costs) that affected results year-over-year according to management?",
        "Financial conditions": "What is the company’s liquidity position and capital resource strategy, including debt, cash flow, and CAPEX?",
        "Significant trends or uncertainties affecting the business": "What trends, risks, or uncertainties has management highlighted that may impact future performance?"
    },
    "Risk Factors": {
        # "Description": "This section provides an overview of various risk factors impacting the company, offering readers a comprehensive understanding of potential challenges and threats.",
        "Market Risks": "What market-related risks (e.g., competition, demand, pricing) does the company face?",
        "Operational Risks": "What operational risks (e.g., supply chain, labor, systems) are identified?",
        "Financial Risks": "What financial risks (e.g., leverage, FX, interest rates) are disclosed?",
        "Compliance Risks": "What compliance, legal, or regulatory risks does the company face?"
    },
    "Innovation and Development Plans": {
        # "Description": "This section outlines the company's innovation and development plans, including its innovation initiatives, development plans, and development risks.",
        "Innovation Initiatives": "What current or planned innovation initiatives has the company disclosed?",
        "Development Plans": "What are the company’s development strategies or project roadmaps?",
        "Development Risks": "What are the key risks related to R&D and product development?"
    },
    "Business Competitiveness" :{
        # "Description": "This section provides information about the core business activities and competitive advantages, offering readers a comprehensive understanding of the company's business model, products/services, market position.",
        "Business Model": "What is the company’s core business model (e.g., subscription, platform, product sales)?",
        "Market Position": "What is the company’s competitive position in key markets (e.g., leader, challenger)?"
    }
}

ANALYSIS_SECTIONS = {
    "Financial Performance": {
        "Description": "This section analyzes the company's financial performance, including its revenue, gross profit, operating income, net income, and other financial metrics.",
        "Profitability": "Why has the company's profitability improved or declined?",
        "Liquidity": "Why has the company’s liquidity position strengthened or weakened recently?",
        "Solvency": "Why is the company’s solvency improving or deteriorating, and what is the role of debt in this?",
        "Asset Turnover": "Why is the company’s asset turnover ratio increasing or decreasing?",
        "Cash Flow": "Why is the company experiencing cash flow fluctuations, and what does it indicate about its operational health?"
    },
}

TOPIC_KEYWORDS = {
      # Management Discussion and Analysis (MDA) Topics
      "DRIVERS": [
          "revenue growth", "performance drivers", "key drivers", "growth factors",
          "operating factors", "volume growth", "price changes", "segment mix",
          "geographic mix", "cost control", "productivity gains", "earnings boost",
          "market share", "competitive position", "demand trends", "pricing power"
      ],
      "LIQUIDITY": [
          "liquidity", "cash flow", "operating cash flow", "investing cash flow",
          "financing cash flow", "cash position", "cash balances", "marketable securities",
          "credit facilities", "debt levels", "debt maturities", "capital resources",
          "working capital", "current ratio", "quick ratio", "cash conversion"
      ],
      "FIN_COND": [
          "financial condition", "capital expenditure", "capex", "commitments",
          "covenant", "debt service", "interest coverage", "leverage ratio",
          "credit rating", "borrowing capacity", "financial flexibility",
          "capital structure", "debt refinancing", "capital allocation"
      ],
      "TRENDS": [
          "trends", "uncertainties", "forward looking", "future performance",
          "outlook", "guidance", "demand cycles", "supply chain", "pricing pressure",
          "regulatory developments", "geopolitical", "foreign exchange",
          "inflation", "technological disruption", "market conditions",
          "economic environment", "industry trends", "competitive dynamics"
      ],
      
      # Risk Factors Topics
      "MARKET_RISKS": [
          "market risk", "market volatility", "economic downturn", "recession",
          "interest rate risk", "currency risk", "foreign exchange", "commodity prices",
          "demand fluctuation", "market competition", "pricing pressure",
          "customer concentration", "market conditions", "economic conditions"
      ],
      "OPERATIONAL_RISKS": [
          "operational risk", "supply chain", "manufacturing", "production",
          "operational disruption", "business interruption", "capacity constraints",
          "quality control", "product defects", "recalls", "operational efficiency",
          "key personnel", "talent retention", "cybersecurity", "data breach"
      ],
      "FINANCIAL_RISKS": [
          "financial risk", "credit risk", "counterparty risk", "liquidity risk",
          "debt obligations", "covenant violations", "cash flow", "financing",
          "capital requirements", "funding", "refinancing", "impairment",
          "bad debt", "collection", "financial instruments", "derivatives"
      ],
      "COMPLIANCE_RISKS": [
          "regulatory risk", "compliance", "legal proceedings", "litigation",
          "regulatory changes", "government regulation", "environmental regulation",
          "safety regulations", "data privacy", "intellectual property",
          "patent infringement", "regulatory approval", "license", "permits"
      ],
      
      # Business Competitiveness Topics
      "BUSINESS_MODEL": [
          "business model", "revenue model", "subscription", "licensing",
          "sales model", "distribution channels", "go-to-market", "value proposition",
          "monetization", "pricing strategy", "business strategy", "operating model"
      ],
      "PRODUCTS_SERVICES": [
          "products", "services", "product portfolio", "service offerings",
          "product development", "product features", "product differentiation",
          "innovation", "technology", "solutions", "platforms", "applications"
      ],
      "MARKET_POSITION": [
          "market share", "market position", "market leadership", "market penetration",
          "competitive position", "industry ranking", "market presence",
          "brand recognition", "customer base", "market segments", "addressable market"
      ],
      "COMPETITIVE_LANDSCAPE": [
          "competitors", "competition", "competitive landscape", "rival companies",
          "competitive threats", "competitive advantages", "differentiation",
          "market dynamics", "industry competition", "competitive pressures"
      ],
      "COMPETITIVE_ADVANTAGES": [
          "competitive advantage", "differentiation", "unique selling proposition",
          "proprietary technology", "patents", "intellectual property", "brand strength",
          "cost advantages", "scale benefits", "network effects", "barriers to entry"
      ],
      
      # Core Technology and R&D Topics
      "CORE_TECHNOLOGY": [
          "technology", "platform", "infrastructure", "architecture", "systems",
          "proprietary technology", "technical capabilities", "technology stack",
          "core competencies", "technological advantages", "digital transformation"
      ],
      "RND_EFFORTS": [
          "research and development", "R&D", "innovation", "product development",
          "research", "development programs", "technology development", "engineering",
          "research investments", "development costs", "innovation pipeline"
      ],
      "INNOVATION_CAPABILITIES": [
          "innovation", "innovation capabilities", "technological innovation",
          "product innovation", "process innovation", "disruptive innovation",
          "innovation strategy", "innovation culture", "creative solutions"
      ],
      "IP_PORTFOLIO": [
          "intellectual property", "patents", "trademarks", "copyrights",
          "trade secrets", "proprietary rights", "IP portfolio", "patent portfolio",
          "licensing", "IP protection", "patent applications", "IP strategy"
      ],
      
      # Financial Performance Topics
      "PROFITABILITY": [
          "profitability", "profit margins", "gross margin", "operating margin",
          "net margin", "earnings", "EBITDA", "operating income", "net income",
          "return on equity", "return on assets", "profit improvement", "margin expansion"
      ],
      "SOLVENCY": [
          "solvency", "debt management", "leverage", "debt-to-equity", "financial stability",
          "creditworthiness", "debt service", "financial strength", "capital adequacy",
          "debt capacity", "financial flexibility", "balance sheet strength"
      ],
      "ASSET_TURNOVER": [
          "asset turnover", "asset utilization", "asset efficiency", "working capital",
          "inventory turnover", "receivables", "asset management", "capital efficiency",
          "resource utilization", "operational efficiency", "productivity"
      ],
      
      # Company Information Topics
      "COMPANY_INFO": [
          "company", "corporation", "business", "organization", "enterprise",
          "headquarters", "incorporation", "establishment", "founding", "history",
          "company profile", "corporate structure", "subsidiaries", "operations"
      ],
      
      # Core Competencies Topics
      "INNOVATION_ADVANTAGES": [
          "innovation advantages", "innovation strengths", "technological leadership",
          "research capabilities", "innovation ecosystem", "creative excellence",
          "breakthrough technologies", "disruptive capabilities", "innovation culture"
      ],
      "PRODUCT_ADVANTAGES": [
          "product advantages", "product superiority", "product quality", "product features",
          "product performance", "product differentiation", "unique products",
          "premium products", "product excellence", "product innovation"
      ],
      "BRAND_RECOGNITION": [
          "brand recognition", "brand awareness", "brand value", "brand equity",
          "brand reputation", "brand loyalty", "brand strength", "brand image",
          "brand positioning", "brand trust", "market recognition"
      ],
      
      # Mission & Vision Topics
      "MISSION_STATEMENT": [
          "mission statement", "company mission", "corporate mission", "purpose",
          "organizational purpose", "core mission", "mission objectives",
          "strategic mission", "company purpose", "business mission"
      ],
      "VISION_STATEMENT": [
          "vision statement", "company vision", "corporate vision", "future vision",
          "strategic vision", "long-term vision", "organizational vision",
          "vision goals", "aspirational vision", "vision objectives"
      ],
      "CORE_VALUES": [
          "core values", "company values", "organizational values", "corporate values",
          "fundamental values", "guiding principles", "ethical values",
          "cultural values", "business principles", "value system"
      ],
      
      # Internal Controls Topics
      "RISK_ASSESSMENT": [
          "risk assessment", "risk evaluation", "risk identification", "risk analysis",
          "risk management", "risk procedures", "control environment",
          "risk monitoring", "risk mitigation", "internal audit"
      ],
      "CONTROL_ACTIVITIES": [
          "control activities", "internal controls", "control procedures",
          "control mechanisms", "control systems", "control processes",
          "operational controls", "financial controls", "compliance controls"
      ],
      "MONITORING_MECHANISMS": [
          "monitoring mechanisms", "control monitoring", "ongoing monitoring",
          "surveillance systems", "oversight mechanisms", "monitoring procedures",
          "continuous monitoring", "performance monitoring", "control oversight"
      ]
  }

MYSQL_HOST=os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT=3306
MYSQL_USER=os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD=os.getenv("MYSQL_PASSWORD", "AidfDmtAlphaRoot")
MYSQL_DATABASE=os.getenv("MYSQL_DATABASE", "level1_test")
MYSQL_TABLE=os.getenv("MYSQL_TABLE", "ent_company_information_bbg_compustat")