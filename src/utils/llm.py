from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model_name="gpt-4o", temperature=0)