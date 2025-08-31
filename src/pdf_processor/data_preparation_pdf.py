import json
import asyncio
import os
from typing import Dict, List, Any
from datetime import datetime
import logging
from pathlib import Path
import re
from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, prepare_env, read_fn

from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser

from src.config import MILVUS_URL, MILVUS_DB_NAME, MILVUS_PW
from src.database.milvus_handler import MilvusHandler
from src.pdf_extractor.mineru_parser import parse_doc
from src.prompts.prompt import CONTENT_SEARCHING_PROMPT
from src.utils.utils import Utils

class PDFProcessor(MilvusHandler):

    def __init__(self):
        super().__init__(host=MILVUS_URL, db_name=MILVUS_DB_NAME, password=MILVUS_PW)
        # __dir__ = os.path.dirname(os.path.abspath(__file__))

        """
        START HERE:
        This part of code is from minerU: https://github.com/opendatalab/MinerU/blob/master/demo/demo.py
        """
        __dir__ = '../../data'
        pdf_files_dir = os.path.join(__dir__, "input")
        self.output_dir = os.path.join(__dir__, "output")
        pdf_suffixes = [".pdf"]
        image_suffixes = [".png", ".jpeg", ".jpg"]

        self.doc_path_list = []
        for doc_path in Path(pdf_files_dir).glob('*'):
            if doc_path.suffix in pdf_suffixes + image_suffixes:
                self.doc_path_list.append(doc_path)
        """END HERE"""

        self.llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
        self.parser = JsonOutputParser()

        self.utils = Utils()

        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)

    def extract_company_year(self, file_name):
        file_name = file_name + '.pdf'
        pattern = re.compile(r"^(?P<company>.+)_(?P<year>\d{4})\.pdf$")
        match = pattern.match(file_name)
        result = {
            "company": match.group("company"),
            "year": int(match.group("year"))
        }
        return result

    # def extract_data_from_multi_pdf(self):
    #     # file_name_list = []
    #     # pdf_bytes_list = []
    #     # lang_list = []
    #     md_content_list = []
    #
    #     for path in self.doc_path_list:
    #         file_name = str(Path(path).stem)
    #         pdf_bytes = read_fn(path)
    #         lang = 'en'
    #         # file_name_list.append(file_name)
    #         # pdf_bytes_list.append(pdf_bytes)
    #         # lang_list.append(lang)
    #         page_results_dict = self.look_for_session_pages(pdf_bytes, file_name, lang)
    #
    #     return page_results

    def extract_data_from_pdf(self,
                              pdf_bytes: bytes,
                              file_name: str,
                              lang: str,
                              page_start: int=None,
                              page_end: int=None,
                              md_name: str=None):

        md = parse_doc(pdf_bytes,
                      self.output_dir,
                      file_name=file_name,
                      lang=lang,
                      backend="pipeline", start_page_id=page_start, end_page_id=page_end, md_name=md_name)
        return md


    def look_for_session_pages(self,
                           pdf_bytes: bytes,
                           file_name: str,
                           lang: str):

        md_contents = self.extract_data_from_pdf(pdf_bytes, file_name, lang, page_start=0, page_end=3, md_name='contents')
        prompt_template = PromptTemplate(
            template=CONTENT_SEARCHING_PROMPT,
            input_variables=["input_markdown"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )
        chain = prompt_template | self.llm | self.parser
        result = chain.invoke({'input_markdown': md_contents})

        return result

    async def session_chunking(self, md_sections, session_name, company, year):

        session_chunks = []
        # Step 1: Chunk the content
        chunks = self.utils.split(md_sections)
        self.logger.info(f"   📄 Created {len(chunks)} chunks")

        # Step 2: Generate embeddings for chunks
        chunk_embeddings = await self.utils.a_embed_documents(chunks)
        chunks = chunk_embeddings['text']
        chunk_embeddings = chunk_embeddings['vector']

        if not chunk_embeddings or len(chunk_embeddings) != len(chunks):
            self.logger.error(f"   ❌ Embedding failed for {session_name}")
            return None
        self.logger.info(f"   🧠 Generated {len(chunk_embeddings)} embeddings")

        for i, (chunk_text, embedding) in enumerate(zip(chunks, chunk_embeddings)):
            if len(chunk_text) >= 2000:
                print('exceeding 2000')
                continue
            chunk_data = {
                "session_name": session_name,
                "company": company,
                "year": year,
                "chunk_text": chunk_text,
                "chunk_index": i,
                "chunk_length": len(chunk_text),
                "embedding": embedding,
                "created_at": datetime.now().isoformat()
            }
            print(len(chunk_text))
            session_chunks.append(chunk_data)
        return session_chunks


    async def extract_content_session(self,
                                    pdf_bytes: bytes,
                                    file_name: str,
                                    lang: str,
                                    page_results_dict):

        company_year = self.extract_company_year(file_name)
        company = company_year['company']
        year = str(company_year['year'])

        all_chunks = []

        for session_name, session_details in page_results_dict.items():
            if session_details['start'] == '' and session_details['end'] == '':
                ## LLM failed to generated pages skipping it
                continue
            page_start = int(session_details['start']) - 2 ## +/- to deal with the pdf page number does not align with actual number
            try:
                page_end = int(session_details['end']) + 2
            except:
                page_end = None ## last page
            if page_start < 0:
                page_start = 1
            md_sections = self.extract_data_from_pdf(pdf_bytes,
                                                     file_name,
                                                     lang,
                                                     page_start=page_start,
                                                     page_end=page_end,
                                                     md_name=session_name)
            if md_sections is not None:
                session_chunks = await self.session_chunking(md_sections, session_name, company, year)
                all_chunks.extend(session_chunks)
            else:
                continue
        return all_chunks

    async def insert_into_vdb(self, all_chunks):

        if all_chunks:

            chunks_stored = self.store_chunks(all_chunks)
            self.logger.info(f"✅ Stored {chunks_stored} chunks")
            # Log summary
            companies = set(chunk["company"] for chunk in all_chunks)
            years = set(chunk["year"] for chunk in all_chunks)
            sessions = set(chunk["session_name"] for chunk in all_chunks)

            self.logger.info(f"   📊 Companies: {', '.join(sorted(companies))}")
            self.logger.info(f"   📅 Years: {', '.join(sorted(years))}")
            self.logger.info(f"   📄 Sessions: {', '.join(sorted(sessions))}")
            return chunks_stored
        else:
            self.logger.warning("No chunks to store")
            return 0

    async def main(self):

        self._initialize_collection()

        md_content_list = []

        for path in self.doc_path_list:
            file_name = str(Path(path).stem)
            pdf_bytes = read_fn(path)
            if '宁德时代' in file_name:
                lang = 'ch'
            else:
                lang = 'en'

            # file_name_list.append(file_name)
            # pdf_bytes_list.append(pdf_bytes)
            # lang_list.append(lang)
            page_results_dict = self.look_for_session_pages(pdf_bytes, file_name, lang)

            with open(os.path.join(self.output_dir, file_name, 'content_results.json'), "w", encoding="utf-8") as f:
                json.dump(page_results_dict, f, ensure_ascii=False, indent=4)

            all_chunks = await self.extract_content_session(pdf_bytes, file_name, lang, page_results_dict)
            chunks_stored = await self.insert_into_vdb(all_chunks)





if __name__ == '__main__':

    pdf_processor = PDFProcessor()
    asyncio.run(pdf_processor.main())







