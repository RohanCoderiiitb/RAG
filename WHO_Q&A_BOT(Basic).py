import os 
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, UnstructuredPowerPointLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma 
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from dotenv import load_dotenv 

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
USER_AGENT = os.getenv("USER_AGENT", "Mozilla/5.0")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Check your .env file format (no quotes).")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["USER_AGENT"] = USER_AGENT

class RAG_indexing:

    def __init__(self, urls, persist_dir, embeddingmodel, apikey, chunk_size=600, chunk_overlap=200):
        self.urls = urls
        self.persist_dir = persist_dir
        self.embeddingmodel = embeddingmodel
        self.apikey = apikey
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_documents(self):
        all_docs = []
        for url in self.urls:
            if url.endswith(".pdf"):
                loader = PyPDFLoader(url)
            elif url.endswith(".pptx"):
                loader = UnstructuredPowerPointLoader(url)
            elif url.endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(url)
            else:
                loader = WebBaseLoader(
                    web_path=[url]
                )
            docs = loader.load()
            for doc in docs:
                doc.metadata['source'] = url
            all_docs.extend(docs)
        return all_docs
    
    def split_documents(self, docs):
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size = self.chunk_size,
            chunk_overlap = self.chunk_overlap
        )
        splits = splitter.split_documents(docs)
        return splits
    
    def embedd_and_store(self, splits):
        embedding_model = GoogleGenerativeAIEmbeddings(model = self.embeddingmodel, google_api_key=self.apikey)
        vector_stores = Chroma.from_documents(documents=splits, embedding=embedding_model, persist_directory=self.persist_dir)
        vector_stores.persist()
        return vector_stores
    
    def build_indexing(self):
        docs = self.load_documents()
        splits = self.split_documents(docs)
        vector_stores = self.embedd_and_store(splits)
        retriever = vector_stores.as_retriever(
            search_type = "mmr",
            search_kwargs = {"k":10}
        )
        return retriever

def query_translation(raw_query, api_key, history=None):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.3
    )

    history_text = ""
    if history:
        for msg in history.messages:
            role = "User" if msg.type=="human" else "Assistant"
            history_text += f"{role}: {msg.content}\n"

    template = """You are a helpful assistant that rewrites user questions to improve document retrieval.
                  Use conversation history to clarify ambiguous or vague questions.

                  Conversation history: {history}
                  Original question: "{query}"
                  Rewritten retrieval query:"""

    prompt = ChatPromptTemplate.from_template(template)
    formatted_prompt = prompt.invoke({"query": raw_query, "history":history_text})
    rewritten_query = llm.invoke(formatted_prompt).content.strip()
    return rewritten_query

class Generator:

    def __init__(self, question, api_key, retriever, chat_history):
        self.question = question
        self.api_key = api_key
        self.retriever = retriever
        self.chat_history = chat_history

    def generate(self):
        history_msg = self.chat_history.messages
        history_text = ""
        for msg in history_msg:
            role = "User" if msg.type=="human" else "Assistant"
            history_text += f"{role}: {msg.content}\n"

        template = """You are a helpful AI assistant.
                      Use the conversation history and the retrieved context to answer the user's questions 
                      Conversation history: {history}
                      context: {context}
                      Question: {question}
                      answer:
                   """
        prompt = ChatPromptTemplate.from_template(template)

        llm = ChatGoogleGenerativeAI(
            model = "gemini-2.5-flash",
            temperature = 0,
            google_api_key = self.api_key        
        ) 
        retrieved_docs = self.retriever.invoke(self.question)
        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        formatted_prompt = prompt.invoke({"context":context, "question": self.question, "history": history_text})
        answer = llm.invoke(formatted_prompt)
        self.chat_history.add_user_message(self.question)
        self.chat_history.add_ai_message(answer.content)
        print(f"Answer: {answer.content}")
        sources = set(doc.metadata.get("source", "Unknown source") for doc in retrieved_docs)
        print("Sources:")
        for source in sources:
            print(f"🔗 {source}")
        print()
    
if __name__=="__main__":
    indexing = RAG_indexing(
        ["https://data.who.int/countries/356",
         "https://www.who.int/news-room/fact-sheets/detail/dengue-and-severe-dengue",
         "https://www.who.int/news-room/fact-sheets/detail/breast-cancer",
         "https://www.who.int/news-room/fact-sheets/detail/hiv-aids",
         "https://www.who.int/news-room/fact-sheets/detail/malaria",
         "https://www.who.int/news-room/fact-sheets/detail/diabetes",
         "./data/world-health-organization-who.pdf"],
        "./chroma-index",
        "models/embedding-001",
        GOOGLE_API_KEY
    )
    retriever = indexing.build_indexing()
    stop = False
    chat_history = InMemoryChatMessageHistory()
    while stop==False:
        question = input("Ask: ")
        formatted_query = query_translation(question, GOOGLE_API_KEY, chat_history)
        if question.lower() != "stop":
            generator = Generator(
            formatted_query,
            GOOGLE_API_KEY,
            retriever,
            chat_history
            )
            generator.generate()
        else:
            break 