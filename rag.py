
from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import TokenTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer


# this function is called to load environment variables from a .env file
load_dotenv()

# Constants
CHUNK_SIZE = 500
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
## EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "real_estate"

# create global variables for llm and vector_store
llm = None
vector_store = None

def initialize_components():
    """ 
    This function initializes the LLM and vector store components.
    It checks if they are already initialized and creates them if not.
    """
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.9, max_tokens=500)

    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}
        )

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTORSTORE_DIR)
        )

def hf_tokenizer_splitter(texts, model_name, chunk_size=500, chunk_overlap=50):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    chunks = []
    for doc in texts:
        tokens = tokenizer.encode(doc.page_content, add_special_tokens=False)
        for i in range(0, len(tokens), chunk_size - chunk_overlap):
            chunk_tokens = tokens[i:i+chunk_size]
            chunk_text = tokenizer.decode(chunk_tokens)
            # Create a new Document object with the chunked text
            chunks.append(type(doc)(page_content=chunk_text, metadata=doc.metadata))
    return chunks

def process_urls(urls):
    """
    This function scraps data from a url and stores it in a vector db
    :param urls: input urls
    :return:
    """

    yield "Initializing Components.....✅"
    initialize_components()

    yield "resetting vector store on every run to create fresh set of vectors everytime...✅"
    vector_store.reset_collection()

    yield "Reading URL data...✅"
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    yield "Splitting text into chunks...✅"
    #docs = hf_tokenizer_splitter(data, EMBEDDING_MODEL, chunk_size=500, chunk_overlap=50)

    # Debug: Check max token length in chunks
    #max_tokens = max(len(AutoTokenizer.from_pretrained(EMBEDDING_MODEL).encode(doc.page_content, add_special_tokens=False)) for doc in docs)
    #print(f"Max tokens in any chunk: {max_tokens}")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE
        )
    docs = text_splitter.split_documents(data)

    # text_splitter = TokenTextSplitter(
    #     chunk_size=500,        # tokens, adjust as needed (≤ 1024)
    #     chunk_overlap=50       # tokens, optional
    # )
    # docs = text_splitter.split_documents(data)

    yield "Add chunks to vector database...✅"
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs, ids=uuids)

    yield "Done adding docs to vector database...✅"

def generate_answer(query):
    """
    This function generates an answer to a query using the LLM and vector store.
    :param query: The question to be answered
    :return: The answer and sources
    """
    if not vector_store:
        raise RuntimeError("Vector database is not initialized ")

    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_store.as_retriever())
    result = chain.invoke({"question": query}, return_only_outputs=True)
    sources = result.get("sources", "")

    return result['answer'], sources

if __name__ == "__main__":
    urls = [
        "https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html",
        "https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html"
    ]

    process_urls(urls)
    answer, sources = generate_answer("Tell me what was the 30 year fixed mortagate rate for the week ending December 19 along with the date ?")
    print(f"Answer: {answer}")
    print(f"Sources: {sources}")