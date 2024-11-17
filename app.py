# Importing libraries
import chromadb
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from gpt4all import GPT4All
import os
import uuid

# Initialize ChromaDB and embedding model
client = chromadb.Client()
embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Define function to chunk and load PDF content
def load_and_chunk_pdf(file_path):
    # Use Langchain's PyPDFLoader to load the PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Adjust the chunk size
        chunk_overlap=10  # Slight overlap between chunks to retain context
    )
    chunked_docs = text_splitter.split_documents(documents)
    
    return chunked_docs

# Function to add documents to ChromaDB with embeddings
def add_documents_to_chroma(documents, collection_name="qa_collection"):
    try:
        client.delete_collection(name=collection_name)
    except:
        pass
    collection = client.create_collection(name=collection_name)

    import chromadb

    persistent_client = chromadb.PersistentClient()

    vector_store_from_client = Chroma(
    client=persistent_client,
    collection_name=collection_name,
    embedding_function=embedding_model,
)
    
    for doc in documents:
        # Generate embedding for each chunk
        embedding = embedding_model.embed_query(doc.page_content)

        # Generate a unique ID for each document chunk
        doc_id = str(uuid.uuid4())  # Generate a unique ID
        
        # Add chunked document and its embedding to ChromaDB
        collection.add(
            ids = [doc_id],
            documents=[doc.page_content], 
            embeddings=[embedding], 
            metadatas={"source": doc.metadata.get("source", "unknown"), "page": doc.metadata.get("page", "unknown"), "hnsw:space": "cosine"}
        )


# Function to query ChromaDB with cosine similarity
def query_chroma_db(question, collection_name="qa_collection", threshold=1.5):
    try:
        collection = client.get_collection(collection_name)
    except:
    #     upload()
    #     collection = client.get_collection(collection_name)
        return []
    query_embedding = embedding_model.embed_query(question)
    
    # Query ChromaDB for the most similar chunk
    results = collection.query(
        query_embeddings=[query_embedding], 
        n_results=2  # Return top result
    )
    documents=[]
    if results['documents']:
        for index, dist in enumerate(results['distances'][0]):
            if dist<threshold:
                documents.append(results['documents'][0][index])
    return documents

persist_directory = "./vectordb"

def vector_func(file_path):
    split_docs=load_and_chunk_pdf(file_path)
    vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding_model,  # Use the model's encode method
    persist_directory=persist_directory
    )

    # Persist the vector store
    vectorstore.persist()
    return vectorstore

# Create a similarity search function using ChromaDB's built-in search
def retrieve_context(question, file_path):
    vectorstore=vector_func(file_path)
    # Perform a similarity search to retrieve the most relevant documents
    relevant_docs = vectorstore.similarity_search(question, k=3)  # retrieve top 3 most similar documents
    # Concatenate the retrieved document texts into a single context
    context = " ".join([doc.page_content for doc in relevant_docs])
    return context

# Initialize LLM (Phi-3) for fallback answering
def init_llm():
    # generator = pipeline('text-generation', model='tiiuae/falcon-7b-instruct', max_length=200, device='cpu')
    # llm = HuggingFacePipeline(pipeline=generator)
    model = GPT4All(os.getcwd()+r"\model\Phi-3-mini-4k-instruct-q4.gguf")
    # model.load_model()
    return model

# Function to answer questions
def answer_question(question, llm):
    # First, search ChromaDB for a similar chunk
    similar_chunk = query_chroma_db(question)
    return similar_chunk

def upload(pdf_file):
    # Load and chunk a PDF file
    chunked_docs = load_and_chunk_pdf(pdf_file)
    # Add the PDF chunks to the ChromaDB collection
    add_documents_to_chroma(chunked_docs)

def output_func(question):
    llm = init_llm()
    # question = "What aspects of css is used? Give in bullet points"
    context = answer_question(question, llm)

    template = f"""You are a document analyzer which takes the {question}.Based on the {context} provided, answer the question. Do not give Start Of Line and End Of Line keywords. Also do not repeat the statements. 
    Follow the points while generating answer.
        1. Do not generate any irrelevant information.  
        2. Do not give any explanation. 
        3. Do not give repeated sentences. 
        4. Give the answer which is given in context only. Do not compare it with other meanings.
        5. Give point to point answer only.
        6. Do not create any new question and do not answer that question.
        7. Do not give the question which is asked.
        8. Do not give response if there is an answer.
        9. Do not give words like <|end|><|assistant|>
    """

    # prompt = PromptTemplate.from_template(template)

    # llm_chain = LLMChain(prompt=prompt, llm=llm)
    # print(llm.invoke(prompt))
    if len(context)>0:
        output=llm.generate(template, max_tokens=100, temp=0.0, repeat_penalty=1.18, top_k=1)
        # output = output.split('!!!')[1]
        output=output.replace("===","")
    else:
        output ="No information found."
    try:
        output=output.split("<|end|>")[0]
    except:
        output = output
    return output