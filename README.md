# RAG_with_Local_LLM

Discover how to build a Retrieval Augmented Generation (RAG) app in Python that enables you to query and chat with your files using open source models. This app utilizes the Ada model for embedding and the Llama2 model for interaction.

### Dependency
```
!pip install ollama
!pip install langchain
!pip install chromadb # vector storage
!pip install pypdf # pdf reader
!pip install pytest # unit test
!pip install langchain_openai
!pip install azure-identity # azure authentication
!pip install pymupdf # pdf reader
```

### Interact with Llama2
1. through ollama
```
import ollama
response = ollama.chat(model='llama2', messages=[
    {
        'role':'user',
        'content':'tell me a joke',
    },
])
print(response['message']['content'])
```
2. through langchain
```
from langchain_community.llms import Ollama

llm = Ollama(model = 'llama2')
result = llm.invoke('tell me a joke')
print(result)
```

### Load and chunk the local PDF
1. Load the PDF file
```
from langchain.document_loaders.pdf import PyPDFDirectoryLoader

# https://python.langchain.com/docs/modules/data_connection/document_loaders/
def load_documents():
    document_loader = PyPDFDirectoryLoader("D:\Workshop\Open Source LLMs\docs")
    return document_loader.load()

# see the loaded documents
documents = load_documents()
if documents: 
    print(documents[0]) 
else: 
    print("No documents loaded.")
```
2. Chunk the PDF file
```
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

# see the chunked documents
documents = load_documents()
chunks = split_documents(documents)
for chunk in chunks:
    print(chunk.page_content)
```

### Embedding with Azure OpenAI
For better embedding results, we use the Ada model from Azure OpenAI for embedding. However, you can also choose open-source or other models.
```
import os
from openai import AzureOpenAI

def get_openai_client():
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version="2024-02-01",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

def get_embeddings(client, texts):
    embeddings = []
    for text in texts:
        response = client.embeddings.create(
            input=text,
            model="ada"
        )
        embeddings.append(response.model_dump_json(indent=2))
    return embeddings

client = get_openai_client()
response = get_embeddings(client, ["My test text string"])

print(response)
```

```
def get_doc_embeddings():
    documents = load_documents()
    chunks = split_documents(documents)
    
    client = get_openai_client()
    doc_embeddings = []

    for chunk in chunks:
        embeddings = get_embeddings(client, [chunk.page_content])
        doc_embeddings.append(embeddings)

    return doc_embeddings

doc_embeddings = get_doc_embeddings()
print("Embedding Type:", type(doc_embeddings))
for index, embeddings in enumerate(doc_embeddings):
    if embeddings:
        print(f"Document {index + 1} First Chunk Embedding:")
        print(embeddings[0])
        print("\n")
```

### Calculate the vector similarity
In a Retrieval Augmented Generation (RAG) solution, calculating vector similarity is crucial because it allows the system to find and retrieve the most relevant pieces of information from a large dataset. By embedding documents and queries into vector space, the system can measure how similar they are, ensuring that the retrieved information closely matches the user's query. 

```
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def retrieve_documents(query_embedding, doc_embeddings, top_k=3):
    # Flatten the list of embeddings and keep track of document indices
    all_embeddings = [emb for sublist in doc_embeddings for emb in sublist]
    doc_indices = [i for i, sublist in enumerate(doc_embeddings) for emb in sublist]

    # Compute cosine similarity between the query and all document embeddings
    similarities = cosine_similarity([query_embedding], all_embeddings)[0]

    # Get the top-k indices of the most similar embeddings
    top_indices = np.argsort(similarities)[::-1][:top_k]

    # Return the top-k most similar document chunks
    return [doc_indices[idx] for idx in top_indices], [similarities[idx] for idx in top_indices]
```

### Covert the embedded JSON string into a dictionary
```
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def extract_embedding_from_json(json_string):
    # Convert the JSON string back to a dictionary
    data_dict = json.loads(json_string)
    return data_dict['data'][0]['embedding']

question = "What is the capital of France?"
json_query_embeddings = get_embeddings(get_openai_client(), [question])
query_embedding = [extract_embedding_from_json(embed) for embed in json_query_embeddings][0]
print("Embedding Type:", type(query_embedding))
```

### Generate the answer
```
def generate_answer(context, question):
    response = ollama.chat(
        model='llama2', messages = [
            {'role': 'system', 'content': context},
            {'role': 'user', 'content': question}
        ])
    return response['message']['content']

response = generate_answer("You are an Geography experts", "What is the capital of France?")
print(response)
```

```
import re
import json

def rag_application(question):
    json_query_embeddings = get_embeddings(get_openai_client(), [question])
    query_embedding = [extract_embedding_from_json(embed) for embed in json_query_embeddings][0]
    doc_embeddings = get_doc_embeddings()

    # Check the embeddings before using them
    print("Query Embedding Sample:", query_embedding[:5])
    print("Document Embeddings Sample:", doc_embeddings[:5])

    # Check the type of embeddings before using them
    print("Query Embedding Type:", type(query_embedding))
    print("Document Embeddings Type:", type(doc_embeddings))  

    doc_embedding_lists = []
    for embedding_data in doc_embeddings:
        match = re.search(r'"embedding": \[(.*?)\]', embedding_data[0], re.DOTALL)
        if match:
            embedding_str = match.group(1)
            embedding_str = embedding_str.replace('\n', '').replace(' ', '')
            doc_embedding_list = json.loads(f'[{embedding_str}]')
            doc_embedding_lists.append(doc_embedding_list)
        else:
            print("Embedding not found")

    query_embedding_np = np.array(query_embedding).reshape(1, -1)
    doc_embeddings_np = np.array(doc_embedding_lists)

    print("Query Embedding Shape:", query_embedding_np.shape)
    print("Document Embeddings Shape:", doc_embeddings_np.shape)

    similarities = cosine_similarity(query_embedding_np, doc_embeddings_np)[0]
    top_indices = np.argsort(similarities)[::-1]

    # top_indices, _ = retrieve_documents(query_embedding, doc_embedding_lists)
    context = ' '.join([chunks[idx].page_content for idx in top_indices])
    
    answer = generate_answer(context, question)
    return answer

question = "What are the core values of Contoso Electronics?"
print(rag_application(question))
```

## Ways to improve the retrieve accuracy
1. Explore different ways to calculate the vector similarity
2. Use pytest or prompt flow to evaluate the performance
