# Import relevant libraries
import pandas as pd
from sklearn.model_selection import train_test_split
import spacy
from LoadFiles import DataLoad
from Preprocessing import DiseaseExtractor
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
from RAGModel import RAG

# Load datasets
file_paths =[
    'mle_screening_dataset.csv',
    'hf://datasets/lavita/MedQuAD/data/*.parquet'
    ]

# Load the data
data_loader = DataLoad()
dataframes = data_loader.load_multiple(file_paths)

#  Extract the disease (question_focus) for the 'mle_screening_dataset.csv'
sci_nlp = spacy.load("en_ner_bc5cdr_md") # model used to perform the disease extraction
extractor = DiseaseExtractor(sci_nlp)
diseases  = extractor.extract_diseases(dataframes[0]) #Apply extraction solely for the first dataframe


# Prepare each dataset to concatenate:
dataframes[0]['question_focus'] = diseases 

# Obtain relevant columns from the 'hf://datasets/lavita/MedQuAD/data/*.parquet' data
dataframes[1] = dataframes[1][['question_focus','question', 'answer']]

def reorder_dataframes(dataframes: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """Order both dataframes so we obtain the same colum order for each dataframe"""
    desired_order = ['question_focus','question', 'answer']
    return [df[desired_order] for df in dataframes]

# Call the functon that orders the columns
ordered_dataframes = reorder_dataframes(dataframes)
        
# Concatenate the data
df_combined = pd.concat([dataframes[0], dataframes[1]], axis=0, ignore_index=True)

# Split the data into training, validation, and testing sets
# First, split out the test set (20% of total)
train_val_df, test_df = train_test_split(df_combined, test_size=0.2, random_state=42)
# Then split the remaining data into training and validation (20% of train_val_df for validation)
train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42)


def convert_df_to_documents(df):
    """Convert DataFrame rows to LangChain Document objects with metadata"""
    documents = []
    
    for idx, row in df.iterrows():
        # Prepare the content
        content = f"Question: {row['question']}\nAnswer: {row['answer']}"
        
        # Prepare metadata
        metadata = {
            "disease_type": row["question_focus"],
        }
        
        # Create Document object
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)
    
    return documents

# Convert dataframes into LangChain Document objects with metadata
my_documents = convert_df_to_documents(train_df)


# Split the data into more managle sizes (default = macimum of 512 tokens)
text_splitter = RecursiveCharacterTextSplitter(
     chunk_size=512,
     chunk_overlap=10,
     length_function=len,
     add_start_index=True
)

# Split the documents into chunks
chunks = text_splitter.split_documents(my_documents)


def build_vector_store(documents):
    """Build a FAISS vector store from documents using medical embeddings"""
    # Use a medical-specific embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="pritamdeka/S-PubMedBert-MS-MARCO"
        )
    
    # Create FAISS index
    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )
    
    return vector_store


def main():
    vector_store = build_vector_store(chunks)
    
    rag = RAG(vector_store)
    # Example queries
    example_queries = [
        "What are the symptoms of diabetes?",
        "How is pneumonia treated?",
        "What are common medications for hypertension?",
        "Can you tell me about the complications of asthma?"
    ]
    
    # Answer queries
    for query in example_queries:
        result = rag.answer_query(query)
        answer = result["answer"]
        print("\nAnswer:")
        print(answer)
    
        # Calculate cosine similarity
        similarity = cosine_similarity([answer], [train_val_df["answer"][1]])[0][0]
        print(f"Similarity: {similarity}")
    
        # Calculate f1_score
        f1 = f1_score([answer], [train_val_df["answer"][1]])
        print(f"F1 Score: {f1}")
    


if __name__ == "__main__":
    main()
