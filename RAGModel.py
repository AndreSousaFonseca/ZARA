import os
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

class RAG:
    def __init__(self, vector_store, model_name="llama3-70b-8192"):
        """
        Initialize Medical RAG system
        
        Parameters:
        vector_store: FAISS vector store with medical documents
        model_name: Groq model to use (default: llama3-70b-8192)
        """
        self.vector_store = vector_store
        
        # Set up the Groq LLM
        self.llm = ChatGroq(
            api_key=os.environ.get("GROQ_API_KEY"),
            model_name=model_name
        )
        
        # Set up the RAG chain
        self._setup_rag_chain()
    
    def _setup_rag_chain(self):
        """Set up the RAG chain using LangChain components"""
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_template("""
        You are a medical assistant providing information about diseases, symptoms, treatments, and other medical topics.
        Your responses should be accurate, helpful, and based solely on the medical context provided.
        
        Answer the following question based on the provided context.
        If the context doesn't contain enough information to answer the question, clearly state what's missing.
        Include a brief disclaimer that this is for informational purposes only and not medical advice.
        
        Context:
        {context}
        
        Question: {question}
        """)
        
        # Create the document chain
        document_chain = create_stuff_documents_chain(
            self.llm,
            prompt
        )
        
        # Create the retrieval chain
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        self.rag_chain = create_retrieval_chain(
            retriever,
            document_chain
        )
    
    def extract_medical_entities(self, text):
        """Extract medical entities from text"""
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append(ent.text)
        return entities
    
    def answer_query(self, query):
        """
        Process a medical query using RAG
        
        Parameters:
        query: User's medical question
        
        Returns:
        Dict with answer and sources
        """
        # Run the RAG chain
        response = self.rag_chain.invoke({"question": query})
        
        # Extract sources from context
        sources = []
        if "context" in response and response["context"]:
            for doc in response["context"]:
                if hasattr(doc, "metadata"):
                    sources.append({
                        "disease_type": doc.metadata.get("disease_type", "Unknown"),
                        "qa_id": doc.metadata.get("qa_id", "Unknown")
                    })
        
        return {
            "query": query,
            "answer": response["answer"],
            "sources": sources
        }
