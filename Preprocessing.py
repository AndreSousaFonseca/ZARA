import string

### EXPLORE THE SIZE OF EACH DATASET ####

#for i in dataframes:
#    print (i.shape) 
#(16406, 2)
#(16407, 3)
#(47441, 13)

#len(dataframes[1]["Question"].unique())
# 14979


### CHECK THE COLUMNS OF EACH DATASET ####

#for i in dataframes:
#    print (i.columns)
    
#Index(['question', 'answer'], dtype='object')
#Index(['qtype', 'Question', 'Answer'], dtype='object')
#Index(['document_id', 'document_source', 'document_url', 'category',
#       'umls_cui', 'umls_semantic_types', 'umls_semantic_group', 'synonyms',
#       'question_id', 'question_focus', 'question_type', 'question', 'answer'],
#      dtype='object')

### CHECK QUESTION FOCUS ####
#len(dataframes[2]["question_focus"].unique())
#Out[142]: 10538

""" The question_focus can help us to browse through our vector database
faster. This means that i can use thsi information as metadata to apply 
metadata filters during retrieval, limiting the search space to only those 
entries relevant to the specific disease type. By adding Metadata 
I can combine semantic similarity search with metadata filters.
For this we are going to use scyspacy's pre-trained model 
en_ner_bc5cdr_md  which as disease records
https://allenai.github.io/scispacy/

! pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz


# Instead I could have used Metathesaurus:
https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html

from scispacy.umls_linking import UmlsEntityLinker
self.nlp = spacy.load("en_core_sci_md")
linker = UmlsEntityLinker()
self.nlp.add_pipe("scispacy_linker", config={"linker": linker})
"""



class DiseaseExtractor:
    def __init__(self, sci_nlp):
        self.sci_nlp = sci_nlp
        
    def extract_diseases(self, df):
        disease = []
        for i, question in enumerate(df["question"]):
            print(i)
            doc = self.sci_nlp(question)
            diseases_in_question = [
                ''.join(char for char in ent.text if char not in string.punctuation)
                for ent in doc.ents if ent.label_ == "DISEASE"]
    
        # Append the first disease found or "UNKNOWN" if none
            if diseases_in_question:
                disease.append(diseases_in_question[0])
            else:
                disease.append("UNKNOWN")
        return disease

