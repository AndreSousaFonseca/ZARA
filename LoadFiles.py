#Load relevant libraries
import os
import dask.dataframe as dd
from typing import List


# Load the document
""" Given that we are working with large files we can use different strategies
to make the loading process even faster. This becomes even more benficial as
the size of the datasets become larger. Three of the different approaches can
be found bellow:
    
# 1. Chunking with pandas (eg. pandas):  Loads the CSV in chunks instead of 
the entire file into memory, useful when laoding large datasets:
    import pandas as pd
    chunk_size = 5000
    chunks = pd.read_csv('mle_screening_dataset.csv', chunksize=chunk_size)
    df_mle = pd.concat(chunks, ignore_index=True)
    
#2. Use Batch Loading (eg. torch): more suitable for images and tensors
in deep learning model trainig phases and, does not have a direct conversion
to pandas dataframe. Moreover can't directly read parquet files:
    from torch.utils.data import DataLoader
    data_loader = DataLoader(mle_screening_dataset.csv, batch_size=64, shuffle=True)
    link: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html    

3. Use parallel loading (eg. daksk): Loads data in chunks and 
processes them in parallel. optemized for pandas and works with parquet files:
    import dask.dataframe as dd
    chunks = pd.read_csv('mle_screening_dataset.csv', chunksize=chunk_size)
    https://docs.dask.org/en/stable/dataframe.html
    

Here I will be using dask as I will be working with pandas and parquet files
"""


class DataLoad:
    def __init__(self):
        """
        Initialize the DataLoad class.
        """
        self.data = None

    def dataload(self, file_path):
        """
        Load data from a .csv or .parquet file using Dask.
        
        Args:
            file_path (str): The path to the file.
        
        Returns:
            pd.DataFrame: The loaded data as a Pandas DataFrame.
        """
        self.extension = os.path.splitext(file_path)[1].lower()
        
        if self.extension == '.csv':
            self.load = dd.read_csv(file_path)
            self.data = self.load.compute()
            
        elif self.extension == '.parquet':
            self.load = dd.read_parquet(file_path)
            self.data = self.load.compute()
        
        else:
            raise ValueError("Unsupported file format. Please use .csv or .parquet files.")
        
        return self.data
   
    def load_multiple(self, file_paths: List[str]):
        """
        Load multiple files into a list of Pandas DataFrames.
        
        Args:
            file_paths (List[str]): A list of file paths.
        
        Returns:
            List[pd.DataFrame]: A list of loaded data as Pandas DataFrames.
        """
        return list(map(self.dataload, file_paths))
            
        


