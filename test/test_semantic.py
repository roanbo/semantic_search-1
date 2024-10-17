#import unittest
import pytest
import numpy as np
import pandas as pd
from src.main import DataLoader, CosineSimilarityStrategy, EmbeddingModel, SemanticSearch

def test_file_not_found():
    loader = DataLoader('non_existent_file.csv')
    with pytest.raises(FileNotFoundError):
        loader.load_data()

def test_load_data():
    loader = DataLoader('./basedatos/IMDB top 1000.csv')
    df = loader.load_data()
    assert isinstance(df, pd.DataFrame)
    assert 'newcol' in df.columns
    
def test_generate_embeddings():
    df = pd.DataFrame({'newcol': ['movie title description']})
    model = EmbeddingModel()
    df = model.generate_embeddings(df)
    assert 'embeddings' in df.columns

def test_search():
    df = pd.DataFrame({
        'Title': ['Matrix Recargado'],
        'Description': ['pelicula de ciencia ficcion'],
        'Cast': ['Cast 1'],
        'Genre': ['Genre 1'],
        'embeddings': [np.random.rand(384).astype(np.float32)]
    })
    query = 'matrix'
    model = EmbeddingModel().model
    search_engine = SemanticSearch(model, CosineSimilarityStrategy())
    result = search_engine.search(df, query)
    assert len(result) == 1
    assert 'similarity' in result.columns

