import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
from main import DataLoader, EmbeddingModel, SemanticSearch, CosineSimilarityStrategy, UserInterface

class TestDataLoader(unittest.TestCase):
    @patch('pandas.read_csv')
    @patch('os.path.exists')
    def test_load_data(self, mock_exists, mock_read_csv):
        mock_exists.return_value = True
        mock_read_csv.return_value = pd.DataFrame({
            'Title': ['Movie1', 'Movie2'],
            'Description': ['Desc1', 'Desc2'],
            'Cast': ['Cast1', 'Cast2'],
            'Info': ['Info1', 'Info2']
        })

        loader = DataLoader('./basedatos/IMDB top 1000.csv')
        df = loader.load_data()

        self.assertEqual(len(df), 2)
        self.assertIn('newcol', df.columns)

    @patch('os.path.exists')
    def test_load_data_file_not_found(self, mock_exists):
        mock_exists.return_value = False
        loader = DataLoader('./basedatos/IMDB top 1000.csv')

        with self.assertRaises(FileNotFoundError):
            loader.load_data()

class TestEmbeddingModel(unittest.TestCase):
    @patch('main.SentenceTransformer')
    def test_generate_embeddings(self, mock_sentence_transformer):
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_sentence_transformer.return_value = mock_model

        model = EmbeddingModel()
        df = pd.DataFrame({'newcol': ['Sample text']})

        df_with_embeddings = model.generate_embeddings(df)

        self.assertIn('embeddings', df_with_embeddings.columns)
        self.assertEqual(len(df_with_embeddings['embeddings']), 1)

class TestSemanticSearch(unittest.TestCase):
    @patch('main.SentenceTransformer')
    def test_search(self, mock_sentence_transformer):
        mock_model = MagicMock()
        mock_model.encode.return_value = [0.5, 0.5, 0.5]
        mock_sentence_transformer.return_value = mock_model

        df = pd.DataFrame({
            'Title': ['Movie1', 'Movie2'],
            'Description': ['Desc1', 'Desc2'],
            'embeddings': [[0.5, 0.5, 0.5], [0.1, 0.1, 0.1]]
        })

        search_engine = SemanticSearch(mock_model, CosineSimilarityStrategy())
        results = search_engine.search(df, 'query')

        self.assertEqual(len(results), 2)
        self.assertIn('similarity', results.columns)

class TestUserInterface(unittest.TestCase):
    @patch('builtins.input', side_effect=['sample query', 'esc'])
    def test_get_user_query(self, mock_input):
        query = UserInterface.get_user_query()
        self.assertEqual(query, 'sample query')

if __name__ == '__main__':
    unittest.main()
