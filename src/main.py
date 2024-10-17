import pandas as pd
import os
from sentence_transformers import SentenceTransformer, util
from abc import ABC, abstractmethod

# **Principio DIP: Creación de una abstracción para la estrategia de similitud de embeddings**
class SimilarityStrategy(ABC):
    @abstractmethod
    def compute_similarity(self, embedding, query_embedding):
        pass

# **Aplicación del Patrón Strategy: Estrategia para calcular la similitud utilizando cos_sim**
class CosineSimilarityStrategy(SimilarityStrategy):
    def compute_similarity(self, embedding, query_embedding):
        return util.cos_sim(embedding, query_embedding).item()

# **Principio SRP: Clase para cargar y preparar los datos**
class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
    
    def load_data(self):
        """Carga el archivo CSV y lo prepara eliminando duplicados y creando la columna de búsqueda."""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"El archivo '{self.filepath}' no fue encontrado.")
        
        df = pd.read_csv(self.filepath)
        df = df.drop_duplicates(subset=df.columns[1])  # Elimina duplicados basados en una columna específica
        #df['Busqueda'] = 'Titulo: ' + df['Title'] + ' | ' + 'Description: ' + df['Description'] + '|' + df['Cast'] + ' | ' + 'Genre :' + df['Genre']
        df['newcol']= 'Titulo: '+df['Title']+' | '+'Description: '+df['Description']+'|'+df['Cast']+' | '+df['Info']
        return df

# **Principio SRP: Clase para manejar el modelo y generar embeddings**
class EmbeddingModel:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def generate_embeddings(self, df, column='newcol'):
        """Genera embeddings a partir de una columna del DataFrame y los almacena."""
        embeddings = self.model.encode(df[column], batch_size=64, show_progress_bar=True)
        df['embeddings'] = embeddings.tolist()
        return df

# **Principio OCP: Clase que realiza la búsqueda semántica. Se puede extender sin modificar el código original**
class SemanticSearch:
    def __init__(self, model: SentenceTransformer, similarity_strategy: SimilarityStrategy):
        self.model = model
        self.similarity_strategy = similarity_strategy
    
    def search(self, df, query):
        """Realiza una búsqueda semántica utilizando la estrategia de similitud proporcionada."""
        query_embedding = self.model.encode(query)  # Generamos los embeddings para la consulta
        df['similarity'] = df.apply(lambda x: self.similarity_strategy.compute_similarity(x['embeddings'], query_embedding), axis=1)
        df = df.sort_values(by='similarity', ascending=False)
        return df[['Title', 'Description', 'similarity']].head(10)

# **Principio SRP: Clase para manejar la interfaz de usuario**
class UserInterface:
    @staticmethod
    def get_user_query():
        """Obtiene la consulta del usuario."""
        return input('\nDigite clave busqueda o escriba ESC para salir ')
    
    @staticmethod
    def display_results(results):
        """Muestra los resultados de la búsqueda."""
        print('Este el top 10 de peliculas relacionadas: \n')
        print(results)

# **Función principal utilizando las clases diseñadas**
def main():
    data_loader = DataLoader('./basedatos/IMDB top 1000.csv')
    embedding_model = EmbeddingModel()  # Instanciamos el modelo
    
    try:
        # Cargar y preparar los datos
        df = data_loader.load_data()
        df = embedding_model.generate_embeddings(df)
        
        # Instanciar la búsqueda semántica con la estrategia de similitud (Patrón Strategy)
        search_engine = SemanticSearch(embedding_model.model, CosineSimilarityStrategy())  # Pasamos el modelo y la estrategia
        
        # Interfaz de usuario para obtener las consultas y mostrar resultados
        while True:
            query = UserInterface.get_user_query()
            if query.lower() == 'esc':
                print("Programa finalizado.")
                break
            
            results = search_engine.search(df, query)
            UserInterface.display_results(results)
    
    except FileNotFoundError as e:
        print(e)
    except KeyError as e:
        print(f"Error: Una columna requerida no fue encontrada en el archivo CSV. {e}")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")

if __name__ == '__main__':
    main()
