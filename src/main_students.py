import pandas as pd
from sentence_transformers import SentenceTransformer, util

    #print(df.head())
    # TODO: Completar esta función para realizar búsquedas semánticas con base en el código del archivo test.ipynb
    # Se borran los Duplicados
def modelo():
    df = pd.read_csv('./DATASET/IMDB top 1000.csv')    
    df = df.drop_duplicates(df.columns[1])
    # Se crea una columna Busqueda y se concatena con Descripcio, Cast y Genre
    df['newcol']= 'Titulo: '+df['Title']+' | '+'Description: '+df['Description']+'|'+df['Cast']+' | '+df['Info']

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    embeddings = model.encode(df['newcol'],batch_size=64,show_progress_bar=True)

    df['embeddings'] = embeddings.tolist()
    return df

def compute_similarity(example, query_embedding):
    embedding = example['embeddings']
    similarity = util.cos_sim(embedding, query_embedding).item()
    return similarity

def main(query,df):
  
    try:
        
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        query_embedding = model.encode(query)
        df['similarity'] = df.apply(lambda x: compute_similarity(x, query_embedding), axis=1)
        df = df.sort_values(by='similarity', ascending=False)
        print(df[['Title','Description','similarity']].head())
        # Aquí puedes añadir más lógica según lo que necesites hacer con el número
    except ValueError:
        print("Digite clave busqueda o escriba esc para salir'.")

if __name__ == '__main__':
    
    df = modelo()
    while True:
     query = input('Digite clave busqueda o escriba esc para salir ')

     if query.lower() == 'esc':  # Compara la entrada con 'salir' (sin importar mayúsculas)
        print("Programa finalizado.")
        break
     else:   
        main(query,df)