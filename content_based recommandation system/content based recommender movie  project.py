import pandas as pd
pd.set_option("display.max_columns", None)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings 
warnings.filterwarnings("ignore")
moviedata = pd.read_csv("datasets/movie_metadata.csv")
#https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset
mdatadf = moviedata.copy()

# =============================================================================
# Movie searching function
# =============================================================================

def movies(df, all_movie = False, **contain):
    df["title"] = df["title"].fillna("") 
    
    if all_movie:
        movies = df["title"]
        print(movies)
    
    else:
        movies=df[df['title'].str.contains(f"{contain['intitle']}", case=False)]["title"]
        print(movies)
        
    return movies

movies(mdatadf, intitle = "hateful")
#or
movies(mdatadf,all_movie=True)



def content_based(df,movie):
    # =============================================================================
    # TF-IDF matrix
    # =============================================================================
    df = df[:10000] #optional your  
    tfidf = TfidfVectorizer(stop_words="english")
    df["overview"] = df["overview"].fillna("") 
    tfidf_matrix = tfidf.fit_transform(df["overview"])
    matrix = tfidf_matrix.toarray()
    
    # =============================================================================
    # cosine similiarity & recommendation
    # =============================================================================
    
    cosine_sim = cosine_similarity(matrix,matrix)
    indices = pd.Series(df.index, index = df["title"])
    indices = indices[~indices.index.duplicated(keep="last")]
    movie_index = indices[movie]
    similarity_scores = pd.DataFrame(cosine_sim[movie_index],columns = ["score"])
    movie_indices = similarity_scores.sort_values("score",ascending = False)[1:11].index #optional
    recommended_movies = df["title"].loc[movie_indices]
    print(f"=={movie}==\n\n{recommended_movies}")
    return recommended_movies

content_based(mdatadf,"The Hateful Eight")



