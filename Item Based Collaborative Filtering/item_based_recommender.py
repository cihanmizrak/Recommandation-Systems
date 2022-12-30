###########################################
# Item-Based Collaborative Filtering
###########################################

# Data = https://grouplens.org/datasets/movielens/


######################################
# Step 1: Dataset Preparing
######################################
import pandas as pd
pd.set_option('display.max_columns', 50)
movie = pd.read_csv('datasets\movies.csv')
rating = pd.read_csv('datasets\ratings.csv')
df = movie.merge(rating, how="left", on="movieId")
df.head()

######################################
# Step 2: Creating User Movie Df
######################################

df["title"].nunique()

df["title"].value_counts().head()

comment_counts = pd.DataFrame(df["title"].value_counts())
rare_movies = comment_counts[comment_counts["title"] <= 1000].index #titles
common_movies = df[~df["title"].isin(rare_movies)]
common_movies.shape #(22138587, 6)
common_movies["title"].nunique() #3790
df["title"].nunique() #62325

user_movie_df = common_movies.pivot_table(index=["userId"], columns = ["title"], values = "rating") #ratings table

######################################
# Step 3: Doing recommend movies with Item-Based 
######################################

movie_name = "Matrix, The (1999)" 
movie_name = user_movie_df[movie_name]
#corrwith = https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corrwith.html
user_movie_df.corrwith(movie_name).sort_values(ascending = False).head(10) # The movies that which movies matched your selected movie. 


#another method to get the movies

def check_film(keyword, user_movie_df):
    print(f"=={keyword}==")
    return [col for col in user_movie_df.columns if keyword in col]

movie_name = pd.Series(user_movie_df.columns).sample(1).values[0] #selecting a random movie name and showing recommendations

check_film(movie_name, user_movie_df) #recommendation movies
