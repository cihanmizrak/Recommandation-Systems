############################################
# User-Based Collaborative Filtering
#############################################

# Step 1: Data Preparation
# Step 2: Determining the Movies Watched by the User to Suggest
# Step 3: Accessing Data and Ids of Other Users Watching the Same Movies
# Step 4: Identification of Users with the Most Similar Behaviors to the User to be Suggested
# Step 5: Calculating the Weighted Average Recommendation Score


#############################################
# Step 1: Data Preparation
#############################################
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 50)
pd.set_option('display.expand_frame_repr', False)
movie = pd.read_csv('datasets/movie.csv')
rating = pd.read_csv('datasets/ratings.csv')

def create_user_movie_df(movie, rating):
    import pandas as pd
    df = movie.merge(rating, how = "left", on = "movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df(movie, rating)

random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)


#############################################
# Step 2: Determining the Movies Watched by the User to Suggest
#############################################
random_user
user_movie_df
random_user_df = user_movie_df[user_movie_df.index == random_user]

movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

user_movie_df.loc[user_movie_df.index == random_user,
                  user_movie_df.columns == "Silence of the Lambs, The (1991)"]


len(movies_watched)

#############################################
# Step 3: Accessing Data and Ids of Other Users Watching the Same Movies
#############################################

movies_watched_df = user_movie_df[movies_watched]

user_movie_count = movies_watched_df.T.notnull().sum()

user_movie_count = user_movie_count.reset_index()

user_movie_count.columns = ["userId", "movie_count"]

user_movie_count[user_movie_count["movie_count"] > 20].sort_values("movie_count", ascending=False)

user_movie_count[user_movie_count["movie_count"] == 33].count()


users_same_movies = user_movie_count[user_movie_count["movie_count"] > 20]["userId"]


# users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]
# perc = len(movies_watched) * 60 / 100

#############################################
# # Step 4: Identification of Users with the Most Similar Behaviors to the User to be Suggested
#############################################

# For this we will perform 3 steps:
# 1. We'll gather the data of Sinan and other users.
# 2. We'll create the correlation df.
# 3. We'll find the most similar finders. (Top Users)


final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                      random_user_df[movies_watched]])

corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()

corr_df = pd.DataFrame(corr_df, columns=["corr"])

corr_df.index.names = ['user_id_1', 'user_id_2']

corr_df = corr_df.reset_index()

top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][["user_id_2", "corr"]].reset_index(drop=True) # optional 

top_users = top_users.sort_values(by='corr', ascending=False)

top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')

top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]


#############################################
# Step 5: Calculating the Weighted Average Recommendation Score
#############################################

top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating'] 

top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})

recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})

recommendation_df = recommendation_df.reset_index()

recommendation_df[recommendation_df["weighted_rating"] > 3.5] # optional 

movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)

movies_to_be_recommend.merge(movie[["movieId", "title"]])