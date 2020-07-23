#Data preprocessing 
#using the data set 'ml-latest-small' which descrives 5-star ratings from MovieLens
#we need all the movies and ratings in a n x m  matrix, n = number of users , m = number of movies  
#n = 610 
#m = 9742 

# X if the user has not rated that movie 
import pandas as pd
import numpy as np
from math import sqrt 

movies_df = pd.read_csv("movies.csv")
ratings_df=pd.read_csv("ratings.csv")
ratings_df.drop('timestamp',axis=1,inplace=True)
movies_df['new_movieId']=movies_df.index
users= ratings_df[ratings_df['movieId'].isin(movies_df['movieId'].tolist())]
users['movieId'] = users['movieId'].map(movies_df.set_index('movieId')['new_movieId'])
users_movie=users.groupby('userId')['movieId'].apply(list).tolist()
users_rating=users.groupby('userId')['rating'].apply(list).tolist()

#building the user-item matrix
#the goal is to predict the blanks-'X' in this user-item matrix by matrix factorization  
def make_matrix(users_rating,users_movie,row):
    temp=[]
    for i in range(9742):
        temp.append('X')
   
    k=0
    for user in range(len(users_rating[row])):
        temp[int(users_movie[row][k])] = users_rating[row][user]
        #temp.insert(users_movie[row][k],users_rating[row][user])
        k+=1
    return temp

big_matrix = []
for i in range(610):
    small_matrix = make_matrix(users_rating,users_movie,i)
    big_matrix.append(small_matrix)

X = big_matrix
#Stochastic gradient descent 


def SGD(X,f):
    n = len(X)
    m = len(X[0])
    #randomly full P and Q with values from a normal distribution 
    p= np.random.uniform(0,1,(n,f))
    q = np.random.uniform(0,1,(m,f))
    
    %run this f times 
    for _ in range(f):
        for u in range(n):
            for i in range(m):
                if X[u][i] != "X":
                    err = float(X[u][i]) - np.dot(p[u], q[i])
                # Update vectors p_u and q_i
                #used an alpha value of 0.02 
                    p[u] += 0.02 * err * q[i]
                    q[i] += 0.02 * err * p[u]
                    
    return p,q

#the larger the f, the more accurate the prediction
p , q = SGD(X,200)

#use the dot product to estimate values 
print(np.dot(p[0],q[43]))
print(X[0][43])
#values are almost the same so it seems like a good prediction 
#In order to avoid overfitting multiple decompositions can be found and when
#making predictions the average of the results of the decompositions can be taken 



