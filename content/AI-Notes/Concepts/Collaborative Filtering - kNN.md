
Files: lec6-ml.pdf
Tags: EECS 445

**kNN vs k-Means:**

- KNN represents a **supervised** classification algorithm that will give new data points accordingly to the k number or the closest data points
- while k-means clustering is an **unsupervised** clustering algorithm that gathers and groups data into k number of clusters.

# Nearest Neighbor Prediction

This is a much more heuristic approach than matrix factorization. You don’t need to bother taking the gradients or doing any minimization.

Suppose user $a$ has not rating movie $i$, to predict the rating

- Compute similarity between user $a$ and all other users in the system
- Find the $k$ ‘nearest neighbors’ of user $a$ who have rated movie $i$
    - The $k$ most similar users to user $a$.
- Compute a prediction based on these user’s ratings of $i$

## How to choose k

- Some heuristics
    - All users are considered as neighbors
    - Limiting neighborhood size can result in more accurate predictions
        - Neighbors with low correlation tend to introduce noise
    - Varying neighborhoods are selected for each item based on a similarity threshold
    - Offline analysis: values in the 20-5 range are reasonable starting points in many domains.

## Computing Similarity

Some of the entries are empty, so you need a way to compare users based on similar movies they’ve seen. Can use **Pearson’s correlation coefficient.** These values lie between [-1, 1].

- 1 means the users are opposite (if one user rates a movie highly, the other will rate lowly)
- 0 means there is no real relationship
- 1 means the users agrees
- Can use the absolute value of the Pearson’s correlation to find which users are most correlated.

![https://paper-attachments.dropbox.com/s_DB73BB6D672F639AA1E5312FADD76ADF4FA9BF7DFAE0A6FD8D9CB51FFD17C3A5_1585149666378_Screen+Shot+2020-03-25+at+11.21.04+AM.png](https://paper-attachments.dropbox.com/s_DB73BB6D672F639AA1E5312FADD76ADF4FA9BF7DFAE0A6FD8D9CB51FFD17C3A5_1585149666378_Screen+Shot+2020-03-25+at+11.21.04+AM.png)

## Notation

$R(a,b)$: the set of items rated by both users a and b

|R(a,b)|: the size of set R(a,b) - aka the cardinality

We write the **average rating** of user a for items rated by both users a and b as follows. Note this is just averaging all user a’s ratings for movies that were reviewed by both a and b.

![https://paper-attachments.dropbox.com/s_DB73BB6D672F639AA1E5312FADD76ADF4FA9BF7DFAE0A6FD8D9CB51FFD17C3A5_1585338431218_Screen+Shot+2020-03-27+at+3.47.05+PM.png](https://paper-attachments.dropbox.com/s_DB73BB6D672F639AA1E5312FADD76ADF4FA9BF7DFAE0A6FD8D9CB51FFD17C3A5_1585338431218_Screen+Shot+2020-03-27+at+3.47.05+PM.png)

We denote the **similarity between two users** a and b as:

![https://paper-attachments.dropbox.com/s_DB73BB6D672F639AA1E5312FADD76ADF4FA9BF7DFAE0A6FD8D9CB51FFD17C3A5_1585338135426_Screen+Shot+2020-03-27+at+3.42.10+PM.png](https://paper-attachments.dropbox.com/s_DB73BB6D672F639AA1E5312FADD76ADF4FA9BF7DFAE0A6FD8D9CB51FFD17C3A5_1585338135426_Screen+Shot+2020-03-27+at+3.42.10+PM.png)

![https://paper-attachments.dropbox.com/s_DB73BB6D672F639AA1E5312FADD76ADF4FA9BF7DFAE0A6FD8D9CB51FFD17C3A5_1585338530519_Screen+Shot+2020-03-27+at+3.48.46+PM.png](https://paper-attachments.dropbox.com/s_DB73BB6D672F639AA1E5312FADD76ADF4FA9BF7DFAE0A6FD8D9CB51FFD17C3A5_1585338530519_Screen+Shot+2020-03-27+at+3.48.46+PM.png)

![https://paper-attachments.dropbox.com/s_DB73BB6D672F639AA1E5312FADD76ADF4FA9BF7DFAE0A6FD8D9CB51FFD17C3A5_1585338565497_Screen+Shot+2020-03-27+at+3.48.58+PM.png](https://paper-attachments.dropbox.com/s_DB73BB6D672F639AA1E5312FADD76ADF4FA9BF7DFAE0A6FD8D9CB51FFD17C3A5_1585338565497_Screen+Shot+2020-03-27+at+3.48.58+PM.png)

## Prediction

Look at a weighted average of the similarity between a and b.

![https://paper-attachments.dropbox.com/s_DB73BB6D672F639AA1E5312FADD76ADF4FA9BF7DFAE0A6FD8D9CB51FFD17C3A5_1585338662861_Screen+Shot+2020-03-27+at+3.50.59+PM.png](https://paper-attachments.dropbox.com/s_DB73BB6D672F639AA1E5312FADD76ADF4FA9BF7DFAE0A6FD8D9CB51FFD17C3A5_1585338662861_Screen+Shot+2020-03-27+at+3.50.59+PM.png)

Weight each user b’s rating by their similarity to a. You might normalize by how user’s rate their movies (in case someone rates on extremes vs. someone rates around 2-4).

## Example

![https://paper-attachments.dropbox.com/s_DB73BB6D672F639AA1E5312FADD76ADF4FA9BF7DFAE0A6FD8D9CB51FFD17C3A5_1585338626578_Screen+Shot+2020-03-27+at+3.50.23+PM.png](https://paper-attachments.dropbox.com/s_DB73BB6D672F639AA1E5312FADD76ADF4FA9BF7DFAE0A6FD8D9CB51FFD17C3A5_1585338626578_Screen+Shot+2020-03-27+at+3.50.23+PM.png)

![https://paper-attachments.dropbox.com/s_DB73BB6D672F639AA1E5312FADD76ADF4FA9BF7DFAE0A6FD8D9CB51FFD17C3A5_1585338641233_Screen+Shot+2020-03-27+at+3.50.38+PM.png](https://paper-attachments.dropbox.com/s_DB73BB6D672F639AA1E5312FADD76ADF4FA9BF7DFAE0A6FD8D9CB51FFD17C3A5_1585338641233_Screen+Shot+2020-03-27+at+3.50.38+PM.png)

![https://paper-attachments.dropbox.com/s_DB73BB6D672F639AA1E5312FADD76ADF4FA9BF7DFAE0A6FD8D9CB51FFD17C3A5_1585338799486_Screen+Shot+2020-03-27+at+3.53.16+PM.png](https://paper-attachments.dropbox.com/s_DB73BB6D672F639AA1E5312FADD76ADF4FA9BF7DFAE0A6FD8D9CB51FFD17C3A5_1585338799486_Screen+Shot+2020-03-27+at+3.53.16+PM.png)

## User-item or item-item

You can recommend a certain product to a user A in two main ways:

- User-item: you look at similar users to user A and see what they liked. You then recommend these products.
- Item-item: you look at products user A liked and find similar products. You recommend these products.

When choosing which to use, you want to figure out which provides the denser matrix. If a lot of users have rated items, go with user-item. However, if most users have few reviews, go with item-item.