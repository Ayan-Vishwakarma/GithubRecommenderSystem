GitHubRecommendation System: 

This model recommends

1. Most similar repositories based on a given input repository for possible contribution.
2. Contributors given an input repository which have done work in similar or equal domain as of the repository.

This is end-to-end model,where following steps were performed to create this model:

1. Data Extraction: Given the list of users, a list of repositories is computed, with the last 30 repositories for each user is taken into account as it dictates the user's current interests. The content of each of repo is extracted from the text and the source files. For each user, a user behaviour matrix UB is calculated, where 
UB{ i , j } represents relationship of user i with the repository j. The following relationships are considered with the following scores:

a. Create -> 10

b. Fork -> 5

c. Star -> 3

2. Data PreProcessing : In this step,text and source tfidf vector is comptuted seperately, for each repo which is used to calculate similarity scores afterwords. Since, it was impossible to take into account of every possible words in every possible repo, hashing trick was used, where each string was hashed using a hash function in 10007 MOD space. So, a 10007 dimensional source and text tfidf vectors were computed for each repo. It was observed that the matrices being generated generally had only around 500 non-zero matrices and were consuming large space, therefore, sparse matrices were employed. While computing tfidf vectors manually, the general pipeline for preprocessing was employed, consisting for removing meaningless symbols and digits, stemming of digits, removal of words of length less than 3, and other general methods. 

3. Model Training: K-nearest neighbors method was used based on cosine similariy between the tfidf vectors. For improving speed, Ball-tree was used which works on euclidean distance and Spotify/Annoy was used as another model for implementing the model using Locality Sensitive Hashing.

RETRAIN MODELS:

Step 1: Fill the user.txt file in githubrecsys1 with the names of users desired by the system.

Step 2: RUN ExtractRepoNames.py
```python
python ExtractRepoNames.py --token GITHUB_TOKEN
```

Step 3: RUN ExtractRepoFeatures.py
Running this takes a lot of time and multithreading was not used as the original code was manually distributed to many kaggle CPU kernels.
bg and nd was manually set with many Github tokens to be used to extract the files as the GITHUB rate limit may get exceeded very easily. This step made the large scale implementation of Recommender system impossible and the system currently recommends options for any given user/repo from a limited set of 300 users only.
```python
python ExtractRepoFeatures.py --bg BEGINNING_INDEX -nd ENDING_INDEX --tokens GITHUB_TOKEN_1 GITHUB_TOKEN_2 ... GITHUB_TOKEN_N
```

Step 4: RUN MergeSegments.py
```python
python MergeSegments.py
```

Step 5: RUN ConstructUBMatrix.py 
```python
python ConstructUBMatrix.py
```

Step 6: Train models
```python
python TrainModel.py
```

RUN MODELS:
```python
python ModelInference.py --model LSH|KNN --user_name USER_NAME --repo_name REPO_NAME --token GITHUB_TOKEN --top_n NUMBER_OF_RECOMMENDATIONS
```
