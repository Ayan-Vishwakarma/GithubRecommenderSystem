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
