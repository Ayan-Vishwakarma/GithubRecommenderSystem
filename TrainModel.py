from sklearn.neighbors import NearestNeighbors
import numpy as np
from scipy.sparse import load_npz,csr_matrix
from sklearn.metrics.pairwise import cosine_distances,cosine_similarity
import pickle
import matplotlib.pyplot as plt
import itertools
from sklearn.decomposition import TruncatedSVD
import scipy.spatial.distance as ssd
from annoy import AnnoyIndex

if __name__ == "__main__":
    tfidf_source = csr_matrix(load_npz(os.path.dirname(__file__) + ["/" if os.path.dirname(__file__) else ""][0] + "githubrecsys1/tfidf_source.npz"))
    tfidf_text = csr_matrix(load_npz(os.path.dirname(__file__) + ["/" if os.path.dirname(__file__) else ""][0] + "githubrecsys1/tfidf_text.npz"))
    UB_matrix = load_npz(os.path.dirname(__file__) + ["/" if os.path.dirname(__file__) else ""][0] + "githubrecsys1/UB_matrix.npz")

    ind_usr = np.where(np.load(os.path.dirname(__file__) + ["/" if os.path.dirname(__file__) else ""][0] + "githubrecsys1/valid_users.npy"))[0]
    ind_repo = np.where(np.load(os.path.dirname(__file__) + ["/" if os.path.dirname(__file__) else ""][0] + "githubrecsys1/valid_repos.npy"))[0]

    UB_matrix = UB_matrix[ind_usr[:,np.newaxis],ind_repo]
    repo_hash = np.load(os.path.dirname(__file__) + ["/" if os.path.dirname(__file__) else ""][0] + "githubrecsys1/repo_hash.npy",allow_pickle = True)
    user_features = load_npz(os.path.dirname(__file__) + ["/" if os.path.dirname(__file__) else ""][0] + "githubrecsys1/Xevent.npz")

    with open(os.path.dirname(__file__) + ["/" if os.path.dirname(__file__) else ""][0] + "githubrecsys1/users.txt",'r') as f:
        usr = [x.strip() for x in f]
    usr = np.array(usr)[ind_usr]

    with open(os.path.dirname(__file__) + ["/" if os.path.dirname(__file__) else ""][0] + "githubrecsys1/repos.txt",'r') as f:
        txt = [x.strip() for x in f]
    txt = np.array(txt)[ind_repo]

    w = np.load(os.path.dirname(__file__) + ["/" if os.path.dirname(__file__) else ""][0] + "githubrecsys1/w.npy")
    b = np.load(os.path.dirname(__file__) + ["/" if os.path.dirname(__file__) else ""][0] + "githubrecsys1/b.npy")

    query_text = tfidf_text[100]
    query_source = tfidf_source[100]

    #____________________________________________________________________________
    #
    #                           Hyperparameters
    #
    #____________________________________________________________________________
    K = 200
    Pearson = False
    Lmbda = 0.35
    assert( 0<=Lmbda<=1),"Lambda should be in the range [0,1]"
    #____________________________________________________________________________

    if Pearson:
        tfidf_text = csr_matrix(tfidf_text - tfidf_text.mean(axis = 0))
        tfidf_source = csr_matrix(tfidf_source - tfidf_source.mean(axis = 0))


    model1 = NearestNeighbors(n_neighbors = K,metric = cosine_distances)
    model1.fit(tfidf_text)
    model2= NearestNeighbors(n_neighbors = K,metric = cosine_distances)
    model2.fit(tfidf_source)

    pickle.dump(model1,open(os.path.dirname(__file__) + ["/" if os.path.dirname(__file__) else ""][0] +"githubrecsys1/KNN_text.pkl","wb"))
    pickle.dump(model2,open(os.path.dirname(__file__) + ["/" if os.path.dirname(__file__) else ""][0] +"githubrecsys1/KNN_source.pkl","wb"))

    tfidf_source = tfidf_source.multiply(1/np.sqrt(tfidf_source.multiply(tfidf_source).sum(axis=1)))
    tfidf_text = tfidf_text.multiply(1/np.sqrt(tfidf_text.multiply(tfidf_text).sum(axis=1)))
    
    dec1 = TruncatedSVD(1024,algorithm = 'arpack')
    dec1.fit(tfidf_text)
    dec2 = TruncatedSVD(1024,algorithm = 'arpack')
    dec2.fit(tfidf_source)

    model1 = NearestNeighbors(n_neighbors = K,metric = ssd.cosine,algorithm = 'ball_tree')
    model1.fit(dec1.transform(tfidf_text))
    model2 = NearestNeighbors(n_neighbors = K,metric = ssd.cosine,algorithm = 'ball_tree')
    model2.fit(dec2.transform(tfidf_source))

    pickle.dump(dec1,open(os.path.dirname(__file__) + ["/" if os.path.dirname(__file__) else ""][0] + "githubrecsys1/dec_text.pkl","wb"))
    pickle.dump(dec2,open(os.path.dirname(__file__) + ["/" if os.path.dirname(__file__) else ""][0] + "githubrecsys1/dec_source.pkl","wb"))
    pickle.dump(model1,open(os.path.dirname(__file__) + ["/" if os.path.dirname(__file__) else ""][0] + "githubrecsys1/KNN_cmp_text.pkl","wb"))
    pickle.dump(model2,open(os.path.dirname(__file__) + ["/" if os.path.dirname(__file__) else ""][0] + "githubrecsys1/KNN_cmp_source.pkl","wb"))

    ################################ Hyperparameters ###################################################

    n_trees = 256  # Numbers of trees to be used in LSH Forest

    #__________________________________________________________________________________________________

    model1 = AnnoyIndex(1024,"angular")
    model2 = AnnoyIndex(1024,"angular")

    for ind,i in enumerate(dec1.transform(tfidf_text)):
        model1.add_item(ind,i)
    for ind,i in enumerate(dec2.transform(tfidf_source)):
        model2.add_item(ind,i)

    model1.build(n_trees)
    model2.build(n_trees)

    model1.save(os.path.dirname(__file__) + ["/" if os.path.dirname(__file__) else ""][0] + "githubrecsys1/LSH_Text.ann")
    model2.save(os.path.dirname(__file__) + ["/" if os.path.dirname(__file__) else ""][0] + "githubrecsys1/LSH_Source.ann")