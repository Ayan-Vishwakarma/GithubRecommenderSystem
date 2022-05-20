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
from ExtractRepoFeatures import *
import argparse
import os

with open(os.path.dirname(__file__) + ["/" if os.path.dirname(__file__) else ""][0] + 'githubrecsys1/dec_text.pkl', 'rb') as f:
    dec1 = pickle.load(f)
with open(os.path.dirname(__file__) + ["/" if os.path.dirname(__file__) else ""][0] + 'githubrecsys1/dec_source.pkl', 'rb') as f:
    dec2 = pickle.load(f)

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

Lmbda = 0.35
K = 200

with open(os.path.dirname(__file__) + ["/" if os.path.dirname(__file__) else ""][0] + 'githubrecsys1/KNN_cmp_text.pkl', 'rb') as f:
    model1knn = pickle.load(f)
with open(os.path.dirname(__file__) + ["/" if os.path.dirname(__file__) else ""][0] + 'githubrecsys1/KNN_cmp_source.pkl', 'rb') as f:
    model2knn = pickle.load(f)

model1ann = AnnoyIndex(1024,"angular")
model2ann = AnnoyIndex(1024,"angular")
model1ann.load(os.path.dirname(__file__) + ["/" if os.path.dirname(__file__) else ""][0] + 'githubrecsys1/LSH_Text.ann')
model2ann.load(os.path.dirname(__file__) + ["/" if os.path.dirname(__file__) else ""][0] + 'githubrecsys1/LSH_Source.ann')

def get_usr_and_repo_scores(query_text,query_source,model_name):
    if model_name.lower() == "lsh":
        s1 = np.array(model1ann.get_nns_by_vector(dec1.transform(query_text)[0],K))
        s2 = np.array(model2ann.get_nns_by_vector(dec2.transform(query_source)[0],K))
        r_inds = sorted(list(set(s1.ravel()).union(set(s2.ravel()))))
        u_inds = sorted(list(set(itertools.chain.from_iterable(repo_hash[r_inds]))))

        sim = cosine_similarity(csr_matrix(tfidf_text)[r_inds],query_text) * Lmbda + \
              (1 - Lmbda) * cosine_similarity(csr_matrix(tfidf_source)[r_inds],query_source)
        usr_score = UB_matrix[np.array(u_inds)[:,np.newaxis],r_inds] * sim
        usr_score = sorted([(x[0],y) for x,y in zip(usr_score,usr[u_inds])],reverse = True)
        repo_score = (sorted([(x[0],y) for x,y in zip(sim,txt[r_inds])],reverse = True))
        return usr_score,repo_score
    else:
        s1 = model1knn.kneighbors(dec1.transform(query_text),return_distance = False)
        s2 = model2knn.kneighbors(dec2.transform(query_source),return_distance = False)

        r_inds = sorted(list(set(s1.ravel()).union(set(s2.ravel()))))
        u_inds = sorted(list(set(itertools.chain.from_iterable(repo_hash[r_inds]))))

        sim = cosine_similarity(csr_matrix(tfidf_text)[r_inds],query_text) * Lmbda + \
                (1 - Lmbda) * cosine_similarity(csr_matrix(tfidf_source)[r_inds],query_source)
        usr_score = UB_matrix[np.array(u_inds)[:,np.newaxis],r_inds] * sim 
        usr_score = sorted([(x[0],y) for x,y in zip(usr_score,usr[u_inds])],reverse = True)
        repo_score = (sorted([(x[0],y) for x,y in zip(sim,txt[r_inds])],reverse = True))
        return usr_score,repo_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        default=None,
                        type=str,
                        required=True,
                        help="Model to be used. [ LSH | KNN ]")
    parser.add_argument("--user_name",
                        default=None,
                        type=str,
                        required=True,
                        help="User Name")
    parser.add_argument("--repo_name",
                        default=None,
                        type=str,
                        required=True,
                        help="User's repo")
    parser.add_argument("--token",
                        default=None,
                        type=str,
                        required=True,
                        help="Github Token")
    parser.add_argument("--top_n",
                        default=10,
                        type=int,
                        required=False,
                        help="Top n Most similar items to print")
    args = parser.parse_args()
    model_name = args.model
    query_text,query_source = get_tfidf_vectors_of(args.user_name,args.repo_name,args.token)
    
    query_source = query_source.multiply(1/np.sqrt(query_source.multiply(query_source).sum(axis=1)))
    query_text = query_text.multiply(1/np.sqrt(query_text.multiply(query_text).sum(axis=1)))

    usr_score,repo_score = get_usr_and_repo_scores(query_text,query_source,model_name)    
    print("User Scores")
    for i in usr_score[:args.top_n]:
        print(i[0],i[1])
    print()
    print("Repo Scores")
    for i in repo_score[:args.top_n]:
        print(i[0],i[1])