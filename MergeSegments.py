from glob import glob
import numpy as np
from scipy.sparse import load_npz,vstack,save_npz,csr_matrix

if __name__ == "__main__":
    txt = []
    usr = []

    with open(os.path.dirname(__file__) + ['/' if os.path.dirname(__file__) else ""][0] + "githubrecsys1/repos.txt") as f:
        txt = [x.strip() for x in f.readlines()]
    with open(os.path.dirname(__file__) + ['/' if os.path.dirname(__file__) else ""][0] + "githubrecsys1/users.txt") as f:
        usr = [x.strip() for x in f.readlines()]

    txt_ind = np.zeros(len(txt),dtype = bool)
    usr_ind = np.zeros(len(usr),dtype = bool)

    hash_map = {}
    for i in range(len(usr)):
        hash_map[usr[i]] = i

    src_c = []
    txt_c = []
    txt_f = []
    src_f = []

    for i in glob(os.path.dirname(__file__) + ['/' if os.path.dirname(__file__) else ""][0] + "temp/*"):
        if "count_source" in i:
            src_c.append(i)
        elif "count_text" in i:
            txt_c.append(i)
        elif "TextFeature" in i:
            txt_f.append(i)
        elif "SourceFeature" in i:
            src_f.append(i)

    for i in txt_c:
        temp = i.split('.')[-2].split('_')
        for i in range(int(temp[-2]),int(temp[-1])+1):
            assert(txt_ind[i] == False),"Intervals Should not overlap"
            txt_ind[i] = True
            usr_ind[hash_map[txt[i].split('/')[0]]] = True

    src_c = sorted([(int(x.split('.')[-2].split("_")[-2]),x) for x in src_c])
    txt_c = sorted([(int(x.split('.')[-2].split("_")[-2]),x) for x in txt_c])
    txt_f = sorted([(int(x.split('.')[-2].split("_")[-2]),x) for x in txt_f])
    src_f = sorted([(int(x.split('.')[-2].split("_")[-2]),x) for x in src_f])

    count_source = []
    count_text = []
    feature_source = []
    feature_text = []

    for i,j in src_c:
        count_source.append(load_npz(j))
    for i,j in txt_c:
        count_text.append(load_npz(j))
    for i,j in src_f:
        feature_source.append(load_npz(j))
    for i,j in txt_f:
        feature_text.append(load_npz(j))

    count_source = vstack(count_source)
    count_text = vstack(count_text)
    feature_source = vstack(feature_source)
    feature_text = vstack(feature_text)

    try:
        assert( count_source.shape == count_text.shape == feature_source.shape == feature_text.shape)
    except:
        print("count_source_shape:",count_source.shape)
        print("count_text_shape:",count_text.shape)
        print("feature_source_shape:",feature_source.shape)
        print("feature_text_shape:",feature_text.shape)
    else:
        print(count_source.shape)

    idf_source = np.log(count_source.shape[0] / (count_source.sum(axis = 0) + 1))
    idf_text = np.log(count_text.shape[0] / (count_text.sum(axis = 0) + 1))
    tfidf_text = feature_text.multiply(idf_text)
    tfidf_source = feature_source.multiply(idf_source)

    np.save(os.path.dirname(__file__) + ['/' if os.path.dirname(__file__) else ""][0] + "githubrecsys1/valid_repos",txt_ind)
    np.save(os.path.dirname(__file__) + ['/' if os.path.dirname(__file__) else ""][0] + "githubrecsys1/valid_users",usr_ind)
    np.save(os.path.dirname(__file__) + ['/' if os.path.dirname(__file__) else ""][0] + "githubrecsys1/idf_source",idf_source)
    np.save(os.path.dirname(__file__) + ['/' if os.path.dirname(__file__) else ""][0] + "githubrecsys1/idf_text",idf_text)
    save_npz(os.path.dirname(__file__) + ['/' if os.path.dirname(__file__) else ""][0] + "githubrecsys1/tfidf_text",tfidf_text)
    save_npz(os.path.dirname(__file__) + ['/' if os.path.dirname(__file__) else ""][0] + "githubrecsys1/tfidf_source",tfidf_source)