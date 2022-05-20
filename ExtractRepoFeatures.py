# Due to limit of GitHub api calls, only a subsample of repos will be explored at this time between the range (bg)th repo to (nd)th repo
# If in between the api limit is reached before the completion of reading from bg to nd, then the process will terminate 
# saving the obtained data in npz files with the index representing the range of values successfully read and the program will output the
# index in which the api call limit exceeded.

import re
import base64
import json
import requests
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from scipy.sparse import vstack,csr_matrix,save_npz

import argparse

def hash(text:str):
    h=0
    for ch in text:
        h = ( h*281  ^ ord(ch)*997) & 0xFFFFFFFF
    return h

source_extentions = ['py','ipynb','cpp','c','cfg','js','json','vue',"xml","java","sh","php","rb","ts"]
text = []
source = []

def github_read_file(username, repository_name, file_path):
    headers = {}
    global github_token
    if github_token:
        headers['Authorization'] = f"token {github_token}"
        
    url = f'https://api.github.com/repos/{username}/{repository_name}/contents/{file_path}'
    try:
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        data = r.json()
    except:
        return ""
    file_content = data['content']
    file_content_encoding = data.get('encoding')
    if file_content_encoding == 'base64':
        file_content = base64.b64decode(file_content).decode()
    return file_content


def get_files(username,repository_name,file_path):
    file_content = github_read_file(username, repository_name, file_path)
    return file_content


def get_contents(username, repository_name):
    headers = {}
    global github_token
    if github_token:
        headers['Authorization'] = f"token {github_token}"
        
    url = f'https://api.github.com/repos/{username}/{repository_name}/contents'
    r = requests.get(url, headers=headers)
    try:
        r.raise_for_status()    
        data = r.json()
        return data
    except:
        return []

def recur(user,repo,r):
    headers = {}
    global github_token,text,source
    if github_token:
        headers['Authorization'] = f"token {github_token}"
    for i in r.json():
        if i["name"].startswith('.'):
            continue
        if i['type'] == 'file':
            if i['name'].split('.')[-1] == 'md' or i['name'].split('.')[-1] == 'txt':
                text.append(i)
            elif i['name'].split('.')[-1] in source_extentions :
                source.append(i)
        elif i['type'] == 'dir':
            url = i['url']
            try:
                r = requests.get(url, headers=headers)
                r.raise_for_status()
                recur(user,repo,r)
            except:
                pass
        
def get_filenames(user, repo):
    headers = {}
    global github_token,source,text,stop_procedure
    source = []
    text = []
    if github_token:
        headers['Authorization'] = f"token {github_token}"
    url = f'https://api.github.com/repos/{user}/{repo}/contents'
    try:
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        recur(user,repo,r)
    except:
        try:
            print("error @ get_filenames",user,repo)
            print(r.json())
            if r.json()['message'] not in  ['This repository is empty.' , "Repository access blocked" , "Not Found"]:
                stop_procedure = True
        except:
            print("Dual Error")
            print(r,r.json())
 
def extract_words(user,repo,text):
    puntuation = '1234567890-=!@#$%^&*()+[]{};:"\'|\\<>~/?`<>.,'
    out = ""
    ps = PorterStemmer()    
    for i in text:
        try:
            data = get_files(user,repo,i['path'])
            for i in puntuation:
                data = data.replace(i," ")
            data = data.replace("\n",' ')
            data = re.sub(r'\b\w{1,2}\b', '', data)
            data += ' '
            data = re.sub("\s\s+" , " ", data)
            if len(data)>0 and data[0]==' ':
                data = data[1:]
            data = data.lower()
            ps.stem(data)
            data = ' '.join([str(hash(i)%MOD) for i in data.split(' ')])
            out += data + ' '
        except:
            print("error reading",i["name"])
            pass
    return out

def get_tfidf_vectors_of(user_name,repo_name,token):
    global github_token,MOD
    github_token = token

    MOD = 10007
    Repo_text_hashed = []
    Repo_source_hashed = []

    get_filenames(user_name,repo_name)
    Repo_text_hashed.append(extract_words(user_name,repo_name,text))
    Repo_source_hashed.append(extract_words(user_name,repo_name,source))

    vocab = {}
    for i in range(MOD):
        vocab[str(i)] = i

    Repo_text_points = []
    Repo_source_points = []
    idf_count_text = []
    idf_count_source = []

    def get_mapper(voc):
        def mapper(key):
            if key not in voc:
                key = '0'
            return voc[key]
        return mapper

    mapper = get_mapper(vocab)

    for i in Repo_text_hashed:
        res = map(lambda x:mapper(x),i.split(' '))
        temp = np.bincount(list(res),minlength = MOD)
        if len(i.split(' '))>0:
            temp = temp / np.array([len(i.split(' '))],dtype = np.float64)
        Repo_text_points.append(csr_matrix(temp , dtype  = np.float64))

    for i in Repo_source_hashed:
        res = map(lambda x:mapper(x),i.split(' '))
        temp = np.bincount(list(res),minlength = MOD)
        if len(i.split(' '))>0:
            temp = temp / np.array([len(i.split(' '))],dtype = np.float64)
        Repo_source_points.append(csr_matrix(temp,dtype = np.float64))

    Repo_text_points = Repo_text_points
    Repo_source_points = Repo_source_points
    idf_text = np.load(os.path.dirname(__file__) + ["/" if os.path.dirname(__file__) else ""][0] + "githubrecsys1/idf_text.npy")
    idf_source = np.load(os.path.dirname(__file__) + ["/" if os.path.dirname(__file__) else ""][0] + "githubrecsys1/idf_source.npy")
    tfidf_text = Repo_text_points[0].multiply(idf_text)
    tfidf_source = Repo_source_points[0].multiply(idf_source)
    return tfidf_text,tfidf_source

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--bg",
                        default=None,
                        type=int,
                        required=True,
                        help="Beginning index.")
    parser.add_argument("--nd",
                        default=None,
                        type=int,
                        required=True,
                        help="Ending index.")
    parser.add_argument("--tokens",
                        default=[],
                        nargs = '+',
                        required=True,
                        help="List of Github Tokens to use Github API.")
    args = parser.parse_args()

    bg = args.bg
    nd = args.nd
    
    github_tokens = args.tokens
    ptr = 0
    github_token = github_tokens[ptr]
    MOD = 10007
    MOD_space = []
    for i in range(MOD):
        MOD_space.append(str(i))
    MOD_space = ' '.join(MOD_space)

    stop_procedure = False
    stop_index = bg - 1

    with open(os.path.dirname(__file__) + ['/' if os.path.dirname(__file__) else ""][0] + "githubrecsys1/repos.txt","r") as f:
        out = [l.strip() for l in f]
    ind = {}

    for i in range(len(out)):
        ind[out[i]] = i

    Repo_text_hashed = []
    Repo_source_hashed = []
    for i in out[bg:nd]:
        print(i)
        stop_index+=1
        ptr+=1
        ptr %= len(github_tokens)
        github_token = github_tokens[ptr]
        get_filenames(*i.split("/"))
        if stop_procedure:
            break
        Repo_text_hashed.append(extract_words(*i.split("/"),text))
        Repo_source_hashed.append(extract_words(*i.split("/"),source))

    if not stop_procedure:
        stop_index+=1

    vocab = {}
    for i in range(MOD):
        vocab[str(i)] = i

    Repo_text_points = []
    Repo_source_points = []
    idf_count_text = []
    idf_count_source = []

    def get_mapper(voc):
        def mapper(key):
            if key not in voc:
                key = '0'
            return voc[key]
        return mapper

    mapper = get_mapper(vocab)
    for i in Repo_text_hashed:
        res = map(lambda x:mapper(x),i.split(' '))
        temp = np.bincount(list(res),minlength = MOD)
        if len(i.split(' '))>0:
            temp = temp / np.array([len(i.split(' '))],dtype = np.float64)
        Repo_text_points.append(csr_matrix(temp , dtype  = np.float64))
        temp = np.zeros(MOD)
        for j in set(i.split(' ')):
            temp[mapper(j)] = 1
        idf_count_text.append(csr_matrix(temp,dtype = np.float64))

    for i in Repo_source_hashed:
        res = map(lambda x:mapper(x),i.split(' '))
        temp = np.bincount(list(res),minlength = MOD)
        if len(i.split(' '))>0:
            temp = temp / np.array([len(i.split(' '))],dtype = np.float64)
        Repo_source_points.append(csr_matrix(temp,dtype = np.float64))
        temp = np.zeros(MOD)
        for j in set(i.split(' ')):
            temp[mapper(j)] = 1
        idf_count_source.append(csr_matrix(temp,dtype = np.float64))

    del Repo_source_hashed,Repo_text_hashed
    Repo_text_points = vstack(Repo_text_points)
    Repo_source_points = vstack(Repo_source_points)
    idf_count_source = vstack(idf_count_source)
    idf_count_text = vstack(idf_count_text)

    print(stop_index)
    stop_index -= 1
    save_npz(os.path.dirname(__file__) + ['/' if os.path.dirname(__file__) else ""][0] + "temp/Repo_TextFeature_{}:{}".format(bg,stop_index),Repo_text_points)
    save_npz(os.path.dirname(__file__) + ['/' if os.path.dirname(__file__) else ""][0] + "temp/Repo_SourceFeature_{}:{}".format(bg,stop_index),Repo_source_points)
    save_npz(os.path.dirname(__file__) + ['/' if os.path.dirname(__file__) else ""][0] + "temp/Repo_IDF_count_text_{}:{}".format(bg,stop_index),idf_count_text)
    save_npz(os.path.dirname(__file__) + ['/' if os.path.dirname(__file__) else ""][0] + "temp/Repo_IDF_count_source_{}:{}".format(bg,stop_index),idf_count_source)