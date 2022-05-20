import requests
import numpy as np
from scipy.sparse import csr_matrix,save_npz,lil_matrix
import json
import argparse
    
def readfile(path):
    user_list = [line.rstrip() for line in open(path)]   #Change path in place of sample.txt
    return user_list

def get_starred_repos(username):
    starred_repos = []
    headers = {}
    global github_token
    if github_token:
        headers['Authorization'] = f"token {github_token}"
    request = requests.get('https://api.github.com/users/'+username+'/starred',headers = headers)
    r = request.json()
    for i in r:
        if i["full_name"] in repoind:
            starred_repos.append(repoind[i['full_name']])
    starred_repos = sorted(starred_repos)
    return starred_repos

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--token",
                        default=None,
                        type=str,
                        required=True,
                        help="Github token to access the Github API")
    args = parser.parse_args()

    github_token = args.token

    user = readfile(os.path.dirname(__file__) + ['/' if os.path.dirname(__file__) else ""][0] + 'githubrecsys1/users.txt')
    repo = readfile(os.path.dirname(__file__) + ['/' if os.path.dirname(__file__) else ""][0] + 'githubrecsys1/repos.txt')

    mat = lil_matrix(np.zeros((len(user),len(repo))))

    hashmap = {}
    for i in range(len(user)):
        hashmap[user[i]] = i

    repoind = {}
    for i in range(len(repo)):
        repoind[repo[i]] = i

    out = []
    for i in user:
        out.append(get_starred_repos(i))

    for i in range(len(out)):
        for j in out[i]:
            mat[i,j] = 2

    for i in range(len(repo)):
        headers = {}
        if github_token:
            headers['Authorization'] = f"token {github_token}"
        r = requests.get('https://api.github.com/repos/'+repo[0],headers = headers)
        r.raise_for_status()
        print(r['name'],r['fork'])
        if r.json()['fork'] == False:
            mat[hashmap[repo[i].split('/')[0]],i] = 10
        else :
            mat[hashmap[repo[i].split('/')[0]],i] = 5

    save_npz(os.path.dirname(__file__) + ['/' if os.path.dirname(__file__) else ""][0] + "githubrecsys1/UB_matrix",csr_matrix(mat))