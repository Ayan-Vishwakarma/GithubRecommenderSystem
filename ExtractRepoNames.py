import re
import base64
import json
import requests
import os
import numpy as np
import argparse
import time

def get_repos(path):
    with open(path,'r') as f:
        save = [x.strip() for x in f.readlines()]
    headers = {}
    global github_token
    if github_token:
        headers['Authorization'] = f"token {github_token}"
    l = []
    ec = 0
    Error_Limit = 30
    for i in save:
        print(i)
        time.sleep(0.4)
        try:
            r = requests.get("https://api.github.com/users/{}/repos".format(i),headers = headers)
            r.raise_for_status()
            for e in r.json():
                if e['full_name'] not in l:
                    l.append(e['full_name'])
        except:
            ec+=1
            print("error @ ",i)
        if ec>Error_Limit:
            break
    out = open("./repos.txt",'w')
    print(out.writelines([i+'\n' for i in l]))
    out.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token",
                        default=None,
                        type=str,
                        required=True,
                        help="Github token to access the Github API")
    args = parser.parse_args()
    github_token = args.token
    get_repos(os.path.dirname(__file__) + ['/' if os.path.dirname(__file__) else ''][0] + "githubrecsys1/users.txt")