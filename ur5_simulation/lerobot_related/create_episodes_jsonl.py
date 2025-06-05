import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import pyarrow as pa
import pyarrow.parquet as pq
import json

mypath = os.environ["HOME"] + '/Project/Aveiro_project/lerobot/xarm_lift_data/data/chunk-000'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
onlyfiles.sort()
#print(onlyfiles)

#'''
jsonl_data = []

for file in onlyfiles:
    df = pd.read_parquet(mypath + '/' + file)
    print(f"file:{file} first index:{df['index'][0]}, last index:{df['index'][len(df)-1]}")
    episode_dic = {}
    episode_dic['episode_index'] = int(df['episode_index'][0])
    episode_dic['tasks'] = ["Lift the nlock above a height"]
    episode_dic['length'] = len(df)
    jsonl_data.append(episode_dic)
#'''

#with open('fuga.jsonl', 'w') as f:
#    f.writelines([json.dumps(l) for l in jsonl_data])

with open('episodes.jsonl', 'w') as f:
    for l in jsonl_data:
        f.writelines([json.dumps(l)])
        f.writelines("\n")

#json_string = json.dumps(episode_dic)

#print(json_string)

#for i in range(len(onlyfiles)-1):
    
