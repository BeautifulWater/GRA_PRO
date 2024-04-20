import plotly.express as px
from flask import render_template,Flask, request, jsonify
from flask_cors import CORS
import pandas as pd

import json
import os
import copy
import torch
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

import csv

warnings.filterwarnings('ignore')

from scipy import stats   
import random

# from torch_geometric.utils import to_networkx
# from torch_geometric.data import Data, DataLoader
# import torch.nn.functional as F
# from torch.nn import Linear, Dropout
# from torch_geometric.nn import GCNConv, GATv2Conv

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import VotingClassifier
from sklearn.base import clone 

import xgboost as xgb


import pickle
from  vis_utils import  class_to_info,class_to_color

class Config:
    seed = 0
    learning_rate = 0.001
    weight_decay = 1e-5
    input_dim = 165
    output_dim = 1
    hidden_size = 128
    num_epochs = 100
    checkpoints_dir = './models/elliptic_gnn'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
print("Using device:", Config.device)

app = Flask(__name__)
CORS(app)  # 启用 CORS
nums = 50



#得到危险结点，用于page2
@app.route('/api/get_node_danger',methods=['POST','GET'])
def get_node_danger():

    df = pd.read_csv("Dataset/vis/nodes_features_vis.csv")
    ids = find_node_danger_id()
    rows = []
    dics = []
    
    print("get_node_danger:",ids)
    for id,row in df.iterrows():
        if(id in ids):
            print(row)
            rows.append(row)
            dic = {}
            for key,value in row.items():
                dic[key] = value
            dics.append(dic)
    file_path = "Dataset/vis/nodes_removed.csv"
    # 打开文件，以写入模式创建或覆盖

       
    with open(file_path, 'w', newline='') as csvfile:
        # 创建csv写入器对象
        writer = csv.writer(csvfile)
        # 写入数据行
        writer.writerow(df.columns)
        for row in rows:
            writer.writerow(row)
    return dics, 200, {'Content-Type': 'application/json'}


#取得相连的结点且不是危险结点
def find_connect(name):
    ST = set()
    df = pd.read_csv("Dataset/vis/nodes_links_vis.csv")
    df2 = pd.read_csv("Dataset/vis/nodes_features_vis.csv")
    name = int(name)

    for i in range(len(df["source"])):
        print(df['source'][i],df["target"][i])
        if df["source"][i] == name:
            print("source")
            ST.add(df["target"][i])
        if df["target"][i] == name:
            print("target")
            ST.add(df["source"][i])
    return ST

#找到危险结点的id
def find_node_danger_id():
    df = pd.read_csv("Dataset/vis/nodes_features_vis.csv")
    L = []
    for id,row in df.iterrows():   
        if row['kind'] == 1:
            print(row)
            L.append(id)
    return L

#得到危险结点，用于page2
@app.route('/api/get_node_in_danger',methods=['POST','GET'])
def find_node_in_danger():
    df = pd.read_csv("Dataset/vis/nodes_features_vis.csv")
    ids = find_node_danger_id()
    rows = []
    dics = []
    print("get_node_danger:",ids)
    ST = set()

    for id in ids:
        ST = ST.union(find_connect(id))
    for id,row in df.iterrows():
        if((id+1) in ST) and row["info"]=="Normal":
            print(row)
            rows.append(row)
            dic = {}
            for key,value in row.items():
                dic[key] = value
            dics.append(dic)
    file_path = "Dataset/vis/nodes_to_remind.csv"
    # 打开文件，以写入模式创建或覆盖
    with open(file_path, 'w', newline='') as csvfile:
        # 创建csv写入器对象
        writer = csv.writer(csvfile)
        # 写入数据行
        writer.writerow(df.columns)
        for row in rows:
            writer.writerow(row)
    return dics, 200, {'Content-Type': 'application/json'}

def check_danger():
    df = pd.read_csv("Dataset/vis/nodes_features_vis.csv")
    print(get_node_danger())
    rows = []
    for id,row in df.iterrows():
        print(row)
        rows.append(row)
    return rows



#删除结点
@app.route('/api/del_node',methods=['POST','GET'])
def del_node():
    node_name = request.args.get('name')
    ST = find_connect(node_name[node_name.index('_')+1:])
    print(ST)
    return node_name


#检测csv是否为空
def csv_file_has_non_empty_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return not df.empty
    except FileNotFoundError:
        return False
    except pd.errors.EmptyDataError:
        return False

#传输要使用的handle_post_mode
@app.route('/api/post_model',methods=['GET'])
def handle_post_model():
    # 从请求的查询参数中获取模型名称
    model = request.args.get('model')
    if model is None:
        return jsonify(error="Missing required parameter 'model'"), 400
    # 指定已保存的.pkl文件的路径

    input_file = "pkls/{}_model.pkl".format(model)   
    with open(input_file, 'rb') as f:
        loaded_data = pickle.load(f)
        df = pd.read_csv("Dataset/vis/nodes_features.csv")
        for id,row in df.iterrows():
            pre = loaded_data.predict(row[2:].values.reshape(1,-1))
            print(id,pre)

    with open(input_file, 'rb') as f:
        loaded_data = pickle.load(f)
        df = pd.read_csv("Dataset/vis/nodes_features.csv")
        pres = []
        for id,row in df.iterrows():
            print("row is",row)
            pre = loaded_data.predict(row[2:].values.reshape(1,-1))
            pres.append(pre)
        df2 = pd.read_csv("Dataset/vis/nodes_features_vis.csv")
        df2.to_csv("Dataset/vis/nodes_features_vis.csv")  
        print("df_check:",df2.loc[0,"kind"])
        for i in range(0,len(pres)):
            df2.loc[i, 'kind'] = pres[i]
            print(class_to_color(pres[i][0]))#{'color': '#808080'}
        #     # 使用 json.loads() 将字符串转换为字典
            df2.loc[i,'itemStyle'] = "{\"color\":"+ "\""+str(class_to_color(pres[i][0]))+"\"}"
            df2.loc[i,'info'] = class_to_info(pres[i][0])
            print(df2)
        df2.to_csv("Dataset/vis/nodes_features_vis.csv",index=False)  
        
    return {"name": "Alice", "age": 30}, 200,{'Content-Type': 'application/json'}

#得到连接
@app.route('/api/get_links')
def link_api():
    file_path = "Dataset/vis/nodes_links_vis.csv"
    if csv_file_has_non_empty_data(file_path):
        df = pd.read_csv(file_path)
        dics = []
        for id,row in df.iterrows():
            dic = {"source":int(row[0]),"target":int(row[1])}
            dics.append(dic)
        
        return dics, 200, {'Content-Type': 'application/json'}
    else:
        df = pd.read_csv("Dataset/vis/nodes_features.csv")
        names = []
        for name in df['name']:
            names.append(name)
        L = len(names)
        dics = []
        for i in range(L):
            for j in range(i+1,L):
                random_number = random.uniform(0, 10)
                if random_number>8:
                    dic = {"source": i+1, "target": j+1}
                    dics.append(dic)
        filename = "Dataset/vis/nodes_links_vis.csv"
        with open(filename, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # 写入数据行
            writer.writerow(["source","target"])
            for item in dics:
                writer.writerow([item["source"],item["target"]])
        print(dics)
    return dics, 200, {'Content-Type': 'application/json'}

#得到类别
@app.route('/api/classes')
def classes_api():
    df_classes = pd.read_csv("Dataset/txs_classes.csv")
    return df_classes.to_json(orient="records"), 200, {'Content-Type': 'application/json'}

#得到节点
@app.route('/api/get_nodes')
def get_nodes_api():
    file_path = "Dataset/vis/nodes_features_vis.csv"
    if csv_file_has_non_empty_data(file_path):
        print(f"{file_path} exists and contains non-empty data.")
        df = pd.read_csv(file_path)
        dics = []
        for id,row in df.iterrows():
            dic = {}
            for key,value in row.items():
                dic[key] = value
            dics.append(dic)
        return dics, 200, {'Content-Type': 'application/json'}
    else:
        print(f"{file_path} is either missing or contains only empty data.")
        df = pd.read_csv("Dataset/vis/nodes_features.csv")
        cols = df.columns[:2]
        nodes = []
        keys = []
        for i in range(df.shape[0]):
            keys.append([])

        vals = []
        for i in range(df.shape[0]):
            vals.append([])

        id = 0
        for value in df[df.columns[0]]:
            keys[id].append("name")
            vals[id].append(value)
            id = id + 1

        id = 0
        for value in df[df.columns[1]]:
            keys[id].append("x")
            vals[id].append(random.randint(0, 1000))
            keys[id].append("y")
            vals[id].append(random.randint(0, 1000))
            keys[id].append("kind")
            vals[id].append(value)
            keys[id].append("itemStyle")
            tmp = {"color":class_to_color(value-1)}
            vals[id].append(tmp)
            keys[id].append("info")
            vals[id].append(class_to_info(value-1))
            id = id + 1   

        nodes = []
        dics = []
        for i in range(df.shape[0]):
            dics.append(dict(zip(keys[i], vals[i])))

        rows = []
        for dic in dics[:1]:
            row = []
        for key, value in dic.items():
            row.append(key)
        rows.append(row)

        for dic in dics:
            row = []
            for key, value in dic.items():
                row.append(value)
            rows.append(row)    
        print(rows)
    # 指定文件路径和模式
        filename = "Dataset/vis/nodes_features_vis.csv"
        with open(filename, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
        # 写入数据行
            for row in rows:
                writer.writerow(row)
    return dics, 200, {'Content-Type': 'application/json'}

#测试apis
@app.route('/test_apis')
def test():
    return render_template('test_apis.html')

@app.route('/page3.html')
def index3():
    return render_template('page3.html')


@app.route('/page2.html')
def index2():
    return render_template('page2.html')


@app.route('/page1.html')
def index1():
    return render_template('page1.html')

@app.route('/page0.html')
def index0():
    return render_template('page0.html')

@app.route('/')
def index():
    return render_template('page0.html')
                                                                    
def init():
    # 遍历文件夹
    directory_path = "Dataset/vis"
    for filename in os.listdir(directory_path):
         # 获取完整文件路径
        file_path = os.path.join(directory_path, filename)

      # 检查是否为文件（而非子目录）
        if os.path.isfile(file_path):
          print(file_path)
          os.remove(file_path)
          fd = os.open(file_path, os.O_CREAT | os.O_WRONLY)
          os.close(fd)

    df = pd.read_csv("Dataset/utils/mus_stds.csv")
    cols = df.columns.to_list()
    lists = []
    lists = [[] for _ in range(nums+1)]
    col_cnt = 0
    for col in cols:
        col_cnt = col_cnt + 1
        val = np.random.normal(df[col][0], df[col][1],nums)   
        for i in range(1,nums+1):
            lists[i].append(val[i-1])

    lists_to_write = []
    if('class' not in cols):
        cols.insert(0,"class")
        cols.insert(0,"name")
        lists_to_write.append(cols)
        for i in range(1,nums+1):
            lists[i].insert(0,3)
            lists[i].insert(0,"node_"+str(i)) 
        for i in range(1,nums+1):
            lists_to_write.append(lists[i])
    
    
    csv_file_path = 'Dataset/vis/nodes_features.csv'
    with open(csv_file_path, mode='w', encoding='utf-8', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(lists_to_write)

if __name__ == '__main__':

    init()
    app.run(debug=True,port=5050)



    