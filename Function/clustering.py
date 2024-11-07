"""
フラグメントのクラスタリングを行い、代表フラグメントの取得を行う関数
"""

from collections.abc import Sequence
from typing import Any, Dict
import numpy.typing as npt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from scipy.cluster.hierarchy import ClusterWarning
from warnings import simplefilter
simplefilter("ignore", ClusterWarning)
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
from .similarity import fragment_similarity
from .typing import rdkitMol

from numpy.random import seed
seed(0)

def __repMol_min(dict: Dict[Any, Any]):
   """一番距離が近いものを取得（各クラスタで、どの化合物とも似ているものを取得）"""
   repMol_li={}
   for key, molList in dict.items():
      max_tanimoto = []
      f_simi = fragment_similarity(molList)
      #最も距離が近いものを取得
      for mf in f_simi:
         max_tanimoto.append(max(mf))
      repMol_li[key] = molList[max_tanimoto.index(min(max_tanimoto))]
   return list(repMol_li.values())

def ward_clustering(np_f: npt.NDArray, distance_matrix, num_clusters: int, v: int, filepath: str):
    """階層的クラスタリング（ウォード法を利用）"""
    # 階層的クラスタリングを実行
    linkage_matrix = linkage(distance_matrix, method='ward')
    # クラスタ数を指定してクラスタを形成
    clusters = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
    # 各クラスタに所属する要素を表示
    cluster_dict = {}
    for idx, label in enumerate(clusters):
        if label not in cluster_dict:
            cluster_dict[label] = [np_f[idx]]
        else:
            cluster_dict[label].append(np_f[idx])
      
    for _, value in cluster_dict.items():
        # print(f"クラスタ {key} 要素数: {len(value)}")
        print(f"クラスタの要素数: {len(value)}")
        
    plt.figure(figsize=(200, 100))
    plt.title('linkage=ward')
    dendrogram(linkage_matrix, labels=np.arange(1, len(distance_matrix)+1), truncate_mode='lastp', p=num_clusters)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    title = 'w' + str(num_clusters) + 'v' + str(v) +'_dendrogram.png'
    plt.savefig(filepath+'ward_result/dendrogram/' + title)
    plt.show()
    return cluster_dict



def auto_clustring_new(num: int, frags: Sequence[rdkitMol], v: int, filepath):
    """クラスタリングの自動化"""
    distance_matrix = 1 - np.array(fragment_similarity(frags))
    #ward法によるクラスタリング
    ward_dict = ward_clustering(np.array(frags), distance_matrix, num, v, filepath)
    #代表フラグメントの抽出
    # w_rep=代表フラグメント　が格納されたリスト
    w_rep = [Chem.MolFromSmiles(Chem.MolToSmiles(m)) for m in __repMol_min(ward_dict)] # type: ignore (Pylance, MolFromSmiles(), MolToSmiles)
    #代表フラグメントの描画
    title1 = str(num) + '_ward_rep'
    
    #クラスタの大きさのプロット
    cluster_num = [len(cluster) for i, cluster in enumerate(ward_dict.values())]
    ind = [i for i, cluster in enumerate(ward_dict.values())]
    c_num_df = pd.DataFrame(np.array(w_rep).T, columns=['rep_fragment'])
    c_num_df['ind'] = np.array(ind).T
    c_num_df['cluster_num'] = np.array(cluster_num).T
    ax = c_num_df.plot.bar(figsize=(10, 5), x='ind', y='cluster_num', width=0.8, color='green', alpha=0.7)
    ax.set_xlabel("cluster number", fontsize=18)
    ax.set_ylabel("fragment num", fontsize=18)
    # plt.title("cluster num")
    plt.savefig(filepath + 'ward_result/w' + str(num) + 'v'+str(v) +'_cluster_num.png', format='png')
    plt.show()
    img = Draw.MolsToGridImage(w_rep, maxMols=300, molsPerRow=5, 
                               subImgSize=(500, 500), useSVG=True, 
                               legends=[str(len(cluster)) for i, cluster in enumerate(ward_dict.values())])
    
    place = filepath + 'ward_result/clusterNum' + title1 + '.svg'
    with open(place, mode='w') as f:
        f.write(img.data) # type: ignore (Pylance, img.data)
        
    c_num_df.to_csv(filepath + 'ward_result/cluster' + title1 + '.csv', index=False)

    # 返すのは代表フラグメントが格納されたリスト
    return w_rep