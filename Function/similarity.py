from collections.abc import Sequence
from rdkit.Chem import AllChem, DataStructs, rdMolDescriptors # type: ignore (Pylance, rdMolDescriptors())
import numpy as np

from .typing import rdkitMol


def tanimoto_simi(frag: Sequence[rdkitMol]):
    """モルガンフィンガープリントからTanimoto類似度を計算"""
    #morgan　fingar printを生成してタニモト係数（距離行列）の計算
    morgan_fp = [AllChem.GetMorganFingerprintAsBitVect(x,2,1024) for x in frag] # type: ignore (Pylance, GetMorganFingerprintAsBitVect())
    #index iの分子とのタニモト係数の計算（三角行列）
    dis_matrix = [DataStructs.BulkTanimotoSimilarity(morgan_fp[i], morgan_fp) for i in range(len(morgan_fp))] # type: ignore (Pylance, BulkTanimotoSimilarity())
    dis_ = np.array(dis_matrix)
    
    return dis_

def MQN_Similarity(frags: Sequence[rdkitMol]):
    """MQNで類似度を計算"""
    ###MQN記述子を利用した新しい類似度の計算
    mqns = np.array([rdMolDescriptors.MQNs_(mol) for mol in frags])
    mqn_Simi_arr = np.zeros((len(mqns), len(mqns)))
    for i in range(len(mqns)):
        for j in range(len(mqns)):
            result = mqns[i] - mqns[j]
            mqn_Simi_arr[i, j] = 1 / (1+(sum(abs(result))/42))
    return mqn_Simi_arr

def fragment_similarity(frags: Sequence[rdkitMol]):
    """全体の類似度を計算"""
    tanimoto_arr = tanimoto_simi(frags)
    mqn_arr = MQN_Similarity(frags)
    simi_arr = (tanimoto_arr + mqn_arr) / 2
    return simi_arr


def __tanimoto_simi_rep_allmol(repli: Sequence[rdkitMol], molli: Sequence[rdkitMol]):
    """（代表フラグメントと全てのmolとの類似度計算）
    モルガンフィンガープリントから谷本類似度を計算"""
    all_mfp = [AllChem.GetMorganFingerprintAsBitVect(x,2,1024) for x in molli] # type: ignore (Pylance, GetMorganFingerprintAsBitVect())
    tanimoto = np.zeros((len(repli), len(molli)))
    for i in range(len(repli)):
        rep_mfp = AllChem.GetMorganFingerprintAsBitVect(repli[i],2,1024) # type: ignore (Pylance, GetMorganFingerprintAsBitVect())
        t = DataStructs.BulkTanimotoSimilarity(rep_mfp, all_mfp) # type: ignore (Pylance, BulkTanimotoSimilarity())
        tanimoto[i] = np.array(t)
    return tanimoto

def __mqn_simi_rep_allmol(repli: Sequence[rdkitMol], molli: Sequence[rdkitMol]):
    """（代表フラグメントと全てのmolとの類似度計算）
    MQNで類似度を計算"""
    rep_mqn = np.array([rdMolDescriptors.MQNs_(mol) for mol in repli])
    all_mqn = np.array([rdMolDescriptors.MQNs_(mol) for mol in molli])
    mqn = np.zeros((len(repli), len(molli)))
    for i in range(len(rep_mqn)):
        for j in range(len(all_mqn)):
            result = rep_mqn[i] - all_mqn[j]
            mqn[i, j] = 1 / (1+(sum(abs(result))/42))
    return mqn
            
def fragment_simi_rep_allmol(repli: Sequence[rdkitMol], molli: Sequence[rdkitMol]):
    """（代表フラグメントと全てのmolとの類似度計算）
    全体の類似度を計算"""
    tanimoto = __tanimoto_simi_rep_allmol(repli, molli)
    mqn = __mqn_simi_rep_allmol(repli, molli)
    simi_arr = (tanimoto + mqn) / 2
    return simi_arr

