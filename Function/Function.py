# 必要なもののインポート
# 不必要なものもあるかもしれない
from collections import Counter
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from numpy.random import seed
from rdkit.Chem import Lipinski
# ランダムシードを設定
seed(0)
import seaborn as sns
import matplotlib.pyplot as plt
from .draw import sdf_to_svg1, sdf_to_svg2
from .similarity import fragment_simi_rep_allmol

# 型アノテーション
from collections.abc import Collection, Iterable, Sequence
from typing import Any, Dict, List
from rdkit.Chem.rdchem import Mol
from .typing import rdkitMol




def __canonicalize(smiles: str) -> str:
    """Canonical SMILESを作成"""
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) # type: ignore (Pylance, MolToSmiles(), MolFromSmiles())


def genDataFrameSortedByAtomNum(li: Collection[rdkitMol]) -> pd.DataFrame:
    """
    それぞれのフラグメントを構成する原子数を調べ、原子数順で並び替えてDataFrameを作る関数
    """
    sorted_list = sorted(li, key=lambda mol: mol.GetNumAtoms())

    df = pd.DataFrame(
        data={"mol" : [m for m in sorted_list],
              "atomNum" : [m.GetNumAtoms() for m in sorted_list]}
    )

    return df


#################################################################

def frag_kind_cnt(repmol: Sequence[rdkitMol], moldata: Sequence[rdkitMol], fname: str, filepath: str) -> None:
    """フラグメントの種類で類似度別にフラグメントの数をカウントし、棒グラフとして描画する"""
    simi = fragment_simi_rep_allmol(repmol, moldata).T
    result = []
    for li in simi:
        result.append(max(li))
    plt.figure(figsize=(10, 7))  
     
    hist, edges = np.histogram(result, bins=np.arange(0, 1.05, 0.05))
    normalized_hist = hist / np.sum(hist)
    plt.bar(edges[:-1], normalized_hist, width=np.diff(edges), alpha=0.5, color=['blue'])
    # グラフの設定
    plt.xlabel('Similarity', fontsize=18)
    plt.ylabel('Proportion of fragment count', fontsize=18)

    plt.savefig(filepath + 'ward_result/frag_and_ligand_cnt/fragkind/' + fname  +  '_fragkind.png', format='png')
    plt.close('all')
    
def frag_all_cnt(repmol: Sequence[rdkitMol], moldata: Sequence[rdkitMol], fname: str, filepath: str) -> None:
    """全てのフラグメントについて(同じフラグメントで重複がある)、類似度別にフラグメントの数をカウントし、棒グラフとして描画する"""
    simi = fragment_simi_rep_allmol(repmol, moldata).T
    
    result = []
    for li in simi:
        result.append(max(li))
    plt.figure(figsize=(10, 7)) 
       
    hist, edges = np.histogram(result, bins=np.arange(0, 1.05, 0.05))
    normalized_hist = hist / np.sum(hist)
    plt.bar(edges[:-1], normalized_hist, width=np.diff(edges), alpha=0.5, color=['blue'])
 
    # グラフの設定
    plt.xlabel('Similarity', fontsize=18)
    plt.ylabel('Proportion of fragment count', fontsize=18)
    plt.savefig(filepath + 'ward_result/frag_and_ligand_cnt/allfrag/' + fname  +  '_allfrag.png', format='png')
    plt.close('all')

    
def ligand_cnt(fInfo: Sequence[rdkitMol], rep_fs: Sequence[rdkitMol]) -> pd.DataFrame:
    """構成フラグメントのうち最も高い類似度を持つ、構成フラグメントを持つリガンドの数を類似度別にカウント"""
    #リガンドの数をカウント　クラスタ方法はまとめてグラフ化する
    ligand_data = np.empty((len(fInfo), 2), dtype=Mol)
    for i in range(len(fInfo)):
        ligand_data[i][0] = fInfo[i]
        lst: List[str] = []
        li: List[str] = fInfo[i].GetProp('fragment_info').split(',')
        for f in li:
            lst.append(f)
        ligand_data[i][1] = lst

    cnt_li1=[0] * 21

    for pair in ligand_data:
        ligand = pair[0]
        fs = pair[1]
        #リガンドについて構成フラグメントの最大値を格納する 
        df_max = pd.DataFrame(index=["similarity"],
                              columns=[f for f in fs if Chem.MolFromSmiles(f) is not None]) # type: ignore (Pylance, MolFromSmiles())
        simi = fragment_simi_rep_allmol(rep_fs, [Chem.MolFromSmiles(f) for f in fs if Chem.MolFromSmiles(f) is not None]).T # type: ignore (Pylance, MolFromSmiles())
        df_simi = pd.DataFrame(simi, index=[f for f in fs if Chem.MolFromSmiles(f) is not None], # type: ignore (Pylance, MolFromSmiles())
                               columns=[Chem.MolToSmiles(m) for m in rep_fs]) # type: ignore (Pylance, MolFromSmiles())


        for index, row in df_simi.iterrows():
            df_max[index] = max(row)


        ma = max(df_max.loc["similarity"])
        if ma >= 0.0 and ma < 0.05:
            cnt_li1[0] += 1
        elif ma >= 0.05 and ma < 0.1:
            cnt_li1[1] += 1
        elif ma >= 0.1 and ma < 0.15:
            cnt_li1[2] += 1
        elif ma >= 0.15 and ma < 0.2:
            cnt_li1[3] += 1
        elif ma >= 0.2 and ma < 0.25:
            cnt_li1[4] += 1
        elif ma >= 0.25 and ma < 0.3:
            cnt_li1[5] += 1
        elif ma >= 0.3 and ma < 0.35:
            cnt_li1[6] += 1
        elif ma >= 0.35 and ma < 0.4:
            cnt_li1[7] += 1
        elif ma >= 0.4 and ma < 0.45:
            cnt_li1[8] += 1
        elif ma >= 0.45 and ma < 0.5:
            cnt_li1[9] += 1
        elif ma >= 0.5 and ma < 0.55:
            cnt_li1[10] += 1
        elif ma >= 0.55 and ma < 0.6:
            cnt_li1[11] += 1
        elif ma >= 0.6 and ma < 0.65:
            cnt_li1[12] += 1
        elif ma >= 0.65 and ma < 0.7:
            cnt_li1[13] += 1
        elif ma >= 0.7 and ma < 0.75:
            cnt_li1[14] += 1
        elif ma >= 0.75 and ma < 0.8:
            cnt_li1[15] += 1
        elif ma >= 0.8 and ma < 0.85:
            cnt_li1[16] += 1
        elif ma >= 0.85 and ma < 0.9:
            cnt_li1[17] += 1
        elif ma >= 0.9 and ma < 0.95:
            cnt_li1[18] += 1
        elif ma >= 0.95 and ma < 1.0:
            cnt_li1[19] += 1
        elif ma == 1.0:
            cnt_li1[20] += 1
    dfcnt = pd.DataFrame(columns=["mol count"], index=["0.0~0.05", "0.05~0.1", "0.1~0.15", "0.15~0.2", "0.2~0.25", "0.25~0.3", "0.3~0.35", "0.35~0.4", "0.4~0.45", "0.45~0.5", "0.5~0.55", "0.55~0.6", "0.6~0.65", "0.65~0.7", "0.7~0.75", "0.75~0.8", "0.8~0.85", "0.85~0.9", "0.9~0.95", "0.95~1.0", "1.0"])
    dfcnt["mol count"] = cnt_li1
    return dfcnt


##################################################################################################################################
def make_dis(rep: Sequence[rdkitMol], mols: Sequence[rdkitMol]) -> pd.DataFrame:
    """行が代表フラグメント　列が全てのフラグメントのdfを作り、それぞれのフラグメントの距離を計算して格納したdf"""
    simi = (fragment_simi_rep_allmol(rep, mols)).T
    df_dis = pd.DataFrame(simi, index=[Chem.MolToSmiles(m) for m in mols], # type: ignore (Pylance, MolToSmiles())
                        columns=[Chem.MolToSmiles(m) for m in rep]) # type: ignore (Pylance, MolToSmiles())
    return df_dis

def looklike_fs_to_svg(df_dis: pd.DataFrame, clustering: str, filepath: str) -> pd.DataFrame:
    """行がフラグメント　カラム"Rep fragment"に最も類似度が高い代表フラグメント、カラム"Simirality"にその代表フラグメントとの類似度を格納"""
    df_simi = pd.DataFrame(columns=["Rep fragment", "Simirality"], index=df_dis.index)
    for label, item in df_dis.iterrows():
        li = list(item)
        ind = li.index(max(li))
        df_simi["Rep fragment"].loc[label] = df_dis.columns[ind]
        df_simi["Simirality"].loc[label] = max(li)
     
    dfsimi = df_simi.sort_values('Simirality')
    dfs = []
    filenamelist = []
    width = 0.01
    a = 0.35
    while a <= 0.51:
        dfs.append(dfsimi[(dfsimi['Simirality'] >= a) & (dfsimi['Simirality'] < a+width)])
        itemstr = "_" + str(round(a, 2)) + "_" + str(round(a+width, 2))
        filenamelist.append(itemstr)
        a += width
        
    df_li=[]
    for df in dfs:
        df_li.append(df.sort_values('Rep fragment'))
        
    i=0
    for df in df_li:
        fs_set = []
        filename = 'ward_result/check_similarity/' + clustering + filenamelist[i] + '.svg'
        for label, item in df.iterrows():
            fs_set.append(Chem.MolFromSmiles(item[0])) # type: ignore (Pylance, MolFromSmiles())
            fs_set.append(Chem.MolFromSmiles(label)) # type: ignore (Pylance, MolFromSmiles())
        if len(fs_set) != 0:
            sdf_to_svg1(fs_set, None, None, filename, filepath)
        i += 1
    return dfsimi

#################################################################################################################################################
#以下、評価指数の関数

def make_rep_maxF_similarity(rep: Collection[rdkitMol], allf: Collection[rdkitMol]) -> pd.DataFrame:
    """代表フラグメントと全てのフラグメントの類似度表を作る"""
    rep_sorted: List[str] = [Chem.MolToSmiles(mol) for mol in genDataFrameSortedByAtomNum(rep)['mol']] # type: ignore (Pylance, MolToSmiles())
    # 水素原子の除去
    mols_noH = [Chem.RemoveHs(mol) for mol in allf] # type: ignore (Pylance, RemoveHs())
    # 異常な分子の確認と除去
    allff = []
    for mol in mols_noH:
        if Chem.MolFromSmiles(Chem.MolToSmiles(mol)) is not None: # type: ignore (Pylance, MolFromSmiles(), MolToSmiles())
            allff.append(mol)
            
    fx_: List[str] = []
    for s in allff:
        fx_.append(Chem.MolToSmiles(s)) # type: ignore (Pylance, MolToSmiles())
    fx_sorted = list(set(fx_)) 
    df = pd.DataFrame(fragment_simi_rep_allmol([Chem.MolFromSmiles(s) for s in rep_sorted],  # type: ignore (Pylance, MolFromSmiles())
                                               [Chem.MolFromSmiles(s) for s in fx_sorted]),  # type: ignore (Pylance, MolFromSmiles())
                                               columns=fx_sorted, index=rep_sorted)

    #最も類似度の高い代表フラグメントのみを抽出したdfを作る
    #フラグメントと最も類似度の高い代表フラグメントと、実際の類似度
    df_max = pd.DataFrame(index=fx_sorted, columns=['rep_fragment' , 'similarity'])

    # 各列の最大値とその行名（インデックス）を取得
    df = df.astype(float)
    for col in df.columns:
        max_value = df[col].max()
        max_row = df[col].idxmax()
        df_max.at[col, 'rep_fragment'] = max_row
        df_max.at[col, 'similarity'] = max_value
        
    return df_max

def make_50minligand_fragment_dict(df_max: pd.DataFrame, ligand: Iterable[rdkitMol]) -> Dict[rdkitMol, pd.DataFrame]:
    """代表フラグメントと、重原子数が50以下の全てのフラグメントの類似度表を作る
    （あんまり大きいフラグメントは見なくてもいいかもという助言があったので）"""
    # key: 化合物, value: df(構成フラグメントと代表フラグメントの類似度のうち最大のもの)
    f_dict = {}
    for mol in ligand:
        li = mol.GetProp('fragment_info').split(',')
        n_li = []
        for s in li:
            if Chem.MolFromSmiles(s) != None: # type: ignore (Pylance, MolFromSmiles())
                n_li.append(__canonicalize(s))
        n_li1 = list(n_li)
        df = pd.DataFrame(index=n_li1, columns=['rep_fragment', 'similarity'])
        df = df.astype(float)
        for ss in df.index:
            if ss in list(df_max.index):
                simi = df_max.at[ss, 'similarity']
                df.at[ss, 'similarity'] = simi
                df.at[ss, 'rep_fragment'] = df_max.at[ss, 'rep_fragment']
        if (Chem.RemoveHs(mol)).GetNumAtoms() <= 50: # type: ignore (Pylance, RemoveHs())
            f_dict[Chem.RemoveHs(mol)] = df # type: ignore (Pylance, RemoveHs())
                
    return f_dict


# 
def make_allligand_fragment_dict(df_max: pd.DataFrame, ligand: Iterable[rdkitMol]) -> Dict[rdkitMol, pd.DataFrame]:
    """
    化合物に対して、「その構成フラグメント・構成フラグメントと最も類似度が高い代表フラグメント・その代表フラグメントとの類似度」をdfにして紐づける
    key: 化合物, value: df(構成フラグメントと代表フラグメントの類似度のうち最大のもの)
    """
    f_dict = {}
    for mol in ligand:
        li = mol.GetProp('fragment_info').split(',')
        n_li = []
        for s in li:
            if Chem.MolFromSmiles(s) != None: # type: ignore (Pylance, MolFromSmiles())
                n_li.append(__canonicalize(s))
        n_li1 = list(n_li)
        df = pd.DataFrame(index=n_li1, columns=['rep_fragment', 'similarity'])
        df = df.astype(float)
        for ss in df.index:
            if ss in list(df_max.index):
                simi = df_max.at[ss, 'similarity']
                df.at[ss, 'similarity'] = simi
                df.at[ss, 'rep_fragment'] = df_max.at[ss, 'rep_fragment']
        f_dict[Chem.RemoveHs(mol)] = df # type: ignore (Pylance, RemoveHs())
                
    return f_dict

def occupy_ligand(threshold: float, f_dict: Dict[rdkitMol, pd.DataFrame]) -> pd.DataFrame:
    """構成フラグメントのうち、類似度がds以上のフラグメント重原子数の合計　/ 元のリガンドの重原子数"""
    result_df = pd.DataFrame(index=list(f_dict.keys()))
    for keyy in f_dict:
        df = f_dict[keyy]
        l_atomNum = keyy.GetNumAtoms() 
        # l_atomNum = keyy.GetNumHeavyAtoms() 
        
        f_atomNum = 0
        #構成フラグメントの類似度のうち、閾値を超えるフラグメントの重原子数の合計
        for s in df[df['similarity'] >= threshold].index:
            f_atomNum += Chem.MolFromSmiles(s).GetNumAtoms()   # type: ignore (Pylance, MolFromSmiles())
            # f_atomNum += Chem.MolFromSmiles(s).GetNumHeavyAtoms()  
            
        result_df.at[keyy, 'similarity'] = round(f_atomNum / l_atomNum, 4)
        result_df.at[keyy, 'ligandAtomNum'] = l_atomNum
        

    return result_df


def ligand_hist_scatter(dict: Dict[rdkitMol, pd.DataFrame], filename: str, rep_threshold: float, filepath: str) -> None:
    # #resultをヒストグラムに描画したい
    threshold = rep_threshold
    while threshold < rep_threshold + 0.01:
        result = occupy_ligand(threshold, dict)
        g = sns.jointplot(data=result, 
                          x='similarity', y='ligandAtomNum', 
                          ratio=2, 
                          kind="scatter", 
                          joint_kws={'alpha': 0.2}, 
                          marginal_ticks=True,
                          )
        g.ax_joint.set_title("Threshold = " + str(threshold))

        # 縦軸と横軸のラベルを追加
        g.ax_joint.set_xlabel("S = AtonNum of constitute frags >= threshold / ligand AtomNum")
        g.ax_joint.set_ylabel("ligand AtomNum")
        
        plt.savefig(filepath + 'ward_result/new_similarity_result/' + filename + '_' + str(threshold) + '.png', format='png')
        plt.show()
        threshold = round(threshold+0.01, 2)
        
def plot_bar_allF_repF(rep_f_df: pd.DataFrame, all_f: Sequence[rdkitMol], filename: str, rep_threshold: float, filepath: str) -> pd.DataFrame:
    """
    出現頻度の評価指標
    横軸　代表フラグメント 縦軸　全てのフラグメント(13000個)
    """
    # threshold = 0.45
    freqli = []
    smilesli = [Chem.MolToSmiles(m) for m in rep_f_df['mol']] # type: ignore (Pylance, MolToSmiles())
    for s in smilesli:
        mol = Chem.MolFromSmiles(s) # type: ignore (Pylance, MolFromSmiles())
        simi = fragment_simi_rep_allmol([mol], all_f)
        result = simi[simi >= rep_threshold]
        freqli.append(len(result))
    freq_df = pd.DataFrame(np.array(rep_f_df['mol']).T, columns=['rep_fragment'])
    freq_df['frequency'] = np.array(freqli).T
    freq_df['percentage'] = (np.array(freqli)/len(all_f)).T
    freq_df['smiles'] = smilesli
    
    freq_df_sorted = freq_df.sort_values('frequency', ascending=False).reset_index(drop=True) 
    ax = freq_df_sorted.reset_index().plot.bar(figsize=(13, 7), x='index', y='percentage', width=0.9, alpha=0.7)
    ax.set_xlabel("Each representative fragent's index",  wrap=True, fontsize=18)
    ax.set_ylabel("The proportion of the total number of fragments with high similarity",  wrap=True, fontsize=18)
    ax.legend([])
    title= filename + '_repfrag_freq_percentage'
    plt.savefig(filepath + 'ward_result/each_repfrag_freq/'+ title + '.png')
    plt.show()
    
    sdf_to_svg2(list(freq_df_sorted['rep_fragment']), None, 15, 'ward_result/each_repfrag_freq/' + filename +'_repfrag_freq_sorted_top15.svg', list(freq_df_sorted['frequency']), filepath)
    
    
    return freq_df_sorted

    

 ##各フラグメントごと　代表フラグメントrep1との類似度0.45以上のフラグメントを持つリガンドL1, L2, L3, ...の総原子数をカウント
#その代表フラグメントがどれくらいの原子数を表現できるのか
def plot_bar_AllAtomNum_repF(rep_mol_df: pd.DataFrame, all_fragment: Sequence[rdkitMol], 
                             filename: str, allL_f_dict: Dict[rdkitMol, Any], 
                             rep_threshold: float, filepath: str) -> pd.DataFrame:
    """ 
    各フラグメントごと　代表フラグメントrep1との類似度0.45以上のフラグメントを持つリガンドL1, L2, L3, ...の総原子数をカウント
    その代表フラグメントがどれくらいの原子数を表現できるのか
    """
    # フラグメント全ての原子数を確認
    allLigandAtomNum = 0
    for ligand in allL_f_dict.keys():
        l = Chem.RemoveHs(ligand) # type: ignore (Pylance, RemoveHs())
        allLigandAtomNum += l.GetNumAtoms()
    print("リガンドの原子総数：", allLigandAtomNum) 
    x_list = []
    
    smilesli = [Chem.MolToSmiles(m) for m in rep_mol_df['mol']] # type: ignore (Pylance, MolToSmiles())
    for s in smilesli:
        mol = Chem.MolFromSmiles(s) # type: ignore (Pylance, MolFromSmiles())
        simi = fragment_simi_rep_allmol([mol], all_fragment)
        simi_df = pd.DataFrame(simi.T, index=all_fragment, columns=[s])
        result = simi_df[simi_df[s]>=rep_threshold]
        more045 = result.index
        atomnum = 0
        for mol in more045:
            atomnum += mol.GetNumAtoms()
        x_list.append(atomnum)
            
    atomNum_df = pd.DataFrame(np.array(rep_mol_df['mol']).T, columns=['rep_fragment'])
    atomNum_df['MolAtomNum'] = np.array(rep_mol_df["atomNum"]).T
    atomNum_df['AllAtomNum'] = np.array(x_list).T
    atomNum_df['percentage(' + str(allLigandAtomNum) + ')'] = (atomNum_df['AllAtomNum']/allLigandAtomNum)
    atomNum_df_sorted = atomNum_df.sort_values('AllAtomNum', ascending=False).reset_index(drop=True)
    
    ax = atomNum_df_sorted.reset_index().plot.bar(figsize=(13, 7), x='index', y='percentage(' + str(allLigandAtomNum) + ')', width=0.9, alpha=0.7, color='orange')
    ax.set_xlabel("Each representative fragment's index",  wrap=True, fontsize=18)
    ax.set_ylabel("The proportion of the total number of heavy atoms in fragments of high similarity",  wrap=True, fontsize=18)
    ax.legend([])
    plt.savefig(filepath + 'ward_result/AllAtomNum_AllLigandAtomNum/' + filename +  '_AtomNumPercentage.png', format='png')
    plt.show()
    
    sdf_to_svg2(list(atomNum_df_sorted['rep_fragment']), None, 15, 'ward_result/AllAtomNum_AllLigandAtomNum/' + filename + '_AtomNumCount_sorted_top15.svg', list(atomNum_df_sorted['AllAtomNum']), filepath)
    
    return atomNum_df_sorted 



def fragment_freq_accumulation(freq_df_sorted: pd.DataFrame, all_f: Sequence[rdkitMol], 
                               filename: str, rep_threshold: float, filepath: str) -> pd.DataFrame:    
    """代表フラグメントの出現頻度の累積"""
    result_df = pd.DataFrame(columns=['repf_num', 'fragNum_accumulation'])
    fsnum_li = []
    repf_num = []
    # threshold = 0.45
    
    all_simi = fragment_simi_rep_allmol(freq_df_sorted['rep_fragment'], all_f) # type: ignore (freq_df_sorted['rep_fragment'] = Series[rdkitMol])
    all_simi_df = pd.DataFrame(all_simi.T, 
                               columns=[Chem.MolToSmiles(m) for m in freq_df_sorted['rep_fragment']],  # type: ignore (Pylance, MolToSmiles())
                               index=[Chem.MolToSmiles(m) for m in all_f]) # type: ignore (Pylance, MolToSmiles())
    for i in range(1, len(freq_df_sorted) + 1):
        top_df = freq_df_sorted[freq_df_sorted.index < i]
        repfs = [Chem.MolToSmiles(m) for  m in list(top_df['rep_fragment'])] # type: ignore (Pylance, MolToSmiles())
        a = all_simi_df[repfs]
        b = a[a >= rep_threshold].dropna(how='all')
        fsnum_li.append(len(b))
        repf_num.append(i)
            
    result_df['fragNum_accumulation'] = np.array(fsnum_li).T
    result_df['fragNum_accumulation_per'] = (np.array(fsnum_li)/len(all_f)).T
    result_df['repf_num'] = np.array(repf_num).T
    ax = result_df.reset_index().plot.bar(figsize=(13, 7), x='repf_num', y='fragNum_accumulation_per', width=0.9, alpha=0.7)
    ax.set_xlabel("The number of representative fragments",  wrap=True, fontsize=18)
    ax.set_ylabel("The proportion of the total number of fragments with high similarity",  wrap=True, fontsize=18)
    
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.legend([])
    
    title= filename + '_repfrag_freq_accumulation_per'
    # plt.title(str(rep_threshold) + ' >=fragment similarity Count Accumulation(per) / All fragment Num:' + str(len(all_f)))
    plt.savefig(filepath + 'ward_result/each_repfrag_freq/'+ title + '.png')
    plt.show()
    return result_df
    
      
def all_fragmetntAtomNum_accumulation(atomnum_df_sorted: pd.DataFrame, all_fragment: Sequence[rdkitMol], 
                                      filename: str, allL_f_dict: Dict[rdkitMol, Any], rep_threshold: float, filepath: str) -> pd.DataFrame:
    """代表フラグメントの表現できる原子数の累積"""
    # フラグメント全ての原子数を確認
    allLigandAtomNum = 0
    for ligand in allL_f_dict.keys():
        l = Chem.RemoveHs(ligand) # type: ignore (Pylance, RemoveHs())
        allLigandAtomNum += Lipinski.HeavyAtomCount(l)
        # allLigandAtomNum += l.GetNumAtoms()
        
    result_df = pd.DataFrame(columns=['repf_num', 'AtomNum_accumulation'])
    atomnumli = []
    repf_num = []
    # threshold = 0.45
    
    all_simi = fragment_simi_rep_allmol(atomnum_df_sorted['rep_fragment'], all_fragment) # type: ignore (atomnum_df_sorted['rep_fragment'] = Series[rdkitMol])
    all_simi_df = pd.DataFrame(all_simi.T, 
                               columns=[Chem.MolToSmiles(m) for m in atomnum_df_sorted['rep_fragment']],  # type: ignore (Pylance, MolToSmiles())
                               index=[Chem.MolToSmiles(m) for m in all_fragment]) # type: ignore (Pylance, MolToSmiles())
    
    
    for i in range(1, len(atomnum_df_sorted) + 1):
        top_df = atomnum_df_sorted[atomnum_df_sorted.index < i]
        repfs = [Chem.MolToSmiles(m) for  m in list(top_df['rep_fragment'])] # type: ignore (Pylance, MolToSmiles())
        a = all_simi_df[repfs]
        hit_frags = a[a >= rep_threshold].dropna(how='all').index
        atomnum = 0
        for f in hit_frags:
            atomnum += Lipinski.HeavyAtomCount(Chem.MolFromSmiles(f)) # type: ignore (Pylance, MolFromSmiles())
            
            # atomnum += Chem.MolFromSmiles(f).GetNumAtoms()
        atomnumli.append(atomnum) 
        repf_num.append(len(repfs))
         
    result_df['AtomNum_accumulation'] = np.array(atomnumli).T
    result_df['AtomNum_accumulation_per'] = (np.array(atomnumli)/allLigandAtomNum).T
    result_df['repf_num'] = np.array(repf_num).T
    
    ax = result_df.reset_index().plot.bar(figsize=(13, 7), x='repf_num', y='AtomNum_accumulation_per', width=0.9, alpha=0.7, color='orange')
    ax.set_xlabel("The number of representative fragments",  wrap=True, fontsize=18)
    ax.set_ylabel("The proportion of the total number of heavy atoms in fragments of high similarity",  wrap=True, fontsize=18)
    ax.legend([])
    
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    
    # plt.title("Total Ligand Atoms Num Accumlation per (Ligand " + str(allLigandAtomNum) + ")")
    plt.savefig(filepath + 'ward_result/AllAtomNum_AllLigandAtomNum/' + filename  +  '_AtomNumCountAccumulation_per.png', format='png')
    plt.show()
    return result_df


##################################################################################################################################
# ここからは代表フラグメントをpdfやcsvにするための関数


def make_sdf(mol_li: Collection[rdkitMol], fname: str, ver: int, filepath: str) -> None:
    """sdfファイルを作成するための関数"""
    w = Chem.SDWriter(filepath + "to_pdf/sdffile/" + fname + "ver" + str(ver) + "_rep.sdf") # type: ignore (Pylance, SDWriter())
    rep_li = genDataFrameSortedByAtomNum(mol_li)
    mol_sorted_li = rep_li['mol']
    for i, mol in enumerate(mol_sorted_li):
        if mol is not None:
            AllChem.Compute2DCoords(mol)  # Compute 2D coordinates if not already present # type: ignore (Pylance, Compute2DCoords())
            # 分子に一意のIDを設定
            id = 'v' + str(ver) + '-' + str(i+1)
            mol.SetProp('fragmentID', str(id))
            w.write(mol)
    w.close() 


def all_evaluation(rep_mol: Sequence[rdkitMol], rep_mol_sorted: Sequence[rdkitMol], 
                   fname: str, frags: Sequence[rdkitMol], ligand: Iterable[rdkitMol], 
                   all_fragment: Sequence[rdkitMol], rep_threshold: float, 
                   allL_f_dict: Dict[rdkitMol, pd.DataFrame], 
                   cnt: Counter, filepath: str) -> None:
    """全ての評価をする"""
    df_max = make_rep_maxF_similarity(rep_mol, frags)
    f_dict = make_50minligand_fragment_dict(df_max, ligand)
    allL_f_dict = make_allligand_fragment_dict(df_max, ligand)
    

    # # #フラグメントの種類でカウント
    frag_kind_cnt(rep_mol, frags, fname, filepath)
    # # #フラグメントの総数でカウント
    frag_all_cnt(rep_mol, all_fragment, fname, filepath)
   
    # # #リガンドをカウント
    plt.figure(figsize=(5, 7))
    ligand_cnt(ligand, rep_mol_sorted['mol']).plot.bar()
    plt.savefig(filepath + 'ward_result/frag_and_ligand_cnt/ligand/' + fname  +  '_ligand.png', format='png')
    plt.close('all')
    #ヒストグラムの作成
    ligand_hist_scatter(f_dict, fname + '_hist_scatter', rep_threshold, filepath)

    # 類似度が低いフラグメントたち
    small_simi_sorted = df_max[df_max['similarity'] < rep_threshold].sort_values(by='similarity', ascending=True)
    small_simi = [Chem.MolFromSmiles(s) for s in list(small_simi_sorted.index)] # type: ignore (Pylance, MolFromSmiles())
    small_simi_rep = [Chem.MolFromSmiles(s) for s in list(small_simi_sorted['rep_fragment'])] # type: ignore (Pylance, MolFromSmiles())
    result = [item for sublist in zip(small_simi_rep, small_simi) for item in sublist]
    rep = ['rep' for i in range(len(small_simi))]
    legends = [item for sublist in zip(rep, [str(round(s, 3)) for s in small_simi_sorted['similarity']]) for item in sublist]
    print(len(result), len(legends))
    img = Draw.MolsToGridImage(result, maxMols=3000, molsPerRow=2, subImgSize=(500, 500), useSVG=True, legends=legends)
    place = filepath + 'ward_result/small_simiSet_' + fname + '.svg'
    with open(place, mode='w') as f:
        f.write(img.data) # type: ignore (Pylance, img.data)
    small_simi_atomnum_sorted = genDataFrameSortedByAtomNum(small_simi)
    small_simi_atomnum_sorted_smiles =[Chem.MolToSmiles(m) for m in small_simi_atomnum_sorted['mol']] # type: ignore (Pylance, MolToSmiles())
    small_simi_freq = [cnt[s] for s in small_simi_atomnum_sorted_smiles]
    img = Draw.MolsToGridImage(small_simi_atomnum_sorted['mol'], maxMols=3000, molsPerRow=5, subImgSize=(500, 500), useSVG=True, legends=[str(num) for num in small_simi_freq])
    place = filepath + 'ward_result/small_simi_' + fname + '.svg'
    with open(place, mode='w') as f:
        f.write(img.data) # type: ignore (Pylance, img.data)
    print("どの代表フラグメントとも類似度が低いフラグメントの種類は", len(small_simi_atomnum_sorted_smiles), "フラグメントの数は", sum(small_simi_freq))
    #各代表フラグメントごとの評価
    freq_df_sorted = plot_bar_allF_repF(rep_mol_sorted, all_fragment, fname,rep_threshold, filepath)
    atomnum_df_sorted = plot_bar_AllAtomNum_repF(rep_mol_sorted, all_fragment, fname, allL_f_dict, rep_threshold, filepath)

    # 累積
    frag_accum_df = fragment_freq_accumulation(freq_df_sorted, all_fragment, fname, rep_threshold, filepath)
    atomnum_accum_df = all_fragmetntAtomNum_accumulation(atomnum_df_sorted, all_fragment, fname, allL_f_dict, rep_threshold, filepath)
    
    freq_df_sorted.to_csv(filepath + "ward_result/each_repfrag_freq/" + fname + '_frag.csv', index=True)
    atomnum_df_sorted.to_csv(filepath + "ward_result/AllAtomNum_AllLigandAtomNum/" + fname + '_atomnum.csv', index=True)
    
    frag_accum_df.to_csv(filepath + "ward_result/each_repfrag_freq/" + fname + '_frag_accum.csv', index=True)
    atomnum_accum_df.to_csv(filepath + "ward_result/AllAtomNum_AllLigandAtomNum/" + fname + '_atomnum_accum.csv', index=True)

    
    
