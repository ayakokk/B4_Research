"""
不必要な化合物やフラグメントを除去するために必要な関数
"""

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors # type: ignore (Pylance, rdMolDescriptors)

def __removeH(li):
    """水素原子を除去"""
    rmHs = []
    for mol in li:
        if mol is None:
            continue
        rmHs.append(Chem.RemoveHs(mol)) # type: ignore (Pylance, RemoveHs())

    return rmHs


def __remMols(li):
    """GetAtomでC, N, O, P, S, F, Cl, Br, I 以外が含まれているものを除外"""
    allowed_elements = {'C', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I'}
    # 指定した元素のみを含む分子だけを含む新しいリストを作成
    filtered_list = [mol for mol in li if all(atom.GetSymbol() in allowed_elements for atom in mol.GetAtoms())]

    return filtered_list

def __remIsotopeMols(li):
    """放射性同位体を消去"""
    new_li = []
    for mol in li:
        flag = 0
        for at in mol.GetAtoms():
            if at.GetIsotope() != 0:
                flag = 1

        if flag == 0:
            new_li.append(mol)
    return new_li

def __remRingSize(li):
    """7員環以上含むフラグメント"""
    mols = []
    for mol in li:
        molInfo = mol.GetRingInfo()
        flag = 0
        for at in mol.GetAtoms():
            if molInfo.MinAtomRingSize(at.GetIdx()) > 6:
                flag = 1
        if flag == 0:
            mols.append(mol)
    return mols


def __remAromaticRing(li):
    """4つ以上の芳香環を含むフラグメントは除外"""
    new_li = []
    for mol in li:
        frag = 0
        if Descriptors.NumAromaticRings(mol) > 3: # type: ignore (Pylance, NumAromaticRings())
            frag = 1
        if frag == 0:
            new_li.append(mol)
    return new_li

def __remNonAromaticRing(li):
    """3つ以上の非芳香環を含むフラグメントは除外"""
    new_li = []
    for mol in li:
        frag = 0
        if rdMolDescriptors.CalcNumAliphaticRings(mol) > 2:
            frag = 1
            
        if frag == 0:
            new_li.append(mol)
    return new_li

def __remFourRings(li):
    """4つ以上の環を含むフラグメントは除外"""
    new_li = []
    for mol in li:
        if rdMolDescriptors.CalcNumRings(mol) < 4:
            new_li.append(mol)
    return new_li



def __remHeavyAtom(li):
    """重原子7つ以上からなる非環状フラグメントは除外"""
    filtered_list = [mol for mol in li if mol.GetNumHeavyAtoms() < 7 or mol.GetRingInfo().NumRings() > 0]
    return filtered_list


def __remOneHeavyAtom(li):
    """重原子1つしかないフラグメントは除外"""
    # 重原子が2つ以上の分子だけを含む新しいリストを作成
    filtered_list = [mol for mol in li if mol.GetNumHeavyAtoms() > 1]
    return filtered_list
 

def __remove_non_carbon(li):
    """炭素の含まれないリガンドの除去"""
    filtered_list = [mol for mol in li if any(atom.GetSymbol() == 'C' for atom in mol.GetAtoms())]
    return filtered_list


def filterings(li):
    """フラグメント用の全てのフィルタリングを適応するための関数"""
    frags = __remFourRings(__remNonAromaticRing(__remOneHeavyAtom(__remHeavyAtom(__remAromaticRing(__remRingSize(__remIsotopeMols(__remMols(__removeH(li)))))))))
    return frags


def ligand_filtering(mol_li):
    """以下を満たすリガンドの除去
    C N O P S F Cl Br I以外の原子種が含まれるリガンド
    同位体元素が含まれるリガンド
    炭素を一つも持たないリガンド"""
    ligand = __remove_non_carbon(__remIsotopeMols(__remMols(mol_li)))
    return ligand