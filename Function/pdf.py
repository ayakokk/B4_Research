

from typing import List
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors # type: ignore (Pylance, rdMolDescriptors)
from numpy.random import seed
# ランダムシードを設定
seed(0)
from rdkit.Chem import Crippen
from xhtml2pdf import pisa
import seaborn as sns
import matplotlib.pyplot as plt

from .draw import mol_to_svg
from .Function import genDataFrameSortedByAtomNum

# 型アノテーション
from collections.abc import Sequence
from .typing import rdkitMol

def make_pdf(li: Sequence[rdkitMol], finfo: Sequence[Sequence[rdkitMol]], fname, ver, filepath):
    """代表フラグメントをpdfにするための関数"""
    rep_li = genDataFrameSortedByAtomNum(li)
    rep_li['SMILES'] = [Chem.MolToSmiles(mol) for mol in rep_li['mol']] # type: ignore (Pylance, MolToSmiles())
    rep_li['Molecular Weight'] = [round(rdMolDescriptors._CalcMolWt(mol), 2) for mol in rep_li['mol']]
    rep_li['Log P'] = [round(Crippen.MolLogP(mol), 3) for mol in rep_li['mol']]
    # rep_sorted = rep_li.sort_values(by=['atomNum', 'Molecular Weight'])
    rep_sorted = rep_li
    rep_sorted.index = ['v' + str(ver) + '-' + str(i+1) for i in range(len(rep_sorted))]
    rep_sorted['ligand'] = 'any'
    rep_sorted['ligand path'] = [filepath + 'to_pdf/ligands/' + fname + 'v' + str(ver) + '-' + str(i+1) + '.png' for i in range(len(rep_sorted))]
    rep_sorted.insert(4, 'smiles path',  [filepath + 'to_pdf/mol/' + fname + 'v' + str(ver) + '-' + str(i+1) +'.png' for i in range(len(rep_sorted))])

    repf_ligand: List[Sequence[rdkitMol]] = []
    for repmol in rep_sorted['mol']:
        li = [repmol]
        for f in finfo:
            if Chem.MolToSmiles(repmol) == Chem.MolToSmiles(f[1]): # type: ignore (Pylance, MolToSmiles())
                li.append(f[0])
        repf_ligand.append(li)

    for li in repf_ligand:
        if len(li) > 1:
            for ligand in li[1:]:
                if ligand.GetNumAtoms() >= 20 and ligand.GetNumAtoms() <= 30:
                    ########################################
                    rep_sorted.loc[rep_sorted['mol'] == li[0], 'ligand'] = ligand
                    break
            else:
                rep_sorted.loc[rep_sorted['mol'] == li[0], 'ligand'] = li[1]
                
                continue
        else:
            rep_sorted.loc[rep_sorted['mol'] == li[0], 'ligand'] = None
            
            
    
    for mol in rep_sorted["mol"]:
        if mol is not None:
            
            filename = filepath + 'to_pdf/mol/' + str(fname) + rep_sorted.index[rep_sorted['mol'] == mol].tolist()[0] +'.png'
            mol_to_svg([mol], filename)

    for mol in rep_sorted['ligand']:
        if mol is not None:
        
            filename = filepath + 'to_pdf/ligands/' + str(fname) + rep_sorted.index[rep_sorted['ligand'] == mol].tolist()[0] +'.png'
            mol_to_svg([mol], filename)

    #散布図の保存
    plt.figure()
    rep_sorted.plot.scatter(x='Molecular Weight', y='Log P', title = 'Molecular Weight and LogP Scatter')
    plt.savefig(filepath + 'to_pdf/img/' + str(fname) + '_scatter.png')
    plt.close('all')

    rep_sorted.insert(3, 'SMILES img',  rep_sorted["smiles path"].map(lambda s: "<img src='{}' width='100' />".format(s)))
    rep_sorted["ligand img"] = rep_sorted["ligand path"].map(lambda s: "<img src='{}' width='100' />".format(s))
    rep_sorted.drop(['mol', 'atomNum', 'smiles path', 'ligand', 'ligand path'], axis=1, inplace=True)

    #csvファイルを作成したい
    filename = fname + "ver" + str(ver) + "_rep_fs.csv"
    place = filepath + 'to_pdf/csvfile/' + filename
    rep_sorted.to_csv(place)

    #######変更部分#######
    ##実際にpdfとして出力
    version = 'v' + str(ver)
    scatter_name = filepath + 'to_pdf/img/' + fname + '_scatter.png'
    output_filename = filepath + "to_pdf/pdffile/" + fname+ "ver" + str(ver) + "_rep_fs.pdf"


    # htmlの情報　これに分子の情報を書き込んでいくことでpdfを作成
    html_template = """
    <!doctype html>
    <html lang="ja">
    <head>
        <meta charset="utf-8">
        <title>Fragment Set """ + version + """</title>
        <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css">
    </head>
    <body>
        <h1>Fragment Set """ + version + """</h1>
        <script src="https://code.jquery.com/jquery-3.2.1.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js"></script>
        <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js"></script>
        <div class="container">
            {table}
        </div>
        <div class="scatter">
        <img src=""" + scatter_name +  """  width="1000" alt="img" />
        </div>

    </body>
    </html>
    """


    df = pd.read_csv(filepath + 'to_pdf/csvfile/' + fname + "ver" + str(ver) + "_rep_fs.csv")
    df.index = [version + '-' + str(i+1) for i in range(len(df))]
    df.drop('Unnamed: 0', axis=1, inplace=True)

    table = df.to_html(classes=["table", "table-bordered", "table-hover"], escape=False)
    html_str = html_template.format(table=table)

    with open(filepath + 'to_pdf/htmlfile/' + fname + "ver" + str(ver) + "_rep_fs.html", "w") as f:
        f.write(html_str)


    with open(output_filename, "w+b") as output_file:
        pisa_status = pisa.CreatePDF(src=html_str, dest=output_file)
    
    return