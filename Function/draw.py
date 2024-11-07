from rdkit.Chem import Draw

def mol_to_svg(mols, filename):
    """分子一つ一つをpngファイルで保存"""
    if mols is not None:
        img = Draw.MolsToGridImage(mols, molsPerRow=1, subImgSize=(400, 400))
        with open(filename, mode='wb') as f:
            f.write(img.data) # type: ignore (Pylance, img.data)

def sdf_to_svg(li, num_start, num_end, filename, filepath):
    img = Draw.MolsToGridImage(li[num_start:num_end], maxMols=1870, molsPerRow=5, subImgSize=(500, 500), useSVG=True)
    place = filepath + filename
    with open(place, mode='w') as f:
        f.write(img.data) # type: ignore (Pylance, img.data)

def sdf_to_svg1(li, num_start, num_end, filename, filepath):
    img = Draw.MolsToGridImage(li[num_start:num_end], maxMols=3000, molsPerRow=2, subImgSize=(500, 500), useSVG=True)
    place = filepath + filename
    with open(place, mode='w') as f:
        f.write(img.data) # type: ignore (Pylance, img.data)
        
def sdf_to_svg2(li, num_start, num_end, filename, legends, filepath):
    img = Draw.MolsToGridImage(li[num_start:num_end], maxMols=20, molsPerRow=3, subImgSize=(700, 700), useSVG=True)
    place = filepath + filename
    with open(place, mode='w') as f:
        f.write(img.data) # type: ignore (Pylance, img.data)

def list_to_svg(li, filename, filepath):
    """代表フラグメントを描画したい"""
    img = Draw.MolsToGridImage(li, maxMols=1000, molsPerRow=5, subImgSize=(500, 500), useSVG=True)
    place = filepath+ filename
    with open(place, mode='w') as f:
        f.write(img.data) # type: ignore (Pylance, img.data)