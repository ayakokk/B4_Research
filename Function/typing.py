from typing import TypeVar
from rdkit.Chem.rdchem import Mol

rdkitMol = TypeVar('rdkitMol', bound=Mol)