import os

import numpy as np
import pandas as pd
from rdkit import Chem

import datatypes as dt

class MoleculeParser(object):
    """Analyze the original document of molecular properties."""
    def __init__(self):
        self.mols = {}
        self.parsed = False
    
    def parse_mols_from_csv(self, csv_filename, mol_source_root):
        '''get mols from csv
        Parameters:
            csv_filename - CSV file path
            mol_source_root - path of the Mol file
        '''
        mols = []
        df = pd.read_csv(csv_filename)
        for i in range(len(df)):
            name = df.iloc[i,0]
            mol_file = os.path.join(mol_source_root,str(name)+'.mol')
            rdkit_mol = Chem.MolFromMolFile(mol_file, removeHs=False)
            rdkit_atoms = rdkit_mol.GetAtoms()
            mol = dt.Molecule(name)
            atoms = df.iloc[i,1].split(',')
            coords = df.iloc[i,2].split(',')
            mulliken_charges = df.iloc[i,3].split(',')
            NAO_charges = df.iloc[i,4].split(',')
            NMR_Shieldings = df.iloc[i,5].split(',')
            for atom_id in range(len(atoms)):
                atom = dt.Atom(atom_id, str(atom_id))
                atom.coord = dt.Point(atom_id,'coord_'+str(atom_id),coords[atom_id*3],\
                                      coords[atom_id*3+1],coords[atom_id*3+2])
                atom.mulliken_charge = mulliken_charges[atom_id]
                atom.NAO_charge = NAO_charges[atom_id]
                atom.NMR_Shielding = NMR_Shieldings[atom_id]

                atom.explicit_valence = rdkit_atoms[atom_id].GetExplicitValence()
                atom.implicit_valence = rdkit_atoms[atom_id].GetImplicitValence()
                atom.total_valence = rdkit_atoms[atom_id].GetTotalValence()
                atom.aromatic = rdkit_atoms[atom_id].GetIsAromatic()
                atom.hybridization = rdkit_atoms[atom_id].GetHybridization()
                atom.degree = rdkit_atoms[atom_id].GetDegree()
                atom.total_degree = rdkit_atoms[atom_id].GetTotalDegree()
                atom.formal_charge = rdkit_atoms[atom_id].GetFormalCharge()
                mol.add_atom(atom)

            mol.homo_energy = df.iloc[i,6]
            mol.lumo_energy = df.iloc[i,7]
            mol.dipole = df.iloc[i,8]
            mol.volume = df.iloc[i,9]
            mol.single_energy = df.iloc[i,10]
            mols.append(mol)
        return mols

    def save_mols_to_csv(self,mols,column_names,out_csv_filename):
        """
        Save the features of molecules to a CSV file
        Parameters:
            mols - a list containing molecules
            column_names - column names for CSV
            out_csv_filename - the name of the output CSV
        """
        datas = []
        for mol in mols:
            row = []
            mol.calculate_gridpoint_descriptors()
            mulliken_charges,nao_charges,nmr_shielding,all_atomfeatures_of_molecule = \
                mol.get_result_descriptors()
            row.append(mol.label)
            row.append(all_atomfeatures_of_molecule)
            row.append(mol.homo_energy)
            row.append(mol.lumo_energy)
            row.append(mol.dipole)
            row.append(mol.volume)
            row.append(mol.single_energy)
            datas.append(row)
        #save to a CSV file
        df = pd.DataFrame(datas,columns=column_names)
        df.to_csv(out_csv_filename,index=False)
