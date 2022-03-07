"""
Created on Jun 15, 2021
"""
import sys 
import os
from molecule_parser import MoleculeParser
from copy import deepcopy
from constructor import GridConstructor
    
def gen_molecules_feature(grid_spacing, whose_grid='own'):
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))   
    print('root_path: ',root_path) 
    out_root = root_path+'/data/sphere_based/grid_'+whose_grid+'/gridspacing'+str(grid_spacing)+'/'
    
    data_root = root_path+'/data/mol_files/original/'   
    if not os.path.exists(out_root):
        os.makedirs(out_root)
    molParser = MoleculeParser()
    mol_source_root = root_path+'/data/mol_files/babel_mol_files/'
    catalysts_aligned = molParser.parse_mols_from_csv(data_root+'Catalyst_aligned.csv',mol_source_root)
    reagents_1 = molParser.parse_mols_from_csv(data_root+'Reagent_1.csv',mol_source_root)
    reagents_2 = molParser.parse_mols_from_csv(data_root+'Reagent_2.csv',mol_source_root)

    all_mols = deepcopy(catalysts_aligned)
    all_mols.extend(reagents_1)
    all_mols.extend(reagents_2)
    grid_constructor = GridConstructor(catalysts_aligned,spacing=grid_spacing)
    grid_constructor.generate_grid()
    grid_constructor.set_molecules_grid(catalysts_aligned)
    grid_constructor_of_reagents_1 = GridConstructor(reagents_1,spacing=grid_spacing)
    grid_constructor_of_reagents_1.generate_grid()
    grid_constructor_of_reagents_1.set_molecules_grid(reagents_1)
    grid_constructor_of_reagents_2 = GridConstructor(reagents_2,spacing=grid_spacing)
    grid_constructor_of_reagents_2.generate_grid()
    grid_constructor_of_reagents_2.set_molecules_grid(reagents_2)
    column_names=['name','Catalyst_Atom_feature','Catalyst_HOMO_Energy','Catalyst_LUMO_Energy',\
                                                       'Catalyst_Dipole','Catalyst_Volume','Catalyst_Single_Energy']
    molParser.save_mols_to_csv(catalysts_aligned, column_names, out_root+'Catalyst_aligned.csv')
    column_names=['name','Reagent_1_Atom_feature','Reagent_1_HOMO_Energy','Reagent_1_LUMO_Energy',\
                                                       'Reagent_1_Dipole','Reagent_1_Volume','Reagent_1_Single_Energy']
    molParser.save_mols_to_csv(reagents_1, column_names, out_root+'Reagent_1.csv')
    column_names=['name','Reagent_2_Atom_feature','Reagent_2_HOMO_Energy','Reagent_2_LUMO_Energy',\
                                                       'Reagent_2_Dipole','Reagent_2_Volume','Reagent_2_Single_Energy']
    molParser.save_mols_to_csv(reagents_2, column_names, out_root+'Reagent_2.csv')
