"""
Created on Jun 15, 2021
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def scale_ndarray(inputs):
    scaler = StandardScaler()
    scaler.fit(inputs)
    inputs = scaler.transform(inputs)
    return inputs

def gen_reactions_feature(grid_spacing, whose_grid,excluded):
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    source_root = root_path+'/data/sphere_based/grid_'+whose_grid+\
                  '/gridspacing'+str(grid_spacing)+'/'
    excluded_str = '-'.join(excluded) if len(excluded) <= 6 else '-'.join(excluded[:6])+'-and-rest-'+str((len(excluded)-6))
    output_path = root_path+'/data/select_features/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    df_reagents_1 = pd.read_csv(source_root+'Reagent_1.csv')
    df_reagents_2 = pd.read_csv(source_root+'Reagent_2.csv')
    df_catalysts = pd.read_csv(source_root+'Catalyst_aligned.csv')
    df_reactions = pd.read_csv(root_path+'/data/mol_files/original/19_science_reactions.csv')

    reagents_1_homo = []
    reagents_1_lumo = []
    reagents_1_dipole = []
    reagents_1_volume = []
    reagents_1_single_energy = []
    reagents_1_mulliken_charge = []
    reagents_1_NAO_charge = []
    reagents_1_NMR_Shielding = []
    reagents_2_homo = []
    reagents_2_lumo = []
    reagents_2_dipole = []
    reagents_2_volume = []
    reagents_2_single_energy = []
    reagents_2_mulliken_charge = []
    reagents_2_NAO_charge = []
    reagents_2_NMR_Shielding = []
    catalysts_homo = []
    catalysts_lumo = []
    catalysts_dipole = []
    catalysts_volume = []
    catalysts_single_energy = []
    catalysts_mulliken_charge = []
    catalysts_NAO_charge = []
    catalysts_NMR_Shielding = []
    catalysts_total_valence = []
    catalysts_aromatic = []
    catalysts_hybridization = []
    catalysts_degree = []
    catalysts_formal_charge = []
    ee_values = []
    delta_delta_Gs = []
    for row_ind in range(len(df_reactions)):
        reagent_1 = df_reactions.iloc[row_ind,0]
        reagent_2 = df_reactions.iloc[row_ind,1]
        catalyst = df_reactions.iloc[row_ind,2]
        ee_value = df_reactions.iloc[row_ind,5]
        delta_delta_G = df_reactions.iloc[row_ind,7]
        homo_energy = df_reagents_1[df_reagents_1['name']==reagent_1].values[0,2]
        lumo_energy = df_reagents_1[df_reagents_1['name']==reagent_1].values[0,3]
        dipole = df_reagents_1[df_reagents_1['name']==reagent_1].values[0,4]
        volume = df_reagents_1[df_reagents_1['name']==reagent_1].values[0,5]
        single_energy = df_reagents_1[df_reagents_1['name']==reagent_1].values[0,6]
        reagents_1_homo.append([homo_energy])
        reagents_1_lumo.append([lumo_energy])
        reagents_1_dipole.append([dipole])
        reagents_1_volume.append([volume])
        reagents_1_single_energy.append([single_energy])
        reagent_1_atom_feature = df_reagents_1[df_reagents_1['name']==reagent_1].values[0,1]
        reagent_1_atom_feature = reagent_1_atom_feature.split(',')
        reagent_1_mulliken_charge = reagent_1_atom_feature[0:len(reagent_1_atom_feature):3]
        reagent_1_NAO_charge = reagent_1_atom_feature[1:len(reagent_1_atom_feature):3]
        reagent_1_NMR_Shielding = reagent_1_atom_feature[2:len(reagent_1_atom_feature):3]
        reagents_1_mulliken_charge.append(reagent_1_mulliken_charge)
        reagents_1_NAO_charge.append(reagent_1_NAO_charge)
        reagents_1_NMR_Shielding.append(reagent_1_NMR_Shielding)
        homo_energy = df_reagents_2[df_reagents_2['name']==reagent_2].values[0,2]
        lumo_energy = df_reagents_2[df_reagents_2['name']==reagent_2].values[0,3]
        dipole = df_reagents_2[df_reagents_2['name']==reagent_2].values[0,4]
        volume = df_reagents_2[df_reagents_2['name']==reagent_2].values[0,5]
        single_energy = df_reagents_2[df_reagents_2['name']==reagent_2].values[0,6]
        reagents_2_homo.append([homo_energy])
        reagents_2_lumo.append([lumo_energy])
        reagents_2_dipole.append([dipole])
        reagents_2_volume.append([volume])
        reagents_2_single_energy.append([single_energy])
        reagent_2_atom_feature = df_reagents_2[df_reagents_2['name']==reagent_2].values[0,1]
        reagent_2_atom_feature = reagent_2_atom_feature.split(',')
        reagent_2_mulliken_charge = reagent_2_atom_feature[0:len(reagent_2_atom_feature):3]
        reagent_2_NAO_charge = reagent_2_atom_feature[1:len(reagent_2_atom_feature):3]
        reagent_2_NMR_Shielding = reagent_2_atom_feature[2:len(reagent_2_atom_feature):3]
        reagents_2_mulliken_charge.append(reagent_2_mulliken_charge)
        reagents_2_NAO_charge.append(reagent_2_NAO_charge)
        reagents_2_NMR_Shielding.append(reagent_2_NMR_Shielding)
        homo_energy = df_catalysts[df_catalysts['name']==catalyst].values[0,2]
        lumo_energy = df_catalysts[df_catalysts['name']==catalyst].values[0,3]
        dipole = df_catalysts[df_catalysts['name']==catalyst].values[0,4]
        volume = df_catalysts[df_catalysts['name']==catalyst].values[0,5]
        single_energy = df_catalysts[df_catalysts['name']==catalyst].values[0,6]
        catalysts_homo.append([homo_energy])
        catalysts_lumo.append([lumo_energy])
        catalysts_dipole.append([dipole])
        catalysts_volume.append([volume])
        catalysts_single_energy.append([single_energy])
        catalyst_atom_feature = df_catalysts[df_catalysts['name']==catalyst].values[0,1]
        catalyst_atom_feature = catalyst_atom_feature.split(',')
        catalyst_mulliken_charge = catalyst_atom_feature[0:len(catalyst_atom_feature):8]
        catalyst_NAO_charge = catalyst_atom_feature[1:len(catalyst_atom_feature):8]
        catalyst_NMR_Shielding = catalyst_atom_feature[2:len(catalyst_atom_feature):8]
        catalyst_total_valence = catalyst_atom_feature[3:len(catalyst_atom_feature):8]
        catalyst_aromatic = catalyst_atom_feature[4:len(catalyst_atom_feature):8]
        catalyst_hybridization = catalyst_atom_feature[5:len(catalyst_atom_feature):8]
        catalyst_degree = catalyst_atom_feature[6:len(catalyst_atom_feature):8]
        catalyst_formal_charge = catalyst_atom_feature[7:len(catalyst_atom_feature):8]
        catalysts_mulliken_charge.append(catalyst_mulliken_charge)
        catalysts_NAO_charge.append(catalyst_NAO_charge)
        catalysts_NMR_Shielding.append(catalyst_NMR_Shielding)
        catalysts_total_valence.append(catalyst_total_valence)
        catalysts_aromatic.append(catalyst_aromatic)
        catalysts_hybridization.append(catalyst_hybridization)
        catalysts_degree.append(catalyst_degree)
        catalysts_formal_charge.append(catalyst_formal_charge)
        ee_values.append([ee_value])
    reagents_1_homo = scale_ndarray(np.array(reagents_1_homo))
    reagents_1_lumo = scale_ndarray(np.array(reagents_1_lumo))
    reagents_1_dipole = scale_ndarray(np.array(reagents_1_dipole))
    reagents_1_volume = scale_ndarray(np.array(reagents_1_volume))
    reagents_1_single_energy = scale_ndarray(np.array(reagents_1_single_energy))
    reagents_1_mulliken_charge = scale_ndarray(np.array(reagents_1_mulliken_charge))
    reagents_1_NAO_charge = scale_ndarray(np.array(reagents_1_NAO_charge))
    reagents_1_NMR_Shielding = scale_ndarray(np.array(reagents_1_NMR_Shielding))
    reagents_2_homo = scale_ndarray(np.array(reagents_2_homo))
    reagents_2_lumo = scale_ndarray(np.array(reagents_2_lumo))
    reagents_2_dipole = scale_ndarray(np.array(reagents_2_dipole))
    reagents_2_volume = scale_ndarray(np.array(reagents_2_volume))
    reagents_2_single_energy = scale_ndarray(np.array(reagents_2_single_energy))
    reagents_2_mulliken_charge = scale_ndarray(np.array(reagents_2_mulliken_charge))
    reagents_2_NAO_charge = scale_ndarray(np.array(reagents_2_NAO_charge))
    reagents_2_NMR_Shielding = scale_ndarray(np.array(reagents_2_NMR_Shielding))
    catalysts_homo = scale_ndarray(np.array(catalysts_homo))
    catalysts_lumo = scale_ndarray(np.array(catalysts_lumo))
    catalysts_dipole = scale_ndarray(np.array(catalysts_dipole))
    catalysts_volume = scale_ndarray(np.array(catalysts_volume))
    catalysts_single_energy = scale_ndarray(np.array(catalysts_single_energy))
    catalysts_mulliken_charge = scale_ndarray(np.array(catalysts_mulliken_charge))
    catalysts_NAO_charge = scale_ndarray(np.array(catalysts_NAO_charge))
    catalysts_NMR_Shielding = scale_ndarray(np.array(catalysts_NMR_Shielding))
    catalysts_total_valence = scale_ndarray(np.array(catalysts_total_valence))
    catalysts_aromatic = scale_ndarray(np.array(catalysts_aromatic))
    catalysts_hybridization = scale_ndarray(np.array(catalysts_hybridization))
    catalysts_degree = scale_ndarray(np.array(catalysts_degree))
    catalysts_formal_charge = scale_ndarray(np.array(catalysts_formal_charge))
    df_reagents_1_homo = pd.DataFrame(reagents_1_homo)
    df_reagents_1_homo.to_csv(output_path+'reagents_1_homo_for_ml.csv',index=False,header=None)
    df_reagents_1_lumo = pd.DataFrame(reagents_1_lumo)
    df_reagents_1_lumo.to_csv(output_path+'reagents_1_lumo_for_ml.csv',index=False,header=None)
    df_reagents_1_dipole = pd.DataFrame(reagents_1_dipole)
    df_reagents_1_dipole.to_csv(output_path+'reagents_1_dipole_for_ml.csv',index=False,header=None)
    df_reagents_1_volume = pd.DataFrame(reagents_1_volume)
    df_reagents_1_volume.to_csv(output_path+'reagents_1_volume_for_ml.csv',index=False,header=None)
    df_reagents_1_single_energy = pd.DataFrame(reagents_1_single_energy)
    df_reagents_1_single_energy.to_csv(output_path+'reagents_1_single_energy_for_ml.csv',index=False,header=None)
    df_reagents_1_mulliken_charge = pd.DataFrame(reagents_1_mulliken_charge)
    df_reagents_1_mulliken_charge.to_csv(output_path+'reagents_1_mulliken_charge_for_ml.csv',index=False,header=None)
    df_reagents_1_NAO_charge = pd.DataFrame(reagents_1_NAO_charge)
    df_reagents_1_NAO_charge.to_csv(output_path+'reagents_1_NAO_charge_for_ml.csv',index=False,header=None)
    df_reagents_1_NMR_Shielding = pd.DataFrame(reagents_1_NMR_Shielding)
    df_reagents_1_NMR_Shielding.to_csv(output_path+'reagents_1_NMR_Shielding_for_ml.csv',index=False,header=None)
    df_reagents_2_homo = pd.DataFrame(reagents_2_homo)
    df_reagents_2_homo.to_csv(output_path+'reagents_2_homo_for_ml.csv',index=False,header=None)
    df_reagents_2_lumo = pd.DataFrame(reagents_2_lumo)
    df_reagents_2_lumo.to_csv(output_path+'reagents_2_lumo_for_ml.csv',index=False,header=None)
    df_reagents_2_dipole = pd.DataFrame(reagents_2_dipole)
    df_reagents_2_dipole.to_csv(output_path+'reagents_2_dipole_for_ml.csv',index=False,header=None)
    df_reagents_2_volume = pd.DataFrame(reagents_2_volume)
    df_reagents_2_volume.to_csv(output_path+'reagents_2_volume_for_ml.csv',index=False,header=None)
    df_reagents_2_single_energy = pd.DataFrame(reagents_2_single_energy)
    df_reagents_2_single_energy.to_csv(output_path+'reagents_2_single_energy_for_ml.csv',index=False,header=None)
    df_reagents_2_mulliken_charge = pd.DataFrame(reagents_2_mulliken_charge)
    df_reagents_2_mulliken_charge.to_csv(output_path+'reagents_2_mulliken_charge_for_ml.csv',index=False,header=None)
    df_reagents_2_NAO_charge = pd.DataFrame(reagents_2_NAO_charge)
    df_reagents_2_NAO_charge.to_csv(output_path+'reagents_2_NAO_charge_for_ml.csv',index=False,header=None)
    df_reagents_2_NMR_Shielding = pd.DataFrame(reagents_2_NMR_Shielding)
    df_reagents_2_NMR_Shielding.to_csv(output_path+'reagents_2_NMR_Shielding_for_ml.csv',index=False,header=None)
    df_catalysts_homo = pd.DataFrame(catalysts_homo)
    df_catalysts_homo.to_csv(output_path+'catalysts_homo_for_ml.csv',index=False,header=None)
    df_catalysts_lumo = pd.DataFrame(catalysts_lumo)
    df_catalysts_lumo.to_csv(output_path+'catalysts_lumo_for_ml.csv',index=False,header=None)
    df_catalysts_dipole = pd.DataFrame(catalysts_dipole)
    df_catalysts_dipole.to_csv(output_path+'catalysts_dipole_for_ml.csv',index=False,header=None)
    df_catalysts_volume = pd.DataFrame(catalysts_volume)
    df_catalysts_volume.to_csv(output_path+'catalysts_volume_for_ml.csv',index=False,header=None)
    df_catalysts_single_energy = pd.DataFrame(catalysts_single_energy)
    df_catalysts_single_energy.to_csv(output_path+'catalysts_single_energy_for_ml.csv',index=False,header=None)
    df_catalysts_mulliken_charge = pd.DataFrame(catalysts_mulliken_charge)
    df_catalysts_mulliken_charge.to_csv(output_path+'catalysts_mulliken_charge_for_ml.csv',index=False,header=None)
    df_catalysts_NAO_charge = pd.DataFrame(catalysts_NAO_charge)
    df_catalysts_NAO_charge.to_csv(output_path+'catalysts_NAO_charge_for_ml.csv',index=False,header=None)
    df_catalysts_NMR_Shielding = pd.DataFrame(catalysts_NMR_Shielding)
    df_catalysts_NMR_Shielding.to_csv(output_path+'catalysts_NMR_Shielding_for_ml.csv',index=False,header=None)
    ee_values = np.array(ee_values)
    df_ee_values = pd.DataFrame(ee_values)
    df_ee_values.to_csv(output_path+'ee_values_for_ml.csv',index=False,header=None)
    reaction_features = np.concatenate((reagents_1_homo,reagents_1_lumo),axis=1)
    if not 'reagents_1_dipole' in excluded:
        reaction_features = np.concatenate((reaction_features,reagents_1_dipole),axis=1)
    if not 'reagents_1_volume' in excluded:
        reaction_features = np.concatenate((reaction_features,reagents_1_volume),axis=1)
    if not 'reagents_1_single_energy' in excluded:
        reaction_features = np.concatenate((reaction_features,reagents_1_single_energy),axis=1)
    if not 'reagents_1_mulliken_charge' in excluded:
        reaction_features = np.concatenate((reaction_features,reagents_1_mulliken_charge),axis=1)
    if not 'reagents_1_NAO_charge' in excluded:
        reaction_features = np.concatenate((reaction_features,reagents_1_NAO_charge),axis=1)
    if not 'reagents_1_NMR_Shielding' in excluded:
        reaction_features = np.concatenate((reaction_features,reagents_1_NMR_Shielding),axis=1)
    if not 'reagents_2_homo' in excluded:
        reaction_features = np.concatenate((reaction_features,reagents_2_homo),axis=1)
    if not 'reagents_2_lumo' in excluded:
        reaction_features = np.concatenate((reaction_features,reagents_2_lumo),axis=1)
    if not 'reagents_2_dipole' in excluded:
        reaction_features = np.concatenate((reaction_features,reagents_2_dipole),axis=1)
    if not 'reagents_2_volume' in excluded:
        reaction_features = np.concatenate((reaction_features,reagents_2_volume),axis=1)
    if not 'reagents_2_single_energy' in excluded:
        reaction_features = np.concatenate((reaction_features,reagents_2_single_energy),axis=1)
    if not 'reagents_2_mulliken_charge' in excluded:
        reaction_features = np.concatenate((reaction_features,reagents_2_mulliken_charge),axis=1)
    if not 'reagents_2_NAO_charge' in excluded:
        reaction_features = np.concatenate((reaction_features,reagents_2_NAO_charge),axis=1)
    if not 'reagents_2_NMR_Shielding' in excluded:
        reaction_features = np.concatenate((reaction_features,reagents_2_NMR_Shielding),axis=1)
    if not 'catalysts_homo' in excluded:
        reaction_features = np.concatenate((reaction_features,catalysts_homo),axis=1)
    if not 'catalysts_lumo' in excluded:
        reaction_features = np.concatenate((reaction_features,catalysts_lumo),axis=1)
    if not 'catalysts_dipole' in excluded:
        reaction_features = np.concatenate((reaction_features,catalysts_dipole),axis=1)
    if not 'catalysts_volume' in excluded:
        reaction_features = np.concatenate((reaction_features,catalysts_volume),axis=1)
    if not 'catalysts_single_energy' in excluded:
        reaction_features = np.concatenate((reaction_features,catalysts_single_energy),axis=1)
    if not 'catalysts_mulliken_charge' in excluded:
        reaction_features = np.concatenate((reaction_features,catalysts_mulliken_charge),axis=1)
    if not 'catalysts_NAO_charge' in excluded:
        reaction_features = np.concatenate((reaction_features,catalysts_NAO_charge),axis=1)
    if not 'catalysts_NMR_Shielding' in excluded:
        reaction_features = np.concatenate((reaction_features,catalysts_NMR_Shielding),axis=1)
    reaction_features = np.concatenate((reaction_features,ee_values),axis=1)
    df_reaction_features = pd.DataFrame(reaction_features)
    df_reaction_features.to_csv(output_path+'features_total_by_sphere.csv',index=False,header=None)
