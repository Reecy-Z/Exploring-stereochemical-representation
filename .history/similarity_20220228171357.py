import os
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
from rdkit import rdBase, Chem, DataStructs
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.AtomPairs import Pairs, Torsions
from rdkit.Chem.Draw import SimilarityMaps

def complex(DIR,save,title):
    mol_total = []
    for f in os.listdir(DIR):
        print(f)
        suppl = Chem.SDMolSupplier(DIR+f)
        mols = [x for x in suppl if x is not None]
        mol_total.append(mols[0])
    maccs_fps = [AllChem.GetMACCSKeysFingerprint(mol) for mol in mol_total]
    maccs_total = []
    for i in range(len(mol_total)):
        maccs = DataStructs.BulkTanimotoSimilarity(maccs_fps[i], maccs_fps[0:])
        maccs_total.append(maccs)
    maccs_total = np.array(maccs_total)
    np.savetxt(OUT + save[0] ,maccs_total,delimiter=',')
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    score=maccs_total
    name=np.arange(len(mol_total))
    col=np.arange(len(mol_total))
    im=plt.imshow(score,cmap='viridis')
    # ax.xaxis.set_ticks_position('top')
    ax.set_xticks(np.arange(len(mol_total)))
    ax.set_yticks(np.arange(len(mol_total)))
    # ax.set_xticklabels(col)
    # ax.set_yticklabels(name)
    plt.tick_params(axis='x',colors='white')
    plt.tick_params(axis='y',colors='white')
    cb = fig.colorbar(im,pad=0.03)
    # cb.set_label('ee%',fontsize=16)
    cb.set_ticks(np.arange(0,1.2,0.2))
    cb.set_ticklabels(('0', '0.2', '0.4', '0.6', '0.8', '1'))
    ax.set_title(title,fontsize=16, fontweight='bold', ha='center', va='center')
    plt.savefig(OUT + save[1], dpi = 300)

def score(file):
    data = pd.read_csv(file,header=None)
    data = np.array(data)
    score = (np.sum(data)-len(data))/(len(data)**2-len(data))
    return score

def spider_plots(labels,dataLenth,data,title,OUT):
    colors=list(mcolors.TABLEAU_COLORS.keys())
    angles = np.linspace(0, 2*np.pi, dataLenth, endpoint=False)
    data = np.concatenate((data, [data[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    fig = plt.figure(figsize=(15,15))
    plt.style.use('ggplot')
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, data, '-', linewidth=1,color = 'white')
    ax.fill(angles, data, facecolor=mcolors.TABLEAU_COLORS[colors[3]], alpha=0.7)
    ax.set_thetagrids(angles * 180/np.pi, labels,size = 30,color=mcolors.TABLEAU_COLORS[colors[7]], fontweight='bold')
    ax.set_title(title,fontsize = 40, fontweight='bold',y = 1.1)
    ax.set_rlim(0,1)
    ax.set_rgrids([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],fontsize = 10,alpha =0)
    ax.grid(linestyle='--')
    ax.spines['polar'].set_visible(False)
    plt.savefig(OUT + 'COMPLEX.png', dpi = 300)

OUT = './dataset_similarity/'
titles = ['BHA',
          'SMC',
          'ANSA']

complex('./molecule sdf/BHA/additive/',
        ['BHA\\additive.csv','BHA\\additive.png'],
        'BHA')

complex('./molecule sdf/BHA/additive/',
        ['BHA\\aryl_halide.csv','BHA\\aryl_halide.png'],
        'BHA')

complex('./molecule sdf/BHA/additive/',
        ['BHA\\base.csv','BHA\\base.png'],
        'BHA')

complex('./molecule sdf/BHA/additive/',
        ['BHA\\ligand.csv','BHA\\ligand.png'],
        'BHA')

score_total = []
score_1 = score(OUT + '20 pnas\\catalyst.csv')
score_total.append(score_1)
score_2 = score(OUT + '20 pnas\\imine.csv')
score_total.append(score_2)
score_3 = score(OUT + '20 pnas\\olefin.csv')
score_total.append(score_3)
score_4 = score(OUT + '20 pnas\\solvent.csv')
score_total.append(score_4)

print(score_total)

spider_plots(['catalyst\n(55.88)','imine\n(51.03)','olefin\n(39.94)','solvent\n(12.73)'],
              4,
              score_total,
              titles[0],
              'D:\\analysis\\data\\dataset complex\\20 pnas\\')

