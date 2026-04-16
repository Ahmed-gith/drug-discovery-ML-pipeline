
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='ticks')



df = pd.read_csv('bioactivity_data_eda.csv')



def smiles_to_morgan_fingerprint(smiles, radius=2, nBits=2048):
    """
    Convert one SMILES string to a Morgan fingerprint.
    Returns a list of 2048 numbers (0s and 1s).
    Returns None if the SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        return None
    

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)

    arr = np.zeros((nBits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    
    return arr.tolist()




fingerprints = []    
valid_indices = []   

for idx, row in df.iterrows():

    
    fp = smiles_to_morgan_fingerprint(row['canonical_smiles'])
    
    if fp is not None:
        fingerprints.append(fp)
        valid_indices.append(idx)
    else:
        print(f"  Warning: Could not process {row['molecule_chembl_id']} - skipping")







column_names = [f'FP{i}' for i in range(2048)]


X = pd.DataFrame(fingerprints, columns=column_names)

df_valid = df.loc[valid_indices].reset_index(drop=True)
X = X.reset_index(drop=True)



Y = df_valid['pIC50']




variances = X.var()


X_filtered = X.loc[:, variances > 0]




dataset = pd.concat([X_filtered, Y], axis=1)





active_fps   = X_filtered[df_valid.bioactivity_class == 'active'].mean()
inactive_fps = X_filtered[df_valid.bioactivity_class == 'inactive'].mean()

plt.figure(figsize=(12, 4))
plt.plot(active_fps[:50].values,   label='Active',   alpha=0.7, color='blue')
plt.plot(inactive_fps[:50].values, label='Inactive', alpha=0.7, color='red')
plt.xlabel('Fingerprint Bit Position', fontsize=12)
plt.ylabel('Mean Value (0-1)', fontsize=12)
plt.title('Average Fingerprint Pattern: Active vs Inactive Compounds\n(First 50 bits)')
plt.legend()
plt.tight_layout()
plt.savefig('plot_fingerprint_patterns.png')
plt.show()




plt.figure(figsize=(7, 4))
sns.histplot(
    data=df_valid,
    x='pIC50',
    hue='bioactivity_class',  
    bins=30,
    kde=True   
)
plt.xlabel('pIC50', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Distribution of pIC50 Values by Bioactivity Class')
plt.tight_layout()
plt.savefig('plot_pIC50_distribution.png')
plt.show()




X_filtered.to_csv('X_features.csv', index=False)

Y.to_csv('Y_target.csv', index=False)

dataset.to_csv('dataset_final.csv', index=False)


save_cols = [col for col in ['molecule_cheml_id', 'bioactivity_class', 'pIC50'] 
             if col in df_valid.columns]


df_valid[save_cols].to_csv('compound_info.csv', index=False)

