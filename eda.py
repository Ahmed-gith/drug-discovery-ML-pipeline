import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski


sns.set(style='ticks')


df = pd.read_csv('bioactivity_data_updates.csv')



def calculate_lipinski(smiles_list):
    

    molecules = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        molecules.append(mol)
    

    results = []
    
    for mol in molecules:
        if mol is None:
 
            results.append([None, None, None, None])
            continue
        
  
        mw    = Descriptors.MolWt(mol)
        logp  = Descriptors.MolLogP(mol)
        hdon  = Lipinski.NumHDonors(mol)
        hacc  = Lipinski.NumHAcceptors(mol)
        
        results.append([mw, logp, hdon, hacc])
    

    descriptors_df = pd.DataFrame(
        results,
        columns=['MW', 'LogP', 'NumHDonors', 'NumHAcceptors']
    )
    
    return descriptors_df


df_lipinski = calculate_lipinski(df.canonical_smiles)





df_combined = pd.concat([df , df_lipinski], axis=1)


def normalize_ic50(df_input):
    normalized = []
    
    for value in df_input['standard_value']:
        if float(value) > 100000000:
            value = 100000000    
        normalized.append(float(value))
    

    df_input['standard_value_norm'] = normalized
    df_output = df_input.drop('standard_value', axis=1)

    
    return df_output

df_norm = normalize_ic50(df_combined)



def convert_to_pIC50(df_input):
    pic50_values = []
    
    for value in df_input['standard_value_norm']:
        molar = value * (10 ** -9)    
        pic50 = -np.log10(molar)          
        pic50_values.append(pic50)
    
    df_input['pIC50'] = pic50_values
    df_output = df_input.drop('standard_value_norm', axis=1)
    
    return df_output

df_final = convert_to_pIC50(df_norm)




df_2class = df_final[df_final.bioactivity_class != 'intermediate']
df_2class = df_2class.reset_index(drop=True)




plt.figure(figsize=(5.5 , 5.5))

sns.countplot(
    x = 'bioactivity_class',
    data = df_2class,
    edgecolor='black'
)

plt.xlabel('Bioactivity Class', fontsize=14, fontweight='bold')
plt.ylabel('Frequency', fontsize=14, fontweight='bold')
plt.title('Active vs Inactive Compounds')
plt.tight_layout()          
plt.savefig('plot_bioactivity_class.png')
\





df_plot = df_2class.replace([np.inf, -np.inf], np.nan).dropna(subset=['pIC50', 'MW', 'LogP'])

plt.figure(figsize=(5.5, 5.5))

sns.scatterplot(
    x='MW',
    y='LogP',
    data=df_plot,
    hue='bioactivity_class',
    edgecolor='black',
    alpha=0.7
 
)

plt.xlabel('Molecular Weight (MW)', fontsize=14, fontweight='bold')
plt.ylabel('LogP (fat solubility)', fontsize=14, fontweight='bold')
plt.title('Chemical Space: MW vs LogP')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
plt.tight_layout()
plt.savefig('plot_MW_vs_LogP.png')






descriptors_to_plot = ['pIC50', 'MW', 'LogP', 'NumHDonors', 'NumHAcceptors']

for descriptor in descriptors_to_plot:
    plt.figure(figsize=(5.5, 5.5))
    
    sns.boxplot(
        x='bioactivity_class',
        y=descriptor,
        data=df_2class
    )
    
    plt.xlabel('Bioactivity Class', fontsize=14, fontweight='bold')
    plt.ylabel(descriptor, fontsize=14, fontweight='bold')
    plt.title(f'{descriptor}: Active vs Inactive')
    plt.tight_layout()
    plt.savefig(f'plot_{descriptor}.png')






def run_mannwhitney(descriptor):
    active   = df_2class[df_2class.bioactivity_class == 'active'][descriptor]
    inactive = df_2class[df_2class.bioactivity_class == 'inactive'][descriptor]
    

    stat, p = mannwhitneyu(active, inactive)
    
    alpha = 0.05  
    if p > alpha:
        interpretation = 'Same distribution (NOT significant)'
    else:
        interpretation = 'Different distribution (SIGNIFICANT ✅)'
    

    
    return {'Descriptor': descriptor, 'p-value': p, 'Result': interpretation}



all_results = []
for descriptor in descriptors_to_plot:
    result = run_mannwhitney(descriptor)
    all_results.append(result)

results_df = pd.DataFrame(all_results)
results_df.to_csv('mannwhitney_results.csv', index=False)
df_2class.to_csv('bioactivity_data_eda.csv')
