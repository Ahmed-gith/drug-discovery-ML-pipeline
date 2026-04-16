import pandas as pd
from chembl_webresource_client.new_client import new_client


target = new_client.target


target_query = target.search('aromatase')

targets = pd.DataFrame.from_dict(target_query)



selected_target = targets.target_chembl_id[0]



activity = new_client.activity

res = activity.filter(target_chembl_id=selected_target).filter(standard_type="IC50")

df = pd.DataFrame.from_dict(res)




df2 = df[df.standard_value.notna()]



mol_cid = []
canonical_smile = []
standard_value = []

for i in df2.molecule_chembl_id:
    mol_cid.append(i)

for i in df2.canonical_smiles:
    canonical_smile.append(i)

for i in df2.standard_value:
    standard_value.append(i)




data_truple = list(zip(mol_cid,standard_value,canonical_smile))


df3 = pd.DataFrame(data_truple, columns=['molecule_cheml_id', 'standard_value', 'canonical_smiles'])




df3.to_csv('bioactivity_data.csv', index=False)


df4 = pd.read_csv('bioactivity_data.csv')

bioactivity_class = []

for i in df4.standard_value:
    if float(i)  >= 10000:
        bioactivity_class.append("inactive")
    elif float(i) <= 1000:
        bioactivity_class.append("active")
    else:
        bioactivity_class.append("intermediate")

bioactivity_object = pd.Series(bioactivity_class, name='bioactivity_class')
df5 = pd.concat([df4, bioactivity_object], axis=1)

df5.to_csv('bioactivity_data_updates.csv')