import pandas as pd

def get_models_dfs():
    filename = "analyzing_models/RCMs_and_GCMs.xlsx" 

    rcms = pd.read_excel(filename, sheet_name='RCMs')
    gcms = pd.read_excel(filename, sheet_name='GCMs')
    realizations = pd.read_excel(filename, sheet_name='89realizations')

    return gcms, rcms, realizations

def create_matrix(gcms, rcms, realizations):
    # Use ids instead of names
    gcm_ids = list(gcms['id'])
    rcm_ids = list(rcms['id'])

    # Create a matrix with RCM_ids as rows and GCM_ids as columns
    matrix = pd.DataFrame(0, index=rcm_ids, columns=gcm_ids)

    # Mapping from names to ids
    gcm_name_to_id = dict(zip(gcms['GCM'], gcms['id']))
    rcm_name_to_id = dict(zip(rcms['RCM'], rcms['id']))

    # Fill in the matrix using the ids
    for _, row in realizations.iterrows():
        gcm_id = gcm_name_to_id.get(row['GCM'])
        rcm_id = rcm_name_to_id.get(row['RCM'])
        
        if gcm_id in gcm_ids and rcm_id in rcm_ids:
            matrix.loc[rcm_id, gcm_id] += 1

    return matrix

if __name__ == '__main__':
    gcms, rcms, realizations = get_models_dfs()
    matrix = create_matrix(gcms, rcms, realizations)

    # Display the resulting matrix
    matrix.to_excel("matrix_RCM_GCM.xlsx")
