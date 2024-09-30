import pandas as pd 

def get_models_dfs():

    filename = "analyzing_models\RCMs_and_GCMs.xlsx"

    rcms = pd.read_excel(filename, sheet_name='RCMs')
    gcms = pd.read_excel(filename, sheet_name='GCMs')
    realizations = pd.read_excel(filename, sheet_name='89realizations')

    return gcms, rcms, realizations

def create_matrix(gcms, rcms, realizations):

    g = list(gcms['GCM'])
    r = list(rcms['RCM'])

    matrix = pd.DataFrame(0, index=g, columns=r)

    for _, row in realizations.iterrows():
        if row['GCM'] in g and row['RCM'] in r:
            matrix.loc[row['GCM'], row['RCM']] += 1

    return matrix

if __name__ == '__main__':

    gcms, rcms, realizations = get_models_dfs()

    matrix = create_matrix(gcms, rcms, realizations)
