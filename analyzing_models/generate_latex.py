

import pandas as pd

file_path = 'RCMs_and_GCMs.xlsx'  # Update this with the path to your Excel file
df = pd.read_excel(file_path, sheet_name = '89realizations_short')
# Function to generate LaTeX longtable
def generate_latex_longtable(df, quadro_caption, quadro_label):
    latex_code = r'''\begin{longtable}{|p{1in} | p{2in} | p{2in} |} 
    \caption{''' + quadro_caption + r'''} \label{''' + quadro_label + r'''} \\
    \hline
    \textbf{ID} & \textbf{GCM} & \textbf{RCM} \\ \hline
    \endfirsthead
    \hline
    \textbf{ID} & \textbf{GCM} & \textbf{RCM} \\ \hline
    \endhead
    \hline \multicolumn{3}{r}{Continua na próxima página} \\ \hline
    \endfoot
    \hline
    \endlastfoot
    '''

    # Loop through the DataFrame and add rows to LaTeX longtable
    for index, row in df.iterrows():
        latex_code += f"{row['id']} & {row['GCM']} & {row['RCM']} \\\\ \\hline\n"
    
    latex_code += r'\end{longtable}'

    return latex_code

# Set the caption and label for the LaTeX longtable
caption = 'Combinação de Modelos Climáticos Globais (GCM) e Regionais (RCM)'
label = 'qua:CombGCMRCM'

# Generate LaTeX longtable code
latex_table = generate_latex_longtable(df, caption, label)

# Print the generated LaTeX code
print(latex_table)

# Optionally, save the LaTeX code to a file
with open('latex_longtable_output.tex', 'w') as f:
    f.write(latex_table)


