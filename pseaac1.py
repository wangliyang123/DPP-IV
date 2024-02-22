#from Bio import SeqIO
import numpy as np
from collections import defaultdict
import time
import pandas as pd

def calculate_pseaac(sequence, lambda_value=1, w=0.05):
    protein = sequence.upper()
    aa_list = 'ACDEFGHIKLMNPQRSTVWY*X'
    aa_dict = defaultdict(int)
    for aa in protein:
        if aa in aa_list:
            aa_dict[aa] += 1
    aa_freq = np.array([float(aa_dict[aa])/len(protein) for aa in aa_list])
    lamada = lambda_value
    pseaac = []
    for n in range(1, lamada+1):
        temp = []
        for i in range(len(protein)-n):
            temp.append(protein[i:i+n+1])
        dipeptide_dict = defaultdict(int)
        for dipeptide in temp:
            if dipeptide in dipeptide_dict:
                dipeptide_dict[dipeptide] += 1
        for aa1 in aa_list:
            for aa2 in aa_list:
                dipeptide = aa1+aa2
                pseaac.append(float(dipeptide_dict[dipeptide] + w)/(aa_dict[aa1]*aa_dict[aa2] + w**2))
    pseaac = np.array(pseaac)
    return pseaac

def extract_pseaac_features(input_file, output_file):
    start_time = time.time()
    data = pd.read_excel(input_file)
    result = []
    for sequence in data['Sequence']:
        if not isinstance(sequence, str):
            print(f"Warning: non-string sequence encountered: {sequence}")
            continue
        pseaac_features = calculate_pseaac(sequence)
        pseaac_features = pseaac_features[:20]  # 取前20维特征
        result.append(pseaac_features)
    result = pd.DataFrame(result)
    result.to_excel(output_file)
    end_time = time.time()
    print(f"Batch extraction completed in {end_time-start_time:.2f} seconds.")


input_file = "D:/paper-peptide/DPP IV/training set/train.xlsx"
output_file = "D:/paper-peptide/DPP IV/training set/pseaac.xlsx"
extract_pseaac_features(input_file, output_file)
