import pandas as pd
import joblib
import numpy as np
from collections import defaultdict

# 加载保存的模型
model = joblib.load("D:/paper-peptide/multi/model1.pkl")

# 加载测试集数据
test_file = "D:/paper-peptide/multi/test.xlsx"
test_data = pd.read_excel(test_file)

# 用于存储特征和预测结果的DataFrame
results_data = pd.DataFrame()


def calculate_pseaac(sequence, lambda_value=1, w=0.05):
    # 添加类型检查和处理
    if not isinstance(sequence, str):
        print(f"Invalid sequence: {sequence}. Must be a string.")
        return np.array([])  # 返回一个空的numpy数组
    protein = sequence.upper()
    aa_list = 'ACDEFGHIKLMNPQRSTVWY'
    aa_dict = defaultdict(int)

    for aa in protein:
        if aa in aa_list:
            aa_dict[aa] += 1

    aa_freq = np.array([float(aa_dict[aa]) / len(protein) for aa in aa_list])

    # 计算1级相关因子，这里我们假设物化性质为0
    correlation_factor = np.zeros(1)

    # 按照您的描述，我们还需要考虑6个理化性质
    # 这里，我将它们初始化为0，您可以根据需要修改
    physicochemical_properties = np.zeros(6)

    pseaac = np.concatenate([aa_freq, correlation_factor, physicochemical_properties])

    return pseaac
# 这个变量需要根据你实际的特征数量进行修改
expected_features_length = 27  # 举例，你需要根据实际情况调整这个数字

sequences = []
features_list = []
predictions = []
for index, row in test_data.iterrows():
    sequence = row['Sequence']

    if pd.isna(sequence):  # 检查序列是否为 nan
        print("Encountered a NaN value in sequence. Skipping.")
        continue

    pseaac_features = calculate_pseaac(sequence)

    sequences.append(sequence)

    if pseaac_features.size == 0:
        features_list.append([np.nan] * expected_features_length)
        predictions.append(np.nan)
    else:
        features_list.append(pseaac_features)
        features_array = pseaac_features.reshape(1, -1)
        prediction = model.predict_proba(features_array)
        predictions.append(prediction[:, 1] * 100)

results_data['Sequence'] = sequences
features_data = pd.DataFrame(features_list, columns=[f"Feature_{i + 1}" for i in range(expected_features_length)])
results_data = pd.concat([results_data, features_data], axis=1)
results_data['Prediction'] = predictions

results_data.to_excel("D:/paper-peptide/multi/PseAAC_features test.xlsx", index=False)



# 接下来进行预测
predictions = []
for index, row in features_data.iterrows():
    features_array = row.values.reshape(1, -1)
    prediction = model.predict_proba(features_array)
    predictions.append(prediction[:, 1] * 100)

results_data['Prediction'] = predictions

# 输出预测结果
#results_data.to_excel("D:/paper-peptide/DPP IV/test set/Trypsin_prediction.xlsx", index=False)
