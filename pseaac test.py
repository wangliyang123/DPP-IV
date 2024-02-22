import pandas as pd
import joblib
import numpy as np
from collections import defaultdict

# 加载保存的模型
model = joblib.load("D:/paper-peptide/multi/DPP IV/model4.pkl")

# 加载测试集数据
test_file = "D:/paper-peptide/multi/test.xlsx"
test_data = pd.read_excel(test_file)

# 定义函数用于计算PseAAC特征
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

# 遍历测试集数据进行预测并添加到DataFrame中
for index, row in test_data.iterrows():
    sequence = row['Sequence']
    pseaac_features = calculate_pseaac(sequence)
    # 跳过空的特征数组
    if pseaac_features.size == 0:
        continue
    # 将特征转换为DataFrame格式，并添加到测试集数据中
    for i, value in enumerate(pseaac_features):
        test_data.loc[index, f"Feature_{i + 1}"] = value

    # 进行预测并添加预测结果到DataFrame中
    features_array = test_data.loc[index, [f"Feature_{i + 1}" for i in range(len(pseaac_features))]].values.reshape(1,
                                                                                                                    -1)
    prediction = model.predict_proba(features_array)

    test_data.loc[index, 'Prediction'] = prediction[:, 1] * 100  # 以分数形式表示概率

# 输出结果到Excel文件
output_file = "D:/paper-peptide/multi/predict4.xlsx"
test_data.to_excel(output_file, index=False)
