from collections import defaultdict
import pandas as pd
from numpy.random import shuffle
from lightgbm import LGBMClassifier
import numpy as np
import joblib

# 理化性质数据
properties = {
    'A': [0.62, -0.5, 15.0, 2.35, 9.87, 6.11],
    'C': [0.29, -1.0, 47.0, 1.71, 10.78, 5.02],
    'D': [-0.90, 3.0, 59.0, 1.88, 9.60, 2.98],
    'E': [-0.74, 3.0, 73.0, 2.19, 9.67, 3.08],
    'F': [1.19, -2.5, 91.0, 2.58, 9.24, 5.91],
    'G': [0.48, 0.0, 1.0, 2.34, 9.60, 6.06],
    'H': [-0.40, -0.5, 82.0, 1.78, 8.97, 7.64],
    'I': [1.38, -1.8, 57.0, 2.32, 9.76, 6.04],
    'K': [-1.50, 3.0, 73.0, 2.20, 8.90, 9.47],
    'L': [1.06, -1.8, 57.0, 2.36, 9.60, 6.04],
    'M': [0.64, -1.3, 75.0, 2.28, 9.21, 5.74],
    'N': [-0.78, 0.2, 58.0, 2.18, 9.09, 10.76],
    'P': [0.12, 0.0, 42.0, 1.99, 10.60, 6.30],
    'Q': [-0.85, 0.2, 72.0, 2.17, 9.13, 5.65],
    'R': [-2.53, 3.0, 101.0, 2.18, 9.09, 10.76],
    'S': [-0.18, 0.3, 31.0, 2.21, 9.15, 5.68],
    'T': [-0.05, -0.4, 45.0, 2.15, 9.12, 5.60],
    'V': [1.08, -1.5, 43.0, 2.29, 9.74, 6.02],
    'W': [0.81, -3.4, 130.0, 2.38, 9.39, 5.88],
    'Y': [0.26, -2.3, 107.0, 2.20, 9.11, 5.63],
}


# 自定义PseAAC特征计算函数
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

# 批量提取蛋白质PseAAC特征并将结果写入输出文件
def extract_pseaac_features(input_file, output_file):
    proteins = pd.read_excel(input_file)
    data = pd.DataFrame()
    for index, row in proteins.iterrows():
        sequence = row['Sequence']
        pseaac_features = calculate_pseaac(sequence)
        # 跳过空的特征数组
        if pseaac_features.size == 0:
            continue
        for i, feature in enumerate(pseaac_features):
            data.loc[index, f'Feature_{i+1}'] = feature
        data.loc[index, 'label'] = row['label']
    # ...


    # 导出到excel文件
    data.to_excel(output_file, index=False)


# 执行批量提取蛋白质PseAAC特征的函数
input_file = "D:/paper-peptide/multi/dataset4.xlsx"
output_file = "D:/paper-peptide/multi/pseaac4.xlsx"
extract_pseaac_features(input_file, output_file)


inputfile =  output_file


def readData():

    data = pd.read_excel(inputfile)
    data = data.values.astype(np.float32)
    shuffle(data)  # 随机打乱
    data_train = data[:int(1 * len(data)), :]  # 数据集矩阵
    return data_train


def train(data_train):

    x = data_train[:, 0:27] * 30  # 放大特征，矩阵2维
    y = data_train[:, 27].astype(int)  # 矩阵1维
    return x, y


if __name__ == '__main__':
    data_train = readData()
    x, y = train(data_train)



# 画平均ROC曲线的两个参数
model = LGBMClassifier(
        n_jobs=-1,
        device_type='cpu',
        n_estimators=400,
        learning_rate=0.1,
        max_depth=5,
        num_leaves=32,
        colsample_bytree=0.51,
        subsample=0.6,
        # max_bins=127,
    )


probas_ = model.fit(x, y)

# save model to file 模型保存
joblib.dump(model, open("D:/paper-peptide/multi/model4.pkl", "wb"))

