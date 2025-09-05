import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

# 读取数据
data = pd.read_csv('./data/附件.csv')

# 将孕周转换为数值的函数
def convert_pregnancy_week(week_str):
    # 处理格式如 "13w+0" 或 "13w" 的字符串
    week_str = week_str.lower()
    if '+' in week_str:
        parts = week_str.split('w+')
        weeks = float(parts[0])
        days = float(parts[1]) if len(parts) > 1 else 0
    else:
        weeks = float(week_str.replace('w', ''))
        days = 0
    return weeks + days/7

# 数据预处理
def preprocess_data(data):
    # 转换孕周为数值
    data['孕周数值'] = data['检测孕周'].apply(convert_pregnancy_week)
    
    # 提取特征
    features = ['孕妇代码', '孕周数值', '孕妇BMI', '年龄']
    X = data[features]
    y = data['Y染色体浓度']
    
    return X, y

if __name__ == "__main__":
    X, y = preprocess_data(data)
    # 计算每个孕妇代码的样本数量
    sample_counts = X['孕妇代码'].value_counts()
    print(sample_counts)
    # 统计每个样本数量的孕妇代码数量
    count_distribution = sample_counts.value_counts().sort_index()
    print(count_distribution)
    # 绘制每个样本数量的孕妇代码数量分布直方图
    plt.figure(figsize=(10, 6))
    sns.barplot(x=count_distribution.index, y=count_distribution.values, palette="viridis")
    plt.xlabel('孕妇检验次数')
    plt.ylabel('数量')
    plt.title('孕妇检验次数分布')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig('孕妇检验次数分布.png')
    plt.show()
