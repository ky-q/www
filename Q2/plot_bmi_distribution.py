# -*- coding: utf-8 -*-
"""
可视化孕妇BMI中位数分布
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from planB_constrained import load_empirical_bmi  # 导入已有的函数

# 设置中文显示
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

def main():
    # 配置参数
    EXCEL_PATH = "./data/附件.xlsx"
    SHEET_NAME = "男胎检测数据"
    COL_ID = "孕妇代码"
    COL_BMI = "孕妇BMI"
    DEDUP_BY_MOTHER = True  # 按孕妇去重

    # 读取BMI数据
    bmi_values, weights = load_empirical_bmi(
        EXCEL_PATH, SHEET_NAME, COL_ID, COL_BMI, 
        dedup_by_mother=DEDUP_BY_MOTHER, max_points=1000  # 使用较大的max_points以获得更精细的分布
    )
    
    # 打印统计信息
    print(f"BMI数据统计 (按孕妇去重):")
    print(f"  样本数量: {len(bmi_values)}")
    print(f"  平均值: {np.average(bmi_values, weights=weights):.2f}")
    print(f"  中位数: {np.median(bmi_values):.2f}")
    print(f"  标准差: {np.sqrt(np.average((bmi_values - np.average(bmi_values, weights=weights))**2, weights=weights)):.2f}")
    print(f"  最小值: {np.min(bmi_values):.2f}")
    print(f"  最大值: {np.max(bmi_values):.2f}")
    
    # 计算分位数
    percentiles = [5, 25, 50, 75, 95]
    quantiles = np.percentile(bmi_values, percentiles)
    print("\n分位数:")
    for p, q in zip(percentiles, quantiles):
        print(f"  {p}%: {q:.2f}")

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. 直方图 + 密度曲线
    # 创建DataFrame以便正确处理权重
    df_bmi = pd.DataFrame({"BMI": bmi_values, "weight": weights})
    sns.histplot(data=df_bmi, x="BMI", weights="weight", kde=True, ax=ax1, bins=30)
    ax1.set_title("孕妇BMI分布 (按孕妇去重，取中位数)")
    ax1.set_xlabel("BMI")
    ax1.set_ylabel("频数")
    
    # 标记关键统计量
    median_bmi = np.median(bmi_values)
    mean_bmi = np.average(bmi_values, weights=weights)
    # ax1.axvline(median_bmi, color='red', linestyle='--', label=f'中位数: {median_bmi:.2f}')
    # ax1.axvline(mean_bmi, color='green', linestyle='-.', label=f'平均数: {mean_bmi:.2f}')
    
    # # 标记分位数
    # for p, q in zip(percentiles, quantiles):
    #     if p in [25, 75]:  # 只在图上标记25%和75%分位数，避免太杂乱
    #         ax1.axvline(q, color='purple', linestyle=':', alpha=0.7)
    #         ax1.text(q, ax1.get_ylim()[1]*0.9, f'{p}%', color='purple', rotation=90, alpha=0.7)
    
    ax1.legend()
    
    # 2. 箱型图
    # 使用DataFrame以便正确处理
    ax2.boxplot(bmi_values, vert=True)
    ax2.set_title("孕妇BMI箱型图")
    ax2.set_ylabel("BMI")
    
    # 添加抖动散点图以显示数据分布 (使用matplotlib而不是seaborn以避免权重问题)
    y_jittered = bmi_values + np.random.normal(0, 0.02, len(bmi_values))
    ax2.scatter(np.ones_like(bmi_values), y_jittered, color='red', alpha=0.3, s=20)
    
    # 标记异常值(如果有)
    Q1, Q3 = np.percentile(bmi_values, [25, 75])
    IQR = Q3 - Q1
    outliers = bmi_values[(bmi_values < Q1 - 1.5*IQR) | (bmi_values > Q3 + 1.5*IQR)]
    if len(outliers) > 0:
        print(f"\n异常值 ({len(outliers)}个):")
        print(f"  {outliers}")
    
    plt.tight_layout()
    plt.savefig("bmi_distribution.png", dpi=300)
    print("\n图表已保存为 'bmi_distribution.png'")
    
    # 额外创建BMI分组图
    plt.figure(figsize=(10, 6))
    
    # 定义BMI分类（WHO标准）
    bmi_categories = [
        (0, 18.5, "低体重"),
        (18.5, 24.9, "正常"),
        (25.0, 29.9, "超重"),
        (30.0, 34.9, "1度肥胖"),
        (35.0, 39.9, "2度肥胖"),
        (40.0, float('inf'), "3度肥胖")
    ]
    
    # 计算每个分类的数量
    category_counts = []
    category_labels = []
    
    for low, high, label in bmi_categories:
        mask = (bmi_values >= low) & (bmi_values < high)
        count = np.sum(weights[mask]) if mask.any() else 0
        if count > 0:  # 只显示有数据的分类
            category_counts.append(count)
            category_labels.append(f"{label}\n({low}-{high if high != float('inf') else '∞'})")
    
    # 绘制饼图
    plt.pie(category_counts, labels=category_labels, autopct='%1.1f%%', startangle=90)
    plt.title("孕妇BMI分类分布")
    plt.axis('equal')  # 确保饼图是圆形的
    
    plt.tight_layout()
    plt.savefig("bmi_categories.png", dpi=300)
    print("BMI分类图已保存为 'bmi_categories.png'")
    
    # 显示图表
    plt.show()
    
if __name__ == "__main__":
    main()
