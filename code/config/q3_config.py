"""
Configuration for Q3Model
"""

class CFG:
    # ---------- 数据源配置 ----------
    EXCEL_PATH = "./data/附件.xlsx"
    SHEET_NAME = "男胎检测数据"
    COL_ID = "孕妇代码"
    COL_BMI = "孕妇BMI"
    
    # ---------- 模型参数 ----------
    THRESHOLD = 0.04              # Y染色体浓度阈值
    CONF_LEVEL = 0.99            # 置信水平
    SIGMA_M = 0.10               # 测量误差
    SIGMA_M_LIST = [0.0, 0.1, 0.2]
    
    # ---------- BMI分布 ----------
    MAX_BMI_POINTS = 250         # 最大BMI代表点数
    
    # ---------- 风险分段 ----------
    RISK_EARLY = 1.0             # 早期风险
    RISK_MID = 2.0               # 中期风险
    RISK_LATE = 4.0              # 晚期风险
    
    # ---------- 复检相关 ----------
    RETEST_COST = 1.5            # 复检单位成本
    VISIT_INTERVAL = 1.0         # 复检间隔(周)
    MAX_RETESTS = 10             # 最大复检次数
    FIRST_VISIT_COST = 0.0      # 首检固定成本
    
    # ---------- 达标约束 ----------
    COVERAGE_TARGET = 0.85         # 目标达标率
    COVERAGE_PENALTY_WEIGHT = 100.0       # 约束惩罚权重
    
    # ---------- 分段设置 ----------
    N_GROUPS = 3                  # 分段数
    MIN_SEG_SIZE = 5             # 最小段长
    
    # ---------- 搜索范围 ----------
    T_MIN = 10.0                 # 最早检测周
    T_MAX = 30.0                 # 最晚检测周
    STEP = 0.05               # 时间步长
    
    # ---------- 文件输出 ----------
    OUT_DIR = "./output/Q3/"
