class CFG:
    EXCEL_PATH = "./data/附件.xlsx"
    SHEET_NAME = "男胎检测数据"
    Q1PicPath = "./img/Q1/"
    Q2PicPath = "./img/Q2/"
    Q3PicPath = "./img/Q3/"

    COL_ID = "孕妇代码"
    COL_BMI = "孕妇BMI"
    
    USE_EMPIRICAL_BMI = True      # True: 使用 Excel 实际分布；False: 使用等距网格
    DEDUP_BY_MOTHER = True        # 按孕妇去重（用该孕妇 BMI 的中位数）
    MAX_BMI_POINTS = 250          # 若人数过多，压缩为至多这些代表点（等频分箱），并带权重

    # ---------- 阈值与不确定性 ----------
    THRESHOLD = 0.05
    CONF_LEVEL = 0.975
    SIGMA_M_LIST = [0.0, 0.1, 0.2]
    SIGMA_M = 0.10

    # ---------- 风险分段（题意） ----------
    RISK_EARLY, RISK_MID, RISK_LATE = 1.0, 2.0, 4.0

    # ---------- 复检与成本 ----------
    RETEST_COST = 1.5           # 复检单位成本 c_retest
    VISIT_INTERVAL = 1.0        # 复检间隔（周）
    FIRST_VISIT_COST = 0.0      # 首检固定成本（可留 0）
    MAX_RETESTS = 10            # 安全上限

    # ---------- 分段设置（固定段数） ----------
    N_GROUPS = 4
    MIN_SEG_SIZE = 5

    # ---------- 下限策略（二选一；默认软下限） ----------
    HARD_FLOOR = False          # True=硬下限：不在训练下限之前搜索
    SOFT_FLOOR = True           # True=软下限：允许更早，但人为增大 SE
    EXTRAP_SIGMA_PER_WEEK = 0.45
    SOFT_BAND = 1.5
    SOFT_BETA = 4.0
    SOFT_POWER = 1.5

    # ---------- 搜索范围与步长 ----------
    T_MIN_RAW, T_MAX, STEP = 10.0, 30.0, 0.1

    # ---------- 输出 ----------
    OUT_GROUP_SUMMARY = "group_summary.csv"
    OUT_TSTAR_PNG = "t_star_vs_bmi.png"
    OUT_GROUPS_PNG = "groups_on_curve.png"
    OUT_SENS_TG_PNG = "sensitivity_Tg_sigma.png"

    USE_WAIT_PENALTY = True
    WAIT_PENALTY_PER_WEEK = 1.0   # α，和 RETEST_COST 同量纲