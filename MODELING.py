
"""
数据加载模块
负责从Excel文件读取所有附件数据
"""

import pandas as pd
import numpy as np
from pathlib import Path

class DataLoader:
    """数据加载器"""

    def __init__(self, data_dir):
        """
        初始化数据加载器

        Args:
            data_dir: 附件数据所在目录
        """
        self.data_dir = Path(data_dir)

    def load_attachment1(self):
        """
        加载附件1：典型日数据

        Returns:
            dict: {
                'price': 电价数组 (144,)
                'load': 负载数组 (144,)
                'pv_forecast': 光伏预测数组 (144,)
            }
        """
        file_path = self.data_dir / '附件1.xlsx'
        df = pd.read_excel(file_path)

        # 提取数据（根据实际列名）
        data = {
            'price': df['电价'].values,
            'load': df['小区负载'].values,
            'pv_forecast': df['光伏发电预测功率'].values
        }

        return data

    def load_attachment2(self):
        """
        加载附件2：2025年全年实际数据

        Returns:
            dict: {
                'load': DataFrame (365天 × 144时段)
                'pv_actual': DataFrame (365天 × 144时段)
            }
        """
        file_path = self.data_dir / '附件2.xlsx'

        # 读取负载数据
        load_df = pd.read_excel(file_path, sheet_name='小区负载')
        # 第一列是日期，后面144列是数据
        load_data = load_df.iloc[:, 1:145].values  # (365, 144)

        # 读取光伏实际功率数据
        pv_df = pd.read_excel(file_path, sheet_name='光伏发电实际功率')
        pv_data = pv_df.iloc[:, 1:145].values  # (365, 144)

        # 提取日期
        dates = pd.to_datetime(load_df.iloc[:, 0])

        data = {
            'load': load_data,
            'pv_actual': pv_data,
            'dates': dates
        }

        return data

    def load_attachment3(self):
        """
        加载附件3：光伏发电预报数据

        Returns:
            dict: {
                'forecasts': DataFrame with columns [date, forecast_time, hour_1, ..., hour_24]
                或组织成嵌套字典 {date: {time: array}}
            }
        """
        file_path = self.data_dir / '附件3.xlsx'
        df = pd.read_excel(file_path)

        # 组织预报数据为嵌套字典: {date: {forecast_time: hourly_forecast_array}}
        forecasts = {}
        current_date = None

        for idx, row in df.iterrows():
            # 如果日期列不为空，更新当前日期
            if pd.notna(row.iloc[0]):  # 第一列是日期
                date_val = row.iloc[0]
                if isinstance(date_val, pd.Timestamp):
                    current_date = date_val
                else:
                    # 尝试解析字符串日期
                    current_date = pd.to_datetime(str(date_val))

                forecasts[current_date] = {}

            # 预报时刻（第二列）
            forecast_time = str(row.iloc[1])  # '0:00', '6:00', '12:00', '18:00'

            # 提取24小时预报数据（从第3列开始）
            forecast_24h = row.iloc[2:26].values.astype(float)

            if current_date is not None:
                forecasts[current_date][forecast_time] = forecast_24h

        return {'forecasts': forecasts}

    def load_attachment4(self):
        """
        加载附件4：波动电价数据

        Returns:
            dict: {
                'prices': numpy array shape (365, 144), 每行是一天144个时段的电价
                'dates': pandas Series of Timestamps
            }
        """
        file_path = self.data_dir / '附件4.xlsx'
        df = pd.read_excel(file_path)

        # 第一列是日期，后面144列是电价数据
        dates = pd.to_datetime(df.iloc[:, 0])
        prices = df.iloc[:, 1:145].values.astype(float)  # (365, 144)

        return {
            'prices': prices,
            'dates': dates
        }

def interpolate_hourly_to_10min(hourly_data):
    """
    将整点预报数据插值到10分钟间隔

    Args:
        hourly_data: 整点数据 (25个点: 0:00到24:00)

    Returns:
        np.array: 10分钟数据 (144个点)
    """
    if len(hourly_data) == 25:
        # 线性插值
        from scipy.interpolate import interp1d
        hours = np.arange(25) / 24 * 144  # 映射到144个时间段
        time_10min = np.arange(144)

        f = interp1d(hours, hourly_data, kind='linear')
        data_10min = f(time_10min)

        return data_10min
    elif len(hourly_data) == 144:
        # 已经是10分钟数据
        return np.array(hourly_data)
    else:
        raise ValueError(f"Unexpected data length: {len(hourly_data)}")

def parse_time_index(time_str):
    """
    解析时间字符串到时间段索引

    Args:
        time_str: 如 "10:00" 或 "10:10"

    Returns:
        int: 时间段索引 (0-143)
    """
    parts = time_str.split(':')
    hour = int(parts[0])
    minute = int(parts[1])

    # 计算时间段索引
    index = hour * 6 + minute // 10

    return index

def create_time_array():
    """
    创建时间标签数组

    Returns:
        list: 144个时间标签 ["0:00", "0:10", ..., "23:50"]
    """
    time_labels = []
    for h in range(24):
        for m in range(0, 60, 10):
            time_labels.append(f"{h:02d}:{m:02d}")

    return time_labels

if __name__ == "__main__":
    # 测试数据加载
    loader = DataLoader("../附件")

    print("测试加载附件1...")
    data1 = loader.load_attachment1()
    print(f"电价shape: {data1['price'].shape}")
    print(f"负载shape: {data1['load'].shape}")
    print(f"光伏预测shape: {data1['pv_forecast'].shape}")

    print("\n时间标签示例:")
    times = create_time_array()
    print(times[:10])
    print(f"总共{len(times)}个时间段")
"""
改进的问题2优化器
支持紧急购电和光伏直充储能
"""

import pulp
import numpy as np

class ImprovedProblem2Optimizer:
    """改进的问题2优化器"""

    def __init__(self):
        """初始化优化器参数"""
        # 储能设备参数
        self.E_MAX_CAP = 12000  # 最大容量 (kWh)
        self.E_MIN = 1200       # 运行下限 (kWh)
        self.E_MAX = 10800      # 运行上限 (kWh)
        self.P_MAX = 5000       # 最大充放电功率 (kW)
        self.ETA_CH = 0.9       # 充电效率
        self.ETA_DIS = 0.9      # 放电效率

        # 时间参数
        self.DELTA_T = 1/6      # 时间间隔 (小时)
        self.T_PERIODS = 144    # 时间段数

        # 紧急购电价格倍数
        self.EMERGENCY_MULTIPLIER = 5

    def solve(self, price, load, pv_forecast, E_init, solver_name='PULP_CBC_CMD',
              terminal_soc_weight=None, E_terminal=None):
        """
        计划阶段（0:00决策）：基于预测光伏/负载制定当天计划购电量 P_buy。

        0:00时看不到当天真实数据，故目标函数中不含紧急购电惩罚——
        优化器只对预测场景最小化购电成本，紧急购电的实际发生由
        solve_execution() 在执行阶段用真实数据计算。

        Args:
            price:       电价数组 (144,)
            load:        预测负载数组 (144,)
            pv_forecast: 预测光伏数组 (144,)
            E_init:      初始储能 (kWh)
            solver_name: 求解器名称
            terminal_soc_weight: 终态储能价值权重（元/kWh），None 则自动估算
            E_terminal:  若非 None，强制终态储能等于该值

        Returns:
            dict: 含 P_buy (144,) 等计划结果
        """
        model = pulp.LpProblem("Planning", pulp.LpMinimize)

        # 决策变量
        P_buy       = [pulp.LpVariable(f"P_buy_{t}",  lowBound=0) for t in range(self.T_PERIODS)]
        P_charge    = [pulp.LpVariable(f"P_ch_{t}",   lowBound=0, upBound=self.P_MAX) for t in range(self.T_PERIODS)]
        P_discharge = [pulp.LpVariable(f"P_dis_{t}",  lowBound=0, upBound=self.P_MAX) for t in range(self.T_PERIODS)]
        E_storage   = [pulp.LpVariable(f"E_st_{t}",   lowBound=self.E_MIN, upBound=self.E_MAX) for t in range(self.T_PERIODS + 1)]
        delta_ch    = [pulp.LpVariable(f"dch_{t}",    cat='Binary') for t in range(self.T_PERIODS)]
        delta_dis   = [pulp.LpVariable(f"ddis_{t}",   cat='Binary') for t in range(self.T_PERIODS)]
        P_pv_to_storage = [pulp.LpVariable(f"pvs_{t}", lowBound=0) for t in range(self.T_PERIODS)]
        P_pv_to_load    = [pulp.LpVariable(f"pvl_{t}", lowBound=0) for t in range(self.T_PERIODS)]

        # 终态储能价值权重：存1kWh电的价值 ≈ 明天凌晨充回所需成本
        if terminal_soc_weight is None:
            terminal_soc_weight = float(np.mean(price[:36]) / self.ETA_CH)

        # 目标函数：最小化计划购电费用 - 终态储能价值
        # 注意：此处不含紧急购电，因为0:00看不到当天真实数据
        model += (
            pulp.lpSum(price[t] * P_buy[t] * self.DELTA_T for t in range(self.T_PERIODS))
            - terminal_soc_weight * E_storage[self.T_PERIODS]
        )

        for t in range(self.T_PERIODS):
            # 光伏分配约束（使用预测光伏）
            model += (P_pv_to_load[t] + P_pv_to_storage[t] <= pv_forecast[t], f"pv_split_{t}")

            # 功率平衡约束（无紧急购电：预测场景下计划购电需覆盖所有需求）
            # 购电 + 光伏供负载 + 放电 >= 负载 + 充电 - 光伏直充
            model += (
                P_buy[t] + P_pv_to_load[t] + P_discharge[t] >=
                load[t] + P_charge[t] - P_pv_to_storage[t],
                f"power_balance_{t}"
            )

            # 储能动态约束（P_charge 为充电侧总功率，含光伏直充）
            model += (
                E_storage[t + 1] == E_storage[t]
                + P_charge[t] * self.ETA_CH * self.DELTA_T
                - P_discharge[t] / self.ETA_DIS * self.DELTA_T,
                f"storage_dynamics_{t}"
            )

            # 充放电互斥约束
            model += (P_charge[t]    <= self.P_MAX * delta_ch[t],  f"charge_limit_{t}")
            model += (P_discharge[t] <= self.P_MAX * delta_dis[t], f"discharge_limit_{t}")
            model += (delta_ch[t] + delta_dis[t] <= 1,              f"mutex_{t}")

            # 光伏直充只能在充电模式下
            model += (P_pv_to_storage[t] <= self.P_MAX * delta_ch[t], f"pv_charge_limit_{t}")

        # 边界条件
        model += (E_storage[0] == E_init, "initial_storage")
        if E_terminal is not None:
            model += (E_storage[self.T_PERIODS] == E_terminal, "terminal_storage")

        # 求解
        solver = pulp.getSolver(solver_name, msg=0, timeLimit=300)
        model.solve(solver)

        if model.status != pulp.LpStatusOptimal:
            return None

        P_buy_arr = np.array([pulp.value(P_buy[t]) or 0.0 for t in range(self.T_PERIODS)])

        return {
            'P_buy':           P_buy_arr,
            'P_charge':        np.array([pulp.value(P_charge[t])    or 0.0 for t in range(self.T_PERIODS)]),
            'P_discharge':     np.array([pulp.value(P_discharge[t]) or 0.0 for t in range(self.T_PERIODS)]),
            'E_storage':       np.array([pulp.value(E_storage[t])   or 0.0 for t in range(self.T_PERIODS + 1)]),
            'P_pv_to_storage': np.array([pulp.value(P_pv_to_storage[t]) or 0.0 for t in range(self.T_PERIODS)]),
            'P_pv_to_load':    np.array([pulp.value(P_pv_to_load[t])    or 0.0 for t in range(self.T_PERIODS)]),
            # 计划费用（不含紧急购电，仅供参考）
            'planned_cost':    float(np.sum(price * P_buy_arr * self.DELTA_T)),
            'E_buy':           P_buy_arr * self.DELTA_T,
            'total_purchase':  float(np.sum(P_buy_arr * self.DELTA_T)),
            'status':          model.status,
        }

    def simulate_execution(self, price, load, pv_actual, E_init, P_buy_plan):
        """
        执行阶段（规则仿真，无优化）：
        P_buy 已在0:00锁定，逐时段按确定性规则模拟实际调度：
          1. 计算本时段可用电力 = P_buy_plan[t] + pv_actual[t]
          2. 优先满足负载；多余电力尽量充入储能（不超过 P_MAX 和 E_MAX）
          3. 若可用 + 储能放电仍不足负载，触发紧急购电补足

        没有优化求解器，只有逐时段顺序计算。

        Args:
            price:       电价数组 (144,)
            load:        真实负载数组 (144,)
            pv_actual:   真实光伏数组 (144,)
            E_init:      初始储能 (kWh)
            P_buy_plan:  计划购电功率 (144,)，单位 kW

        Returns:
            dict: 执行结果
        """
        T = self.T_PERIODS
        P_buy_arr = np.array(P_buy_plan, dtype=float)

        P_ch_arr  = np.zeros(T)
        P_dis_arr = np.zeros(T)
        P_emg_arr = np.zeros(T)
        pvs_arr   = np.zeros(T)   # 光伏→储能
        pvl_arr   = np.zeros(T)   # 光伏→负载
        E_arr     = np.zeros(T + 1)
        E_arr[0]  = E_init

        for t in range(T):
            E_cur   = E_arr[t]
            pb      = P_buy_arr[t]
            pv      = float(pv_actual[t])
            ld      = float(load[t])

            # ── 步骤1：总供给（购电 + 光伏）vs 负载 ────────────────
            supply  = pb + pv
            surplus = supply - ld   # >0 富余，<0 缺口

            if surplus >= 0:
                # ── 富余：先用光伏和购电满足负载，剩余尽量充储能 ──
                # 光伏尽量直接供负载
                pv_to_load   = min(pv, ld)
                pv_remaining = pv - pv_to_load   # 负载满足后的剩余光伏

                # 可充入储能的总富余功率
                charge_avail = surplus            # = pb + pv - ld
                charge_cap   = min(self.P_MAX,
                                   (self.E_MAX - E_cur) / (self.ETA_CH * self.DELTA_T))
                charge_cap   = max(0.0, charge_cap)
                p_charge     = min(charge_avail, charge_cap)

                # 光伏直充部分（不超过 pv_remaining 和 p_charge）
                pv_to_storage = min(pv_remaining, p_charge)
                # 其余充电来自购电（隐含）

                pvl_arr[t]  = pv_to_load
                pvs_arr[t]  = pv_to_storage
                P_ch_arr[t] = p_charge
                P_dis_arr[t] = 0.0
                P_emg_arr[t] = 0.0

            else:
                # ── 缺口：先从储能放电补足 ──────────────────────────
                deficit = -surplus   # 需要额外补充的功率

                # 光伏全部供负载（pv < ld）
                pvl_arr[t]    = pv
                pvs_arr[t]    = 0.0
                P_ch_arr[t]   = 0.0

                discharge_cap = min(self.P_MAX,
                                    (E_cur - self.E_MIN) * self.ETA_DIS / self.DELTA_T)
                discharge_cap = max(0.0, discharge_cap)
                p_discharge   = min(deficit, discharge_cap)
                P_dis_arr[t]  = p_discharge

                remaining_deficit = deficit - p_discharge
                P_emg_arr[t]  = max(0.0, remaining_deficit)

            # ── 储能状态更新 ─────────────────────────────────────────
            E_arr[t + 1] = (E_cur
                            + P_ch_arr[t]  * self.ETA_CH  * self.DELTA_T
                            - P_dis_arr[t] / self.ETA_DIS * self.DELTA_T)
            # 截断到合法范围（浮点误差保护）
            E_arr[t + 1] = float(np.clip(E_arr[t + 1], self.E_MIN, self.E_MAX))

        planned_cost   = float(np.sum(price * P_buy_arr * self.DELTA_T))
        emergency_cost = float(np.sum(
            self.EMERGENCY_MULTIPLIER * price * P_emg_arr * self.DELTA_T))

        return {
            'P_buy':           P_buy_arr,
            'P_emergency':     P_emg_arr,
            'P_charge':        P_ch_arr,
            'P_discharge':     P_dis_arr,
            'E_storage':       E_arr,
            'P_pv_to_storage': pvs_arr,
            'P_pv_to_load':    pvl_arr,
            'E_buy':           P_buy_arr * self.DELTA_T,
            'E_emergency':     P_emg_arr * self.DELTA_T,
            'total_purchase':  float(np.sum(P_buy_arr * self.DELTA_T)
                                     + np.sum(P_emg_arr * self.DELTA_T)),
            'planned_cost':    planned_cost,
            'emergency_cost':  emergency_cost,
            'total_cost':      planned_cost + emergency_cost,
        }

if __name__ == "__main__":
    print("改进的问题2优化器已创建")

"""
问题3优化器：带预报更新的购电策略优化
支持0:00计划购电 + 6:00/12:00/18:00调整购电
"""

import pulp
import numpy as np

class Problem3Optimizer:
    """
    问题3优化器：多次预报滚动优化

    特点：
    1. 0:00根据预报制定计划购电策略
    2. 6:00/12:00/18:00根据新预报调整购电策略
    3. 违约费用：计划 > 实际，差额按50%电价
    4. 超出费用：实际 > 计划，差额按150%电价
    5. 紧急购电：供电不足时按5倍电价
    """

    def __init__(self):
        # 储能参数
        self.E_MAX = 12000  # 最大容量 kWh
        self.E_MIN = 1200   # 最小容量 kWh
        self.E_UPPER = 10800  # 运行上限 kWh
        self.P_MAX = 5000   # 最大充放电功率 kW
        self.ETA_CH = 0.9   # 充电效率
        self.ETA_DIS = 0.9  # 放电效率

        # 时间参数
        self.T_PERIODS = 144  # 每天144个10分钟时段
        self.DELTA_T = 1/6    # 10分钟 = 1/6小时

        # 费用参数
        self.BREACH_RATIO = 0.5    # 违约比例：50%
        self.EXCESS_RATIO = 1.5    # 超出比例：150%
        self.EMERGENCY_RATIO = 5.0  # 紧急购电：5倍

    def hourly_to_10min(self, hourly_data):
        """
        将整点数据插值到10分钟间隔

        Args:
            hourly_data: 24小时整点数据

        Returns:
            144个10分钟数据
        """
        # 创建时间点
        hours = np.arange(25)  # 0-24小时
        minutes_10 = np.arange(0, 1440, 10) / 60  # 0-1440分钟转换为小时

        # 扩展hourly_data到25个点（加上24:00 = 0:00+1天）
        hourly_extended = np.append(hourly_data, hourly_data[-1])

        # 线性插值
        data_10min = np.interp(minutes_10, hours, hourly_extended)

        return data_10min[:144]  # 只取前144个（0:00-23:50）

    def solve_planned(self, price, load, pv_forecast_0h, E_init, solver_name='PULP_CBC_CMD',
                      terminal_soc_weight=None):
        """
        0:00时刻：根据预报制定全天计划购电策略

        Args:
            price: 电价数组 (144,)
            load: 负载数组 (144,)
            pv_forecast_0h: 0:00时刻的24小时光伏预报（整点）
            E_init: 初始储能 kWh
            solver_name: 求解器名称

        Returns:
            dict: {
                'P_plan': 计划购电功率 (144,)
                'P_charge': 充电功率 (144,)
                'P_discharge': 放电功率 (144,)
                'E_storage': 储能电量 (145,) - 包含初始和每个时段结束后
                'total_cost': 总费用（计划阶段）
                'status': 求解状态
            }
        """
        # 将整点预报插值到10分钟
        pv_forecast = self.hourly_to_10min(pv_forecast_0h)

        # 创建优化模型
        model = pulp.LpProblem("Problem3_Planned", pulp.LpMinimize)

        # 决策变量
        P_plan = [pulp.LpVariable(f"P_plan_{t}", lowBound=0) for t in range(self.T_PERIODS)]
        P_charge = [pulp.LpVariable(f"P_charge_{t}", lowBound=0, upBound=self.P_MAX)
                    for t in range(self.T_PERIODS)]
        P_discharge = [pulp.LpVariable(f"P_discharge_{t}", lowBound=0, upBound=self.P_MAX)
                       for t in range(self.T_PERIODS)]

        # 光伏分配
        P_pv_to_storage = [pulp.LpVariable(f"P_pv_storage_{t}", lowBound=0)
                          for t in range(self.T_PERIODS)]
        P_pv_to_load = [pulp.LpVariable(f"P_pv_load_{t}", lowBound=0)
                       for t in range(self.T_PERIODS)]

        # 储能状态
        E_storage = [pulp.LpVariable(f"E_storage_{t}", lowBound=self.E_MIN, upBound=self.E_UPPER)
                     for t in range(self.T_PERIODS + 1)]

        # 充放电互斥的二进制变量
        is_charging = [pulp.LpVariable(f"is_charging_{t}", cat='Binary')
                      for t in range(self.T_PERIODS)]

        # 终态储能价值权重：激励优化器为后一天留电
        # 含义：存1kWh电的价值 ≈ 明天凌晨重新充回所需成本
        # = 凌晨电价(0:00-6:00前36个时段均值) / 充电效率
        if terminal_soc_weight is None:
            terminal_soc_weight = float(np.mean(price[:36]) / self.ETA_CH)

        # 目标函数：最小化计划购电费用 - 终态储能价值
        total_cost = pulp.lpSum([P_plan[t] * self.DELTA_T * price[t] for t in range(self.T_PERIODS)]) \
                     - terminal_soc_weight * E_storage[self.T_PERIODS]
        model += total_cost

        # 约束条件
        # 1. 初始储能
        model += E_storage[0] == E_init

        for t in range(self.T_PERIODS):
            # 2. 光伏分配约束
            model += P_pv_to_load[t] + P_pv_to_storage[t] <= pv_forecast[t]

            # 3. 功率平衡约束（改进版：光伏直充储能）
            # P_discharge 为负载侧功率，P_charge 为 AC 侧充电功率，效率已在状态方程中体现，此处不重复
            model += (
                P_plan[t] + P_pv_to_load[t] + P_discharge[t] >=
                load[t] + P_charge[t] - P_pv_to_storage[t]
            )

            # 4. 储能动态约束
            model += (
                E_storage[t + 1] == E_storage[t] +
                (P_charge[t] * self.ETA_CH - P_discharge[t] / self.ETA_DIS) * self.DELTA_T
            )

            # 5. 充放电互斥约束
            model += P_charge[t] <= self.P_MAX * is_charging[t]
            model += P_discharge[t] <= self.P_MAX * (1 - is_charging[t])

        # 6. 终态自由：仅保证储能不低于下限
        model += E_storage[self.T_PERIODS] >= self.E_MIN

        # 求解
        solver = pulp.getSolver(solver_name, msg=False)
        model.solve(solver)

        if model.status != pulp.LpStatusOptimal:
            return None

        # 提取结果
        result = {
            'P_plan': np.array([P_plan[t].varValue for t in range(self.T_PERIODS)]),
            'P_charge': np.array([P_charge[t].varValue for t in range(self.T_PERIODS)]),
            'P_discharge': np.array([P_discharge[t].varValue for t in range(self.T_PERIODS)]),
            'P_pv_to_storage': np.array([P_pv_to_storage[t].varValue for t in range(self.T_PERIODS)]),
            'P_pv_to_load': np.array([P_pv_to_load[t].varValue for t in range(self.T_PERIODS)]),
            'E_storage': np.array([E_storage[t].varValue for t in range(self.T_PERIODS + 1)]),
            'total_cost': pulp.value(model.objective),
            'status': 'Optimal'
        }

        return result

    def solve_adjusted(self, price, load_forecast, pv_forecast, P_plan, E_current,
                      start_period, solver_name='PULP_CBC_CMD',
                      terminal_soc_weight=None):
        """
        6:00/12:00/18:00时刻：根据新预报调整剩余时段购电策略

        Args:
            price: 全天电价数组 (144,)
            load_forecast: 剩余时段负载预报 (remaining_periods,) — ARIMA预测
            pv_forecast: 剩余时段光伏预报 (remaining_periods,) — 附件3新预报，已插值到10分钟
            P_plan: 上一决策步骤的购电计划 (144,)，用于计算违约/超出
            E_current: 当前真实储能 kWh（来自greedy仿真）
            start_period: 调整开始时段 (36/72/108 对应 6:00/12:00/18:00)

        Returns:
            dict: P_adjusted, P_charge, P_discharge, E_storage, P_breach, P_excess,
                  breach_cost, excess_cost, total_cost, status
        """
        remaining_periods = self.T_PERIODS - start_period

        model = pulp.LpProblem(f"Problem3_Adjusted_{start_period}", pulp.LpMinimize)

        P_adjusted = [pulp.LpVariable(f"P_adj_{t}", lowBound=0)
                     for t in range(remaining_periods)]
        P_charge = [pulp.LpVariable(f"P_charge_{t}", lowBound=0, upBound=self.P_MAX)
                   for t in range(remaining_periods)]
        P_discharge = [pulp.LpVariable(f"P_discharge_{t}", lowBound=0, upBound=self.P_MAX)
                      for t in range(remaining_periods)]

        P_pv_to_storage = [pulp.LpVariable(f"P_pv_storage_{t}", lowBound=0)
                          for t in range(remaining_periods)]
        P_pv_to_load = [pulp.LpVariable(f"P_pv_load_{t}", lowBound=0)
                       for t in range(remaining_periods)]

        E_storage = [pulp.LpVariable(f"E_storage_{t}", lowBound=self.E_MIN, upBound=self.E_UPPER)
                    for t in range(remaining_periods + 1)]

        is_charging = [pulp.LpVariable(f"is_charging_{t}", cat='Binary')
                      for t in range(remaining_periods)]

        P_breach = [pulp.LpVariable(f"P_breach_{t}", lowBound=0)
                   for t in range(remaining_periods)]
        P_excess = [pulp.LpVariable(f"P_excess_{t}", lowBound=0)
                   for t in range(remaining_periods)]

        if terminal_soc_weight is None:
            terminal_soc_weight = float(np.mean(price[:36]) / self.ETA_CH)

        normal_cost = pulp.lpSum([
            P_adjusted[t] * self.DELTA_T * price[start_period + t]
            for t in range(remaining_periods)
        ])
        excess_cost_expr = pulp.lpSum([
            P_excess[t] * self.DELTA_T * price[start_period + t] * 0.5
            for t in range(remaining_periods)
        ])
        breach_cost_expr = pulp.lpSum([
            P_breach[t] * self.DELTA_T * price[start_period + t] * 0.5
            for t in range(remaining_periods)
        ])

        total_cost = normal_cost + excess_cost_expr - terminal_soc_weight * E_storage[remaining_periods]
        model += total_cost

        # 1. 当前储能初始化
        model += E_storage[0] == E_current

        for t in range(remaining_periods):
            # 2. 光伏分配约束（使用新预报，无数据泄漏）
            model += P_pv_to_load[t] + P_pv_to_storage[t] <= pv_forecast[t]

            # 3. 违约/超出定义（相对于上一决策步骤的计划）
            model += P_breach[t] >= P_plan[start_period + t] - P_adjusted[t]
            model += P_excess[t] >= P_adjusted[t] - P_plan[start_period + t]

            # 4. 功率平衡（使用预报负载）
            model += (
                P_adjusted[t] + P_pv_to_load[t] + P_discharge[t] >=
                load_forecast[t] + P_charge[t] - P_pv_to_storage[t]
            )

            # 5. 储能动态
            model += (
                E_storage[t + 1] == E_storage[t] +
                (P_charge[t] * self.ETA_CH - P_discharge[t] / self.ETA_DIS) * self.DELTA_T
            )

            # 6. 充放电互斥
            model += P_charge[t] <= self.P_MAX * is_charging[t]
            model += P_discharge[t] <= self.P_MAX * (1 - is_charging[t])

        # 7. 终态储能不低于下限
        model += E_storage[remaining_periods] >= self.E_MIN

        solver = pulp.getSolver(solver_name, msg=False)
        model.solve(solver)

        if model.status != pulp.LpStatusOptimal:
            return None

        result = {
            'P_adjusted': np.array([P_adjusted[t].varValue for t in range(remaining_periods)]),
            'P_charge': np.array([P_charge[t].varValue for t in range(remaining_periods)]),
            'P_discharge': np.array([P_discharge[t].varValue for t in range(remaining_periods)]),
            'P_pv_to_storage': np.array([P_pv_to_storage[t].varValue for t in range(remaining_periods)]),
            'P_pv_to_load': np.array([P_pv_to_load[t].varValue for t in range(remaining_periods)]),
            'E_storage': np.array([E_storage[t].varValue for t in range(remaining_periods + 1)]),
            'P_breach': np.array([P_breach[t].varValue for t in range(remaining_periods)]),
            'P_excess': np.array([P_excess[t].varValue for t in range(remaining_periods)]),
            'breach_cost': pulp.value(breach_cost_expr),
            'excess_cost': pulp.value(excess_cost_expr),
            'total_cost': pulp.value(total_cost),
            'status': 'Optimal'
        }

        return result

"""
问题1：典型日优化（改进模型）
"""
import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from data_loader import DataLoader
from optimizer_improved_p2 import ImprovedProblem2Optimizer
from result_saver import save_problem1_results_v2

def solve_problem1(data_dir, output_dir, template_dir):
    loader = DataLoader(data_dir)
    data = loader.load_attachment1()

    price = data['price']
    load = data['load']
    pv_forecast = data['pv_forecast']

    optimizer = ImprovedProblem2Optimizer()
    # 问题1为典型日，强制要求终态储能回到初始值6000kWh，不需要终态价值激励
    results = optimizer.solve(price, load, pv_forecast, E_init=6000,
                              terminal_soc_weight=0.0, E_terminal=6000)

    if results is None:
        print("优化失败！")
        return None

    print(f"总购电费: {results['total_cost']:.2f} 元")

    template_path = Path(template_dir) / "result1.xlsx"
    output_path = Path(output_dir) / "result1_improved.xlsx"
    save_problem1_results_v2(results, output_path, template_path)
    print(f"结果已保存: {output_path}")

    return results

if __name__ == "__main__":
    base = Path(__file__).parent.parent
    results = solve_problem1(base / "附件", base / "results", base / "附件" / "附件5")
    if results:
        print("✓ 问题1求解完成")
    else:
        print("✗ 问题1求解失败")

"""
使用改进模型求解问题2（334天完整版）
光伏/负载预测：从 results/forecasts_p2.npz 读取预计算结果
（预计算由 precompute_forecasts.py 完成，运行一次即可）
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import time

warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent))
from data_loader import DataLoader
from optimizer_improved_p2 import ImprovedProblem2Optimizer
from result_saver import save_problem2_results_v2

def load_forecasts(forecast_path):
    """
    读取预计算的光伏/负载预测结果。

    Returns:
        pv_forecast   : (334, 144) ndarray
        load_forecast : (334, 144) ndarray
        dates         : (334,) list of pd.Timestamp
    """
    data = np.load(forecast_path, allow_pickle=True)
    pv_forecast   = data['pv_forecast']    # (334, 144)
    load_forecast = data['load_forecast']  # (334, 144)
    dates = [pd.Timestamp(d) for d in data['dates']]
    return pv_forecast, load_forecast, dates

def solve_problem2_improved(data_dir, output_dir, template_dir, forecast_path):
    """
    使用改进模型求解问题2（两阶段：计划用预测，执行用真实数据）

    Args:
        data_dir:      附件数据目录
        output_dir:    结果输出目录
        template_dir:  Excel 模板目录
        forecast_path: 预计算预测文件路径（forecasts_p2.npz）
    """
    print("=" * 60)
    print("问题2：改进模型全年优化（334天）")
    print("=" * 60)

    start_time = time.time()

    # ── 1. 加载数据 ──────────────────────────────────────────
    print("\n[1/6] 加载数据...")
    loader = DataLoader(data_dir)

    data1 = loader.load_attachment1()
    price = data1['price']          # (144,)

    data2 = loader.load_attachment2()
    load_all  = data2['load']        # (365, 144) 真实负载
    pv_all    = data2['pv_actual']   # (365, 144) 真实光伏
    dates_all = data2['dates']

    print(f"  电价: {len(price)} 个时段")
    print(f"  负载/光伏数据: {load_all.shape}")
    print(f"  日期范围: {dates_all.iloc[0]} 到 {dates_all.iloc[-1]}")

    # ── 2. 读取预计算预测 ─────────────────────────────────────
    print("\n[2/6] 读取预计算预测（forecasts_p2.npz）...")
    forecast_path = Path(forecast_path)
    if not forecast_path.exists():
        raise FileNotFoundError(
            f"预测文件不存在: {forecast_path}\n"
            "请先运行 python src/precompute_forecasts.py"
        )
    pv_fc, load_fc, forecast_dates = load_forecasts(forecast_path)
    print(f"  pv_forecast   shape: {pv_fc.shape}")
    print(f"  load_forecast shape: {load_fc.shape}")
    print(f"  日期范围: {forecast_dates[0].strftime('%Y-%m-%d')} "
          f"~ {forecast_dates[-1].strftime('%Y-%m-%d')}")

    # ── 3. 确定优化范围 ───────────────────────────────────────
    print("\n[3/6] 确定优化范围...")
    start_date = pd.Timestamp('2025-02-01')
    start_idx  = np.where(dates_all == start_date)[0][0]

    pv_data   = pv_all[start_idx:]    # (334, 144) 真实光伏（执行阶段用）
    load_data = load_all[start_idx:]  # (334, 144) 真实负载（执行阶段用）
    n_days    = len(pv_data)

    print(f"  优化日期: 2025-02-01 到 2025-12-31，共 {n_days} 天")
    assert len(forecast_dates) == n_days, \
        f"预测天数({len(forecast_dates)})与优化天数({n_days})不匹配"

    # ── 4. 创建优化器 ─────────────────────────────────────────
    print("\n[4/6] 创建改进优化器...")
    optimizer = ImprovedProblem2Optimizer()

    # ── 5. 优化1月（推算2月1日初始储能，不计入输出）────────────
    print("\n[5/6] 优化1月份（结果不计入输出）...")
    E_init = 6000.0  # 2025-01-01 初始储能 kWh

    for day_idx in range(start_idx):
        if (day_idx + 1) % 10 == 0:
            print(f"    1月进度: {day_idx+1}/{start_idx}")
        result = optimizer.solve(price, load_all[day_idx], pv_all[day_idx], E_init)
        if result:
            E_init = result['E_storage'][-1]
        else:
            print(f"  警告: 1月第{day_idx+1}天优化失败")

    print(f"  2月1日初始储能: {E_init:.2f} kWh")

    # ── 6. 逐天两阶段优化（2月～12月）────────────────────────
    print("\n[6/6] 开始逐天两阶段优化（334天）...")
    results_all_days = []

    for i, date in enumerate(forecast_dates):
        if (i + 1) % 30 == 0 or i == 0:
            elapsed   = time.time() - start_time
            avg       = elapsed / (i + 1 + start_idx)
            remaining = avg * (n_days - i - 1)
            print(f"  进度: {i+1}/{n_days} ({(i+1)/n_days*100:.1f}%)"
                  f" - {date.strftime('%Y-%m-%d')}"
                  f" - 预计剩余: {remaining/60:.1f}分钟")

        pv_forecast_today   = pv_fc[i]        # 0:00时刻的预测光伏
        load_forecast_today = load_fc[i]      # 0:00时刻的预测负载
        pv_actual_today     = pv_data[i]      # 当天真实光伏（执行阶段用）
        load_actual_today   = load_data[i]    # 当天真实负载（执行阶段用）

        # ── 阶段1：计划（预测数据） ──────────────────────────
        plan = optimizer.solve(price, load_forecast_today, pv_forecast_today, E_init)
        if plan is None:
            print(f"  警告: {date.strftime('%Y-%m-%d')} 计划阶段失败，跳过")
            continue
        P_buy_plan = plan['P_buy']

        # ── 阶段2：执行（真实数据，P_buy 锁定）──────────────
        result = optimizer.simulate_execution(
            price, load_actual_today, pv_actual_today, E_init, P_buy_plan)
        if result is None:
            print(f"  警告: {date.strftime('%Y-%m-%d')} 执行阶段失败，跳过")
            continue
        # simulate_execution 不会返回 None（纯规则仿真，始终成功）

        result['date'] = date
        results_all_days.append(result)

        # 用执行阶段的真实终态储能更新下一天初值
        E_init = result['E_storage'][-1]

    print(f"\n  完成优化: {len(results_all_days)}/{n_days} 天")
    print(f"  总计算时间: {(time.time()-start_time)/60:.1f} 分钟")

    # ── 保存结果 ──────────────────────────────────────────────
    print("\n保存结果...")
    template_path = Path(template_dir) / "result2.xlsx"
    output_path   = Path(output_dir) / "result2_improved.xlsx"
    save_problem2_results_v2(results_all_days, output_path, template_path)
    print(f"  结果已保存: {output_path}")

    # ── 统计汇总 ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("结果汇总")
    print("=" * 60)

    total_cost           = sum(r['total_cost']     for r in results_all_days)
    total_planned_cost   = sum(r['planned_cost']   for r in results_all_days)
    total_emergency_cost = sum(r['emergency_cost'] for r in results_all_days)
    total_purchase       = sum(r['total_purchase'] for r in results_all_days)
    emergency_days       = sum(1 for r in results_all_days if np.sum(r['P_emergency']) > 1e-6)

    print(f"  总购电费用:       {total_cost:.2f} 元")
    print(f"    其中计划购电费: {total_planned_cost:.2f} 元")
    print(f"    其中紧急购电费: {total_emergency_cost:.2f} 元")
    print(f"  总购电量:         {total_purchase:.2f} kWh")
    print(f"  紧急购电天数:     {emergency_days}/{len(results_all_days)} 天")
    print(f"  平均日购电费用:   {total_cost/len(results_all_days):.2f} 元/天")

    specified_dates = ['2025-03-20', '2025-06-21', '2025-09-23', '2025-12-21']
    print("\n指定日期结果:")
    print("-" * 60)
    for ds in specified_dates:
        target = pd.Timestamp(ds)
        matched = [r for r in results_all_days if r['date'] == target]
        if matched:
            r = matched[0]
            pv_to_storage = np.sum(r['P_pv_to_storage'] * (1/6))
            print(f"  {ds}:")
            print(f"    购电量: {r['total_purchase']:.2f} kWh  "
                  f"购电费: {r['total_cost']:.2f} 元  "
                  f"光伏充储: {pv_to_storage:.2f} kWh")
            if r['emergency_cost'] > 0:
                print(f"    紧急购电费: {r['emergency_cost']:.2f} 元")

    return results_all_days

if __name__ == "__main__":
    current_dir   = Path(__file__).parent.parent
    data_dir      = current_dir / "附件"
    output_dir    = current_dir / "results"
    template_dir  = current_dir / "附件" / "附件5"
    forecast_path = current_dir / "results" / "forecasts_p2.npz"

    output_dir.mkdir(exist_ok=True)

    results = solve_problem2_improved(data_dir, output_dir, template_dir, forecast_path)

    if results:
        print("\n✓ 问题2（改进版）求解完成！")
    else:
        print("\n✗ 问题2（改进版）求解失败！")


"""
问题3求解脚本：带预报更新的滚动优化
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import time

sys.path.append(str(Path(__file__).parent))

from data_loader import DataLoader
from optimizer_improved_p2 import ImprovedProblem2Optimizer
from optimizer_problem3 import Problem3Optimizer
from result_saver_p3 import save_problem3_results

def partial_hourly_to_10min(hourly_partial, n_hours):
    """
    将 n_hours 个整点预报插值到 n_hours*6 个10分钟时段。
    hourly_partial: 从某调整时刻起的光伏预报, 取前 n_hours 个值
    """
    n_periods = n_hours * 6
    pts_hour = np.arange(n_hours + 1, dtype=float)
    vals_ext = np.append(hourly_partial[:n_hours], hourly_partial[n_hours - 1])
    pts_10min = np.arange(n_periods) / 6.0
    return np.interp(pts_10min, pts_hour, vals_ext)

def greedy_simulate(pv_actual, load_actual, P_buy, E_start, t_start, t_end, opt):
    """
    在 [t_start, t_end) 时段内执行 greedy 仿真。
    P_buy 已锁定，使用真实 PV 和负载，跟踪真实 SOC。
    超过放电能力的缺口记为紧急购电。
    """
    n = t_end - t_start
    P_ch  = np.zeros(n)
    P_dis = np.zeros(n)
    P_emg = np.zeros(n)
    E     = np.zeros(n + 1)
    E[0]  = E_start

    for j in range(n):
        t   = t_start + j
        net = P_buy[t] + pv_actual[t] - load_actual[t]

        if net >= 0.0:
            # 盈余 → 充电（受功率和容量限制）
            e_room = (opt.E_UPPER - E[j]) / opt.ETA_CH / opt.DELTA_T
            p_ch   = min(net, opt.P_MAX, max(e_room, 0.0))
            P_ch[j] = p_ch
            E[j+1]  = E[j] + p_ch * opt.ETA_CH * opt.DELTA_T
        else:
            # 缺口 → 放电，不足则紧急购电
            need    = -net
            e_avail = (E[j] - opt.E_MIN) * opt.ETA_DIS / opt.DELTA_T
            p_dis   = min(need, opt.P_MAX, max(e_avail, 0.0))
            P_dis[j] = p_dis
            E[j+1]   = E[j] - p_dis / opt.ETA_DIS * opt.DELTA_T
            gap = need - p_dis
            if gap > 1e-6:
                P_emg[j] = gap

    return {'P_charge': P_ch, 'P_discharge': P_dis, 'P_emergency': P_emg, 'E_storage': E}

def solve_problem3(data_dir, output_dir, template_dir, test_mode=False):
    """
    求解问题3：带预报更新的购电策略优化

    Args:
        data_dir: 数据目录
        output_dir: 输出目录
        template_dir: 模板目录
        test_mode: 是否为测试模式（只运行前几天）

    Returns:
        list: 所有天的优化结果
    """
    print("=" * 60)
    print("问题3：带预报更新的滚动优化（334天）")
    print("=" * 60)

    start_time = time.time()

    # 1. 加载数据
    print("\n[1/7] 加载数据...")
    loader = DataLoader(data_dir)

    # 加载电价
    data1 = loader.load_attachment1()
    price = data1['price']
    print(f"  电价数据: {len(price)} 个时段")

    # 加载附件2：实际负载和光伏
    data2 = loader.load_attachment2()
    load_all = data2['load']  # (365, 144)
    pv_all = data2['pv_actual']  # (365, 144)
    dates_all = data2['dates']

    print(f"  负载数据: {load_all.shape}")
    print(f"  光伏数据: {pv_all.shape}")

    # 加载附件3：光伏预报
    data3 = loader.load_attachment3()
    forecasts = data3['forecasts']
    print(f"  预报数据: {len(forecasts)} 天")

    # 加载预计算 SARIMA 负载预报（与问题2共用）
    forecasts_npz_path = Path(output_dir) / 'forecasts_p2.npz'
    if not forecasts_npz_path.exists():
        raise FileNotFoundError(
            f"缺少预计算文件 {forecasts_npz_path}，请先运行 precompute_forecasts.py"
        )
    fc_npz = np.load(forecasts_npz_path, allow_pickle=True)
    fc_dates = fc_npz['dates']
    fc_pv = fc_npz['pv_forecast']      # (334, 144)
    fc_load = fc_npz['load_forecast']   # (334, 144)
    fc_date_map = {pd.Timestamp(str(d)).normalize(): idx for idx, d in enumerate(fc_dates)}
    print(f"  负载预报数据: {fc_load.shape}")

    # 2. 确定优化范围（2月1日到12月31日）
    print("\n[2/7] 确定优化范围...")
    start_date = pd.Timestamp('2025-02-01')
    start_idx = np.where(dates_all == start_date)[0][0]

    dates = dates_all.iloc[start_idx:].reset_index(drop=True)
    load_data = load_all[start_idx:]
    pv_data = pv_all[start_idx:]

    n_days = len(dates) if not test_mode else 5
    print(f"  优化范围: 2025-02-01 到 2025-12-31")
    print(f"  {'测试模式' if test_mode else '完整模式'}: {n_days} 天")

    # 3. 创建优化器
    print("\n[3/7] 创建优化器...")
    optimizer = Problem3Optimizer()
    warmup_optimizer = ImprovedProblem2Optimizer()

    # 4. 优化1月份（结果不写入输出，仅推算2月1日初始储能）
    print("\n[4/7] 优化1月份（结果不计入输出）...")
    E_init = 6000

    print("  优化1月份（31天）...")
    for day_idx in range(start_idx):
        if (day_idx + 1) % 10 == 0:
            print(f"    1月进度: {day_idx+1}/{start_idx} 天")

        date = dates_all.iloc[day_idx]
        load_jan = load_all[day_idx]
        pv_jan = pv_all[day_idx]

        result = warmup_optimizer.solve(price, load_jan, pv_jan, E_init)

        if result:
            E_init = result['E_storage'][-1]
        else:
            print(f"  警告: 1月第{day_idx+1}天({date.strftime('%Y-%m-%d')})优化失败")

    print(f"  2月1日初始储能: {E_init:.2f} kWh")

    # 5. 逐天求解2月-12月（滚动优化）
    print("\n[5/7] 开始滚动优化...")
    results_all_days = []

    for i in range(n_days):
        date = dates[i]

        if (i + 1) % 30 == 0 or i == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1 + start_idx)
            remaining_time = avg_time * (n_days - i - 1)
            print(f"  进度: {i+1}/{n_days} ({(i+1)/n_days*100:.1f}%) - {date.strftime('%Y-%m-%d')} - 预计剩余: {remaining_time/60:.1f}分钟")

        # 当天真实数据（仅用于 greedy 仿真，不用于优化决策）
        load_actual = load_data[i]    # (144,) 真实负载
        pv_actual   = pv_data[i]      # (144,) 真实光伏

        # ── 第一步：0:00 根据预报制定全天计划 ──────────────────────
        forecast_0h = forecasts.get(date, {}).get('0:00')
        if forecast_0h is None:
            print(f"  警告: {date.strftime('%Y-%m-%d')} 缺少0:00预报，跳过")
            continue

        # 当天 ARIMA 负载预报（用于优化决策）
        fc_idx = fc_date_map.get(pd.Timestamp(date).normalize())
        load_fc = load_actual if fc_idx is None else fc_load[fc_idx]

        planned_result = optimizer.solve_planned(price, load_fc, forecast_0h, E_init)
        if planned_result is None:
            print(f"  警告: {date.strftime('%Y-%m-%d')} 计划优化失败")
            continue

        # P_final：当前最优购电计划，随滚动调整更新
        P_final = planned_result['P_plan'].copy()

        # 全天结果数组（由 greedy 仿真逐段填充）
        P_charge_final    = np.zeros(144)
        P_discharge_final = np.zeros(144)
        P_emergency_final = np.zeros(144)
        E_storage_final   = np.zeros(145)
        E_storage_final[0] = E_init

        # 各调整步骤 breach/excess 费用累加
        total_breach_cost   = 0.0
        total_excess_cost   = 0.0

        # ── 第二步：greedy 仿真 0:00→6:00（P_buy 锁定为计划值）──
        seg = greedy_simulate(pv_actual, load_actual, P_final, E_init, 0, 36, optimizer)
        P_charge_final[0:36]    = seg['P_charge']
        P_discharge_final[0:36] = seg['P_discharge']
        P_emergency_final[0:36] = seg['P_emergency']
        E_storage_final[0:37]   = seg['E_storage']

        # ── 第三步：6:00/12:00/18:00 滚动调整 + greedy 仿真 ──────
        # (period_start, forecast_key, remaining_hours, seg_end)
        adjustment_steps = [
            (36,  '6:00',  18, 72),
            (72,  '12:00', 12, 108),
            (108, '18:00', 6,  144),
        ]

        for period_start, time_str, remaining_hours, seg_end in adjustment_steps:
            current_E = E_storage_final[period_start]  # 来自 greedy 仿真的真实 SOC

            forecast_t = forecasts.get(date, {}).get(time_str)
            if forecast_t is not None:
                # 将整点预报插值为 10 分钟值（仅取当天剩余小时）
                pv_fc_rem = partial_hourly_to_10min(
                    forecast_t[:remaining_hours], remaining_hours
                )  # shape: (remaining_hours*6,) = (remaining_periods,)

                # 调整剩余时段购电计划（使用预报，无数据泄漏）
                adjusted_result = optimizer.solve_adjusted(
                    price,
                    load_fc[period_start:],   # ARIMA 负载预报（剩余时段）
                    pv_fc_rem,                 # 光伏预报（剩余时段，已插值）
                    P_final,                   # 上一步计划（违约/超出基准）
                    current_E,
                    period_start
                )
                if adjusted_result is not None:
                    P_final[period_start:] = adjusted_result['P_adjusted']
                    total_breach_cost += adjusted_result['breach_cost']
                    total_excess_cost  += adjusted_result['excess_cost']

            # greedy 仿真本时段（使用更新后的 P_final 和真实数据）
            seg = greedy_simulate(
                pv_actual, load_actual, P_final,
                current_E, period_start, seg_end, optimizer
            )
            P_charge_final[period_start:seg_end]      = seg['P_charge']
            P_discharge_final[period_start:seg_end]   = seg['P_discharge']
            P_emergency_final[period_start:seg_end]   = seg['P_emergency']
            E_storage_final[period_start:seg_end + 1] = seg['E_storage']

        # ── 费用计算 ─────────────────────────────────────────────
        # 计划购电费（0:00 时刻计划，用于"计划购电量"表格报告）
        total_cost_plan = float(np.sum(planned_result['P_plan'] * price * (1.0 / 6)))

        adj_breach = total_breach_cost
        adj_excess = total_excess_cost

        cost_0_6 = np.sum(planned_result['P_plan'][:36] * price[:36] * (1/6))
        cost_6_24 = np.sum(P_final[36:] * price[36:] * (1/6))
        emergency_cost_sim = np.sum(P_emergency_final * price * optimizer.EMERGENCY_RATIO * optimizer.DELTA_T)
        total_cost_final = cost_0_6 + cost_6_24 + adj_excess - adj_breach + emergency_cost_sim
        total_emergency_cost = float(emergency_cost_sim)

        result_day = {
            'date': date,
            'P_plan': planned_result['P_plan'],
            'P_final': P_final,
            'P_charge': planned_result['P_charge'],
            'P_discharge': planned_result['P_discharge'],
            'P_charge_final': P_charge_final,
            'P_discharge_final': P_discharge_final,
            'E_storage': planned_result['E_storage'],
            'E_storage_final': E_storage_final,
            'P_emergency': P_emergency_final,
            'total_cost': total_cost_plan,
            'total_cost_final': total_cost_final,
            'adjustment_costs': {
                'breach': total_breach_cost,
                'excess': total_excess_cost,
                'emergency': total_emergency_cost,
            }
        }

        results_all_days.append(result_day)

        # 下一天初始 SOC = 今天 greedy 仿真最终 SOC
        E_init = E_storage_final[-1]

    print(f"\n  完成优化: {len(results_all_days)}/{n_days} 天")

    total_time = time.time() - start_time
    print(f"  总计算时间: {total_time/60:.1f} 分钟")

    # 6. 保存结果
    print("\n[6/7] 保存结果...")
    template_path = Path(template_dir) / "result3.xlsx"
    output_path = Path(output_dir) / "result3.xlsx"

    save_problem3_results(results_all_days, output_path, template_path)

    # 7. 统计汇总
    print("\n[7/7] 结果汇总")
    print("=" * 60)

    total_cost = sum([r['total_cost_final'] for r in results_all_days])
    total_purchase = sum([np.sum(r['P_final'] * (1/6)) for r in results_all_days])

    print(f"  总购电费用: {total_cost:.2f} 元")
    print(f"  总购电量: {total_purchase:.2f} kWh")
    print(f"  平均日购电费用: {total_cost/len(results_all_days):.2f} 元/天")

    # 打印指定日期的结果
    specified_dates = ['2025-03-20', '2025-06-21', '2025-09-23', '2025-12-21']
    print("\n指定日期的结果:")
    print("-" * 60)

    for date_str in specified_dates:
        target_date = pd.Timestamp(date_str)
        matching_results = [r for r in results_all_days if r['date'] == target_date]

        if matching_results:
            result = matching_results[0]
            purchase_kwh = np.sum(result['P_final'] * (1/6))
            print(f"  {date_str}:")
            print(f"    购电量: {purchase_kwh:.2f} kWh")
            print(f"    购电费: {result['total_cost_final']:.2f} 元")

    return results_all_days

if __name__ == "__main__":
    # 设置路径
    current_dir = Path(__file__).parent.parent
    data_dir = current_dir / "附件"
    output_dir = current_dir / "results"
    template_dir = current_dir / "附件" / "附件5"

    # 确保输出目录存在
    output_dir.mkdir(exist_ok=True)

    # 检查是否为测试模式
    test_mode = '--test' in sys.argv

    # 求解问题3
    results = solve_problem3(data_dir, output_dir, template_dir, test_mode=test_mode)

    if results:
        print("\n问题3求解完成！")
    else:
        print("\n问题3求解失败！")

"""
问题4-2：使用波动电价（附件4）重做问题2（两阶段模式）

计划阶段：用 ARIMA 预测电价（forecasts_p4.npz）+ 预测光伏/负载（forecasts_p2.npz）
执行阶段：用附件4真实电价 + 附件2真实光伏/负载

预测文件分别由 precompute_forecasts_p4.py 和 precompute_forecasts.py 生成，运行一次即可。
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import time

warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent))

from data_loader import DataLoader
from optimizer_improved_p2 import ImprovedProblem2Optimizer
from result_saver import save_problem2_results_v2

def load_price_forecasts(forecast_path):
    """
    读取预计算的波动电价预测结果。

    Returns:
        price_forecast : (334, 144) ndarray
        dates          : (334,) list of pd.Timestamp
    """
    data = np.load(forecast_path, allow_pickle=True)
    price_forecast = data['price_forecast']   # (334, 144)
    dates = [pd.Timestamp(d) for d in data['dates']]
    return price_forecast, dates

def load_pv_load_forecasts(forecast_path):
    """
    读取预计算的光伏/负载预测结果（来自 forecasts_p2.npz）。

    Returns:
        pv_forecast   : (334, 144) ndarray
        load_forecast : (334, 144) ndarray
        dates         : (334,) list of pd.Timestamp
    """
    data = np.load(forecast_path, allow_pickle=True)
    pv_forecast   = data['pv_forecast']    # (334, 144)
    load_forecast = data['load_forecast']  # (334, 144)
    dates = [pd.Timestamp(d) for d in data['dates']]
    return pv_forecast, load_forecast, dates

def solve_problem4_2(data_dir, output_dir, template_dir, forecast_path,
                     pv_load_forecast_path=None, test_mode=False):
    """
    使用改进模型求解问题4-2（两阶段：计划用预测电价+预测光伏/负载，执行用真实数据）

    Args:
        data_dir:              附件数据目录
        output_dir:            结果输出目录
        template_dir:          Excel 模板目录
        forecast_path:         预计算电价预测文件路径（forecasts_p4.npz）
        pv_load_forecast_path: 预计算光伏/负载预测文件路径（forecasts_p2.npz），
                               None 时计划阶段也使用真实光伏/负载
        test_mode:             True 时只跑前5天
    """
    print("=" * 60)
    print("问题4-2：波动电价 + 两阶段改进模型优化（334天）")
    print("=" * 60)

    start_time = time.time()

    # ── 1. 加载数据 ──────────────────────────────────────────
    print("\n[1/6] 加载数据...")
    loader = DataLoader(data_dir)

    # 附件4波动电价（真实值，执行阶段使用）
    data4 = loader.load_attachment4()
    price_all = data4['prices']   # (365, 144)
    print(f"  波动电价数据: {price_all.shape}  均值={price_all.mean():.4f} 元/kWh")

    # 附件2光伏 + 负载（真实值，执行阶段使用）
    data2 = loader.load_attachment2()
    load_all  = data2['load']        # (365, 144)
    pv_all    = data2['pv_actual']   # (365, 144)
    dates_all = data2['dates']

    print(f"  负载数据: {load_all.shape}")
    print(f"  光伏数据: {pv_all.shape}")

    # ── 2. 读取预计算电价预测 ─────────────────────────────────
    print("\n[2/6] 读取预计算预测（forecasts_p4.npz + forecasts_p2.npz）...")
    forecast_path = Path(forecast_path)
    if not forecast_path.exists():
        raise FileNotFoundError(
            f"预测文件不存在: {forecast_path}\n"
            "请先运行 python src/precompute_forecasts_p4.py"
        )
    price_fc, forecast_dates = load_price_forecasts(forecast_path)
    print(f"  price_forecast shape: {price_fc.shape}")
    print(f"  日期范围: {forecast_dates[0].strftime('%Y-%m-%d')} "
          f"~ {forecast_dates[-1].strftime('%Y-%m-%d')}")

    # 读取光伏/负载预测（来自问题2预计算结果）
    p2_fc_path = Path(pv_load_forecast_path) if pv_load_forecast_path else None
    if p2_fc_path is None:
        p2_fc_path = forecast_path.parent / "forecasts_p2.npz"
    if not p2_fc_path.exists():
        raise FileNotFoundError(
            f"光伏/负载预测文件不存在: {p2_fc_path}\n"
            "请先运行 python src/precompute_forecasts.py"
        )
    pv_fc, load_fc, _ = load_pv_load_forecasts(p2_fc_path)
    print(f"  pv_forecast   shape: {pv_fc.shape}")
    print(f"  load_forecast shape: {load_fc.shape}")

    # ── 3. 确定优化范围 ───────────────────────────────────────
    print("\n[3/6] 确定优化范围...")
    start_date = pd.Timestamp('2025-02-01')
    start_idx  = int(np.where(dates_all == start_date)[0][0])
    n_days     = len(dates_all) - start_idx   # 334

    if test_mode:
        n_days = 5

    assert len(forecast_dates) >= n_days, \
        f"预测天数({len(forecast_dates)})少于优化天数({n_days})"

    print(f"  优化范围: 2025-02-01 起 {'测试' if test_mode else '完整'} {n_days} 天")

    # ── 4. 创建优化器 ─────────────────────────────────────────
    print("\n[4/6] 创建改进优化器...")
    optimizer = ImprovedProblem2Optimizer()

    # ── 5. 优化1月（推算2月1日初始储能，不计入输出）────────────
    print("\n[5/6] 优化1月份（结果不计入输出）...")
    E_init = 6000.0   # 2025-01-01 初始储能 kWh

    for day_idx in range(start_idx):
        if (day_idx + 1) % 10 == 0:
            print(f"    1月进度: {day_idx+1}/{start_idx}")

        price_jan = price_all[day_idx]
        load_jan  = load_all[day_idx]
        pv_jan    = pv_all[day_idx]

        result = optimizer.solve(price_jan, load_jan, pv_jan, E_init)
        if result:
            E_init = result['E_storage'][-1]
        else:
            print(f"  警告: 1月第{day_idx+1}天优化失败")

    print(f"  2月1日初始储能: {E_init:.2f} kWh")

    # ── 6. 逐天两阶段优化（2月～12月）────────────────────────
    print("\n[6/6] 开始逐天两阶段优化（334天）...")
    results_all_days = []

    for i in range(n_days):
        day_idx = start_idx + i
        date    = dates_all.iloc[day_idx]

        if (i + 1) % 30 == 0 or i == 0:
            elapsed   = time.time() - start_time
            avg       = elapsed / (i + 1 + start_idx)
            remaining = avg * (n_days - i - 1)
            print(f"  进度: {i+1}/{n_days} ({(i+1)/n_days*100:.1f}%)"
                  f" - {date.strftime('%Y-%m-%d')}"
                  f" - 预计剩余: {remaining/60:.1f}分钟")

        price_forecast_today = price_fc[i]          # 预测电价（计划阶段）
        price_actual_today   = price_all[day_idx]   # 真实电价（执行阶段）
        load_forecast_today  = load_fc[i]           # 预测负载（计划阶段）
        pv_forecast_today    = pv_fc[i]             # 预测光伏（计划阶段）
        load_actual_today    = load_all[day_idx]    # 真实负载（执行阶段）
        pv_actual_today      = pv_all[day_idx]      # 真实光伏（执行阶段）

        # ── 阶段1：计划（预测电价 + 预测光伏/负载）──────────────
        plan = optimizer.solve(price_forecast_today, load_forecast_today, pv_forecast_today, E_init)
        if plan is None:
            print(f"  警告: {date.strftime('%Y-%m-%d')} 计划阶段失败，跳过")
            continue
        P_buy_plan = plan['P_buy']

        # ── 阶段2：执行（真实电价+真实光伏/负载，P_buy 锁定计划值）──
        result = optimizer.simulate_execution(
            price_actual_today, load_actual_today, pv_actual_today, E_init, P_buy_plan)
        if result is None:
            print(f"  警告: {date.strftime('%Y-%m-%d')} 执行阶段失败，跳过")
            continue

        result['date'] = date
        results_all_days.append(result)

        # 用执行阶段的真实终态储能更新下一天初值
        E_init = result['E_storage'][-1]

    print(f"\n  完成优化: {len(results_all_days)}/{n_days} 天")
    print(f"  总计算时间: {(time.time()-start_time)/60:.1f} 分钟")

    # ── 保存结果 ──────────────────────────────────────────────
    print("\n保存结果...")
    template_path = Path(template_dir) / "result2.xlsx"
    output_path   = Path(output_dir) / "result4-2.xlsx"
    save_problem2_results_v2(results_all_days, output_path, template_path)
    print(f"  结果已保存到: {output_path}")

    # ── 统计汇总 ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("结果汇总")
    print("=" * 60)

    total_cost           = sum(r['total_cost']     for r in results_all_days)
    total_planned_cost   = sum(r.get('planned_cost',   r['total_cost'])   for r in results_all_days)
    total_emergency_cost = sum(r['emergency_cost'] for r in results_all_days)
    total_purchase       = sum(r['total_purchase'] for r in results_all_days)
    emergency_days       = sum(1 for r in results_all_days if np.sum(r['P_emergency']) > 1e-6)

    print(f"  总购电费用:       {total_cost:.2f} 元")
    print(f"    其中计划购电费: {total_planned_cost:.2f} 元")
    print(f"    其中紧急购电费: {total_emergency_cost:.2f} 元")
    print(f"  总购电量:         {total_purchase:.2f} kWh")
    print(f"  紧急购电天数:     {emergency_days}/{len(results_all_days)} 天")
    print(f"  平均日购电费用:   {total_cost/len(results_all_days):.2f} 元/天")

    # 对比问题2（固定电价）
    p2_cost = 12408855.12
    diff = total_cost - p2_cost
    print(f"\n  对比问题2（固定电价）: {p2_cost:.2f} 元")
    print(f"  问题4-2费用: {total_cost:.2f} 元  差值: {diff:+.2f} 元")

    specified_dates = ['2025-03-20', '2025-06-21', '2025-09-23', '2025-12-21']
    print("\n指定日期结果:")
    print("-" * 60)
    for ds in specified_dates:
        target = pd.Timestamp(ds)
        matched = [r for r in results_all_days if r['date'] == target]
        if matched:
            r = matched[0]
            pv_to_storage = np.sum(r['P_pv_to_storage'] * (1/6))
            print(f"  {ds}:")
            print(f"    购电量: {r['total_purchase']:.2f} kWh  "
                  f"购电费: {r['total_cost']:.2f} 元  "
                  f"光伏充储: {pv_to_storage:.2f} kWh")
            if r['emergency_cost'] > 0:
                print(f"    紧急购电费: {r['emergency_cost']:.2f} 元")

    return results_all_days

if __name__ == "__main__":
    current_dir   = Path(__file__).parent.parent
    data_dir      = current_dir / "附件"
    output_dir    = current_dir / "results"
    template_dir  = current_dir / "附件" / "附件5"
    forecast_path = current_dir / "results" / "forecasts_p4.npz"

    output_dir.mkdir(exist_ok=True)

    import sys as _sys
    test_mode = '--test' in _sys.argv
    results = solve_problem4_2(data_dir, output_dir, template_dir, forecast_path, test_mode=test_mode)

    if results:
        print("\n问题4-2（两阶段版）求解完成！")
    else:
        print("\n问题4-2（两阶段版）求解失败！")

"""
问题4-3：使用波动电价（附件4）重做问题3（带滚动调整策略）
修复：消除数据泄漏，优化阶段全程使用预报，真实数据仅用于greedy仿真。
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import time

sys.path.append(str(Path(__file__).parent))

from data_loader import DataLoader
from optimizer_improved_p2 import ImprovedProblem2Optimizer
from optimizer_problem3 import Problem3Optimizer
from result_saver_p3 import save_problem3_results

def partial_hourly_to_10min(hourly_partial, n_hours):
    """将 n_hours 个整点预报插值到 n_hours*6 个10分钟时段。"""
    n_periods = n_hours * 6
    pts_hour = np.arange(n_hours + 1, dtype=float)
    vals_ext = np.append(hourly_partial[:n_hours], hourly_partial[n_hours - 1])
    pts_10min = np.arange(n_periods) / 6.0
    return np.maximum(np.interp(pts_10min, pts_hour, vals_ext), 0.0)

def greedy_simulate(pv_actual, load_actual, P_buy, E_start, t_start, t_end, opt):
    """greedy执行仿真：P_buy锁定，用真实数据跟踪SOC，缺口记为紧急购电。"""
    n = t_end - t_start
    P_ch = np.zeros(n); P_dis = np.zeros(n); P_emg = np.zeros(n)
    E = np.zeros(n + 1); E[0] = E_start
    for j in range(n):
        t = t_start + j
        net = P_buy[t] + pv_actual[t] - load_actual[t]
        if net >= 0.0:
            e_room = (opt.E_UPPER - E[j]) / opt.ETA_CH / opt.DELTA_T
            p_ch = min(net, opt.P_MAX, max(e_room, 0.0))
            P_ch[j] = p_ch
            E[j+1] = E[j] + p_ch * opt.ETA_CH * opt.DELTA_T
        else:
            need = -net
            e_avail = (E[j] - opt.E_MIN) * opt.ETA_DIS / opt.DELTA_T
            p_dis = min(need, opt.P_MAX, max(e_avail, 0.0))
            P_dis[j] = p_dis
            E[j+1] = E[j] - p_dis / opt.ETA_DIS * opt.DELTA_T
            gap = need - p_dis
            if gap > 1e-6:
                P_emg[j] = gap
    return {'P_charge': P_ch, 'P_discharge': P_dis, 'P_emergency': P_emg, 'E_storage': E}

def solve_problem4_3(data_dir, output_dir, template_dir, test_mode=False):
    print("=" * 60)
    print("问题4-3：波动电价 + 滚动调整优化（334天）")
    print("=" * 60)

    start_time = time.time()

    # 1. 加载数据
    print("\n[1/7] 加载数据...")
    loader = DataLoader(data_dir)

    # 加载附件4波动电价 (365, 144)
    data4 = loader.load_attachment4()
    price_all = data4['prices']   # shape (365, 144)
    print(f"  波动电价数据: {price_all.shape}  均值={price_all.mean():.4f} 元/kWh")

    # 加载附件2
    data2 = loader.load_attachment2()
    load_all = data2['load']
    pv_all = data2['pv_actual']
    dates_all = data2['dates']
    print(f"  负载数据: {load_all.shape}")

    # 加载附件3光伏预报
    data3 = loader.load_attachment3()
    forecasts = data3['forecasts']
    print(f"  预报数据: {len(forecasts)} 天")

    # 加载预计算 SARIMA 负载预报（与问题2/3共用）
    forecasts_npz_path = Path(output_dir) / 'forecasts_p2.npz'
    if not forecasts_npz_path.exists():
        raise FileNotFoundError(
            f"缺少预计算文件 {forecasts_npz_path}，请先运行 precompute_forecasts.py"
        )
    fc_npz = np.load(forecasts_npz_path, allow_pickle=True)
    fc_dates = fc_npz['dates']
    fc_load  = fc_npz['load_forecast']   # (334, 144)
    fc_date_map = {pd.Timestamp(str(d)).normalize(): idx for idx, d in enumerate(fc_dates)}
    print(f"  负载预报数据: {fc_load.shape}")

    # 2. 确定优化范围
    print("\n[2/7] 确定优化范围...")
    start_date = pd.Timestamp('2025-02-01')
    start_idx = int(np.where(dates_all == start_date)[0][0])
    dates = dates_all.iloc[start_idx:].reset_index(drop=True)
    load_data = load_all[start_idx:]
    pv_data = pv_all[start_idx:]

    n_days = len(dates) if not test_mode else 5
    print(f"  {'测试模式' if test_mode else '完整模式'}: {n_days} 天")

    # 3. 创建优化器
    print("\n[3/7] 创建优化器...")
    optimizer = Problem3Optimizer()
    warmup_optimizer = ImprovedProblem2Optimizer()

    # 4. 优化1月份（结果不写入输出，仅推算2月1日初始储能）
    print("\n[4/7] 优化1月份（结果不计入输出）...")
    E_init = 6000.0

    for day_idx in range(start_idx):
        if (day_idx + 1) % 10 == 0:
            print(f"    1月进度: {day_idx+1}/{start_idx}")

        price_day = price_all[day_idx]
        date = dates_all.iloc[day_idx]
        load_jan = load_all[day_idx]
        pv_jan = pv_all[day_idx]

        result = warmup_optimizer.solve(price_day, load_jan, pv_jan, E_init)

        if result:
            E_init = result['E_storage'][-1]
        else:
            print(f"  警告: 1月第{day_idx+1}天({date.strftime('%Y-%m-%d')})优化失败")

    print(f"  2月1日初始储能: {E_init:.2f} kWh")

    # 5. 逐天滚动优化
    print("\n[5/7] 开始滚动优化...")
    results_all_days = []

    for i in range(n_days):
        day_idx = start_idx + i
        date = dates[i]

        if (i + 1) % 30 == 0 or i == 0:
            elapsed = time.time() - start_time
            avg = elapsed / (i + 1 + start_idx)
            remaining = avg * (n_days - i - 1)
            print(f"  进度: {i+1}/{n_days} ({(i+1)/n_days*100:.1f}%) - {date.strftime('%Y-%m-%d')} - 预计剩余: {remaining/60:.1f}分钟")

        price_day   = price_all[day_idx]
        load_actual = load_data[i]   # 真实负载（仅用于greedy仿真）
        pv_actual   = pv_data[i]     # 真实光伏（仅用于greedy仿真）

        # 当天 ARIMA 负载预报（用于优化决策）
        fc_idx  = fc_date_map.get(pd.Timestamp(date).normalize())
        load_fc = load_actual if fc_idx is None else fc_load[fc_idx]

        # 0:00制定全天计划（使用预报）
        forecast_0h = forecasts.get(date, {}).get('0:00')
        if forecast_0h is None:
            print(f"  警告: {date.strftime('%Y-%m-%d')} 缺少0:00预报，跳过")
            continue

        planned_result = optimizer.solve_planned(price_day, load_fc, forecast_0h, E_init)
        if planned_result is None:
            print(f"  警告: {date.strftime('%Y-%m-%d')} 计划优化失败")
            continue

        P_final = planned_result['P_plan'].copy()

        # 全天结果数组（由greedy仿真逐段填充）
        P_charge_final    = np.zeros(144)
        P_discharge_final = np.zeros(144)
        P_emergency_final = np.zeros(144)
        E_storage_final   = np.zeros(145)
        E_storage_final[0] = E_init

        total_breach_cost = 0.0
        total_excess_cost = 0.0

        # greedy仿真 0:00→6:00
        seg = greedy_simulate(pv_actual, load_actual, P_final, E_init, 0, 36, optimizer)
        P_charge_final[0:36]    = seg['P_charge']
        P_discharge_final[0:36] = seg['P_discharge']
        P_emergency_final[0:36] = seg['P_emergency']
        E_storage_final[0:37]   = seg['E_storage']

        # 6:00 / 12:00 / 18:00 滚动调整 + greedy仿真
        for period_start, time_str, remaining_hours, seg_end in [
            (36,  '6:00',  18, 72),
            (72,  '12:00', 12, 108),
            (108, '18:00', 6,  144),
        ]:
            current_E = E_storage_final[period_start]  # 真实SOC

            forecast_t = forecasts.get(date, {}).get(time_str)
            if forecast_t is not None:
                # 附件3预报插值为10分钟（无数据泄漏）
                pv_fc_rem = partial_hourly_to_10min(
                    forecast_t[:remaining_hours], remaining_hours
                )
                adjusted_result = optimizer.solve_adjusted(
                    price_day,
                    load_fc[period_start:],   # ARIMA负载预报
                    pv_fc_rem,                 # 光伏预报
                    P_final,
                    current_E,
                    period_start
                )
                if adjusted_result is not None:
                    P_final[period_start:] = adjusted_result['P_adjusted']
                    total_breach_cost += adjusted_result['breach_cost']
                    total_excess_cost  += adjusted_result['excess_cost']

            # greedy仿真本段
            seg = greedy_simulate(
                pv_actual, load_actual, P_final,
                current_E, period_start, seg_end, optimizer
            )
            P_charge_final[period_start:seg_end]      = seg['P_charge']
            P_discharge_final[period_start:seg_end]   = seg['P_discharge']
            P_emergency_final[period_start:seg_end]   = seg['P_emergency']
            E_storage_final[period_start:seg_end + 1] = seg['E_storage']

        # 费用计算
        cost_0_6  = float(np.sum(planned_result['P_plan'][:36] * price_day[:36] * (1/6)))
        cost_6_24 = float(np.sum(P_final[36:] * price_day[36:] * (1/6)))
        emergency_cost_sim = float(np.sum(
            P_emergency_final * price_day * optimizer.EMERGENCY_RATIO * optimizer.DELTA_T
        ))
        total_cost_final = cost_0_6 + cost_6_24 + total_excess_cost - total_breach_cost + emergency_cost_sim

        result_day = {
            'date': date,
            'P_plan': planned_result['P_plan'],
            'P_final': P_final,
            'P_charge': planned_result['P_charge'],
            'P_discharge': planned_result['P_discharge'],
            'P_charge_final': P_charge_final,
            'P_discharge_final': P_discharge_final,
            'E_storage': planned_result['E_storage'],
            'E_storage_final': E_storage_final,
            'P_emergency': P_emergency_final,
            'total_cost': float(np.sum(planned_result['P_plan'] * price_day * (1/6))),
            'total_cost_final': total_cost_final,
            'adjustment_costs': {
                'breach': total_breach_cost,
                'excess': total_excess_cost,
                'emergency': emergency_cost_sim,
            },
        }

        results_all_days.append(result_day)
        E_init = E_storage_final[-1]   # 真实仿真末尾SOC

    print(f"\n  完成优化: {len(results_all_days)}/{n_days} 天")
    print(f"  总计算时间: {(time.time()-start_time)/60:.1f} 分钟")

    # 6. 保存结果
    print("\n[6/7] 保存结果...")
    template_path = Path(template_dir) / "result3.xlsx"
    output_path = Path(output_dir) / "result4-3.xlsx"
    save_problem3_results(results_all_days, output_path, template_path)
    print(f"  结果已保存到: {output_path}")

    # 7. 汇总
    print("\n[7/7] 结果汇总")
    print("=" * 60)

    total_cost = sum(r['total_cost_final'] for r in results_all_days)
    total_purchase = sum(np.sum(r['P_final'] * (1/6)) for r in results_all_days)
    total_breach = sum(r['adjustment_costs']['breach'] for r in results_all_days)
    total_excess = sum(r['adjustment_costs']['excess'] for r in results_all_days)
    total_emergency = sum(r['adjustment_costs']['emergency'] for r in results_all_days)

    print(f"  总购电费用: {total_cost:.2f} 元")
    print(f"  总购电量: {total_purchase:.2f} kWh")
    print(f"  调整退款（违约）: -{total_breach:.2f} 元")
    print(f"  调整惩罚（超出）: +{total_excess:.2f} 元")
    print(f"  紧急购电费用: +{total_emergency:.2f} 元")
    print(f"  平均日购电费用: {total_cost/len(results_all_days):.2f} 元/天")

    p42_ref = None  # 将在结合运行时对比
    p3_cost = 12090605.18
    diff = total_cost - p3_cost
    print(f"\n  对比问题3（固定电价）: {p3_cost:.2f} 元")
    print(f"  问题4-3费用: {total_cost:.2f} 元  差值: {diff:+.2f} 元")

    specified_dates = ['2025-03-20', '2025-06-21', '2025-09-23', '2025-12-21']
    print("\n指定日期结果:")
    print("-" * 60)
    for ds in specified_dates:
        td = pd.Timestamp(ds)
        match = [r for r in results_all_days if r['date'] == td]
        if match:
            r = match[0]
            print(f"  {ds}: 购电量={np.sum(r['P_final']*(1/6)):.2f} kWh  费用={r['total_cost_final']:.2f} 元")

    return results_all_days

if __name__ == "__main__":
    current_dir = Path(__file__).parent.parent
    data_dir = current_dir / "附件"
    output_dir = current_dir / "results"
    template_dir = current_dir / "附件" / "附件5"
    output_dir.mkdir(exist_ok=True)

    test_mode = '--test' in sys.argv
    results = solve_problem4_3(data_dir, output_dir, template_dir, test_mode=test_mode)

    if results:
        print("\n[OK] 问题4-3求解完成！")
    else:
        print("\n[FAIL] 问题4-3求解失败！")



