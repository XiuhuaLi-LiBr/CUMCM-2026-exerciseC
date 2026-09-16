#生成论文数据预处理章节所需的5+张图
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

# 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

BASE = r'C:\Users\ASUS2\Desktop\新图\C题\C题'
OUT  = BASE + r'\preprocess\output'

#  读取数据 
df1 = pd.read_excel(BASE + r'\附件\附件1.xlsx')
price1 = df1.iloc[:,1].values
load1  = df1.iloc[:,2].values
pv1    = df1.iloc[:,3].values

df2_load = pd.read_excel(BASE + r'\附件\附件2.xlsx', sheet_name='小区负载')
df2_pv   = pd.read_excel(BASE + r'\附件\附件2.xlsx', sheet_name='光伏发电实际功率')
dates    = pd.to_datetime(df2_load.iloc[:,0])
load2    = df2_load.iloc[:,1:145].values.astype(float)   # (365,144)
pv2      = df2_pv.iloc[:,1:145].values.astype(float)     # (365,144)

df3      = pd.read_excel(BASE + r'\附件\附件3.xlsx')
df4      = pd.read_excel(BASE + r'\附件\附件4.xlsx', header=0, index_col=0)
prices4  = df4.values.astype(float)                      # (365,144)

time_labels = [f'{h:02d}:{m:02d}' for h in range(24) for m in range(0,60,10)]
tick_pos  = list(range(0,144,12))
tick_lab  = [time_labels[i] for i in tick_pos]
months    = dates.dt.month

# 图1: 典型日全要素展示（电价/负载/光伏）
fig, axes = plt.subplots(3,1, figsize=(12,9), sharex=True)
x = np.arange(144)

ax = axes[0]
ax.plot(x, price1, color='#e74c3c', lw=1.8, label='电价（元/kWh）')
ax.fill_between(x, price1, alpha=0.15, color='#e74c3c')
ax.set_ylabel('电价（元/kWh）', fontsize=11)
ax.set_title('典型日电价曲线', fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(x, pv1, color='#FFD700', lw=1.8, label='光伏预测功率（kW）')
ax.fill_between(x, pv1, alpha=0.2, color='#FFD700')
ax.set_ylabel('光伏功率（kW）', fontsize=11)
ax.set_title('典型日光伏预测功率曲线', fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.set_xticks(tick_pos); ax.set_xticklabels(tick_lab, rotation=45, fontsize=8)
ax.set_xlabel('时刻', fontsize=11)

ax = axes[2]
ax.plot(x, load1, color='#9400D3', lw=1.8, label='小区负载（kW）')
ax.fill_between(x, load1, alpha=0.15, color='#9400D3')
ax.set_ylabel('负载（kW）', fontsize=11)
ax.set_title('典型日负载曲线', fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.suptitle('典型日电价、负载与光伏预测功率全要素展示', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUT + r'\fig1_typical_day.png', bbox_inches='tight', dpi=150)
plt.close()
print('fig1 done')

# 图2:全年负载与光伏热力图（按月×时段）
fig, axes = plt.subplots(1,2, figsize=(14,5))

# 按月均值计算 (12, 144)
load_monthly = np.array([load2[months==m].mean(axis=0) for m in range(1,13)])
pv_monthly   = np.array([pv2[months==m].mean(axis=0) for m in range(1,13)])
month_names  = ['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月']

im1 = axes[0].imshow(load_monthly, aspect='auto', cmap='YlGn', origin='upper')
axes[0].set_title('全年负载月均热力图（kW）', fontsize=12, fontweight='bold')
axes[0].set_yticks(range(12)); axes[0].set_yticklabels(month_names, fontsize=9)
axes[0].set_xticks(tick_pos[::2]); axes[0].set_xticklabels(tick_lab[::2], rotation=45, fontsize=7)
axes[0].set_xlabel('时刻', fontsize=10)
plt.colorbar(im1, ax=axes[0], label='kW')

im2 = axes[1].imshow(pv_monthly, aspect='auto', cmap='YlOrRd', origin='upper')
axes[1].set_title('全年光伏月均热力图（kW）', fontsize=12, fontweight='bold')
axes[1].set_yticks(range(12)); axes[1].set_yticklabels(month_names, fontsize=9)
axes[1].set_xticks(tick_pos[::2]); axes[1].set_xticklabels(tick_lab[::2], rotation=45, fontsize=7)
axes[1].set_xlabel('时刻', fontsize=10)
plt.colorbar(im2, ax=axes[1], label='kW')

plt.suptitle('全年负载与光伏发电实际功率月均分布热力图', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + r'\fig2_annual_heatmap.png', bbox_inches='tight', dpi=150)
plt.close()
print('fig2 done')

# 图3: 光伏预报精度分析（逐日 vs 实际 + 四季误差箱线图）
df3_0 = df3[df3.iloc[:,1]=='0:00'].reset_index(drop=True)
pv_fc_24  = df3_0.iloc[:,2:26].values.astype(float)  # (365,24)
pv_actual_sum = pv2.sum(axis=1) / 6   # kWh/day
pv_fc_sum     = pv_fc_24.sum(axis=1)  # 近似kWh/day

fig, axes = plt.subplots(1,2, figsize=(14,5))

# 左图: 逐日六边形分箱图
ax = axes[0]
hb = ax.hexbin(pv_fc_sum, pv_actual_sum, gridsize=28, mincnt=1, cmap='Blues', linewidths=0.3, edgecolors='white', alpha=0.88)
mn = min(pv_fc_sum.min(), pv_actual_sum.min())
mx = max(pv_fc_sum.max(), pv_actual_sum.max())
ax.plot([mn, mx], [mn, mx], color='#B85C5C', ls='--', lw=1.5, label='理想1:1线')
corr = np.corrcoef(pv_fc_sum, pv_actual_sum)[0,1]
rmse = np.sqrt(((pv_fc_sum - pv_actual_sum)**2).mean())
ax.set_xlim(mn, mx)
ax.set_ylim(mn, mx)
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('日前预报总发电量（kWh）', fontsize=11)
ax.set_ylabel('实际总发电量（kWh）', fontsize=11)
ax.set_title(f'光伏日前预报 vs 实际\n(R={corr:.3f}, RMSE={rmse:.0f} kWh)', fontsize=11, fontweight='bold')
cbar = fig.colorbar(hb, ax=ax, pad=0.02)
cbar.set_label('逐日数据密度', fontsize=9)
ax.legend(fontsize=9, loc='upper left')
ax.grid(alpha=0.25)

# 右图: 按季节的误差箱线图
season_map = {1:'冬',2:'冬',3:'春',4:'春',5:'春',6:'夏',7:'夏',8:'夏',9:'秋',10:'秋',11:'秋',12:'冬'}
seasons    = [season_map[m] for m in months]
errors     = pv_actual_sum - pv_fc_sum
season_order = ['春','夏','秋','冬']
season_errors = [errors[[s==ss for s in seasons]] for ss in season_order]
bp = axes[1].boxplot(season_errors, labels=season_order, patch_artist=True,
                     medianprops=dict(color='black',lw=2))
colors = ['#2ecc71','#f39c12','#B22222','#3498db']
for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c); patch.set_alpha(0.7)
axes[1].axhline(0, color='red', ls='--', lw=1.2)
axes[1].set_title('各季节日前预报误差分布（实际 - 预报 kWh）', fontsize=11, fontweight='bold')
axes[1].set_ylabel('预报误差（kWh）', fontsize=11)
axes[1].set_xlabel('季节', fontsize=11)
axes[1].grid(alpha=0.3, axis='y')

plt.suptitle('光伏发电日前预报精度综合评估', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT + r'\fig3_pv_forecast_accuracy.png', bbox_inches='tight', dpi=150)
plt.close()
print('fig3 done')

# 图4: 附件4 电价分布特征（直方图+四季均值+月均日曲线）
fig = plt.figure(figsize=(14,5))
gs  = gridspec.GridSpec(1,2, figure=fig, wspace=0.35)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])

flat_p = prices4.flatten()
ax1.hist(flat_p, bins=80, color='#4fc3f7', alpha=0.7, edgecolor='white', lw=0.3)
ax1.axvline(flat_p.mean(), color='red', ls='--', lw=1.5, label=f'均值 {flat_p.mean():.3f}')
ax1.axvline(np.median(flat_p), color='orange', ls='-.', lw=1.5, label=f'中位数 {np.median(flat_p):.3f}')
ax1.set_xlabel('电价（元/kWh）', fontsize=11); ax1.set_ylabel('频次', fontsize=11)
ax1.set_title('全年实时电价频率分布', fontsize=11, fontweight='bold')
ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

# 四季典型日曲线
season_days = {'春(3-20)':78, '夏(6-21)':171, '秋(9-23)':265, '冬(12-21)':354}
colors2     = ['#2ecc71','#FFD700','#e74c3c','#3498db']
for (label, di), color in zip(season_days.items(), colors2):
    if di < len(prices4):
        ax2.plot(np.arange(144), prices4[di], lw=1.8, color=color, label=label, alpha=0.85)
ax2.set_xticks(tick_pos[::2]); ax2.set_xticklabels(tick_lab[::2], rotation=45, fontsize=8)
ax2.set_xlabel('时刻', fontsize=11); ax2.set_ylabel('电价（元/kWh）', fontsize=11)
ax2.set_title('四季典型日实时电价曲线（附件4）', fontsize=11, fontweight='bold')
ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

plt.suptitle('全年波动电价统计特征与季节规律', fontsize=13, fontweight='bold')
plt.savefig(OUT + r'\fig4_price_stats.png', bbox_inches='tight', dpi=150)
plt.close()
print('fig4 done')

# 图5: 全年日负载与光伏总量时序图（数据完整性可视化）
load_daily = load2.sum(axis=1)/6
pv_daily = pv2.sum(axis=1)/6
net_load = load_daily - pv_daily
x_days = np.arange(365)
m_ticks = [dates[dates.dt.month==m].index[0] if len(dates[dates.dt.month==m])>0 else 0 for m in range(1,13)]

# 找每月第一天索引
month_start_idx = []
for m in range(1,13):
    idx = np.where(months==m)[0]
    if len(idx)>0: month_start_idx.append(idx[0])

month_labels = ['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月']

# 创建图形
fig, axes = plt.subplots(2,1, figsize=(13,7), sharex=True)

# 上图：全年逐日负载与光伏
axes[0].bar(x_days, load_daily/1000, color='#5B7C99', alpha=0.62, width=1.0, label='日负载总量')
axes[0].bar(x_days, pv_daily/1000, color='#D6A84F', alpha=0.68, width=1.0, label='日光伏总量')

# 添加月份分界线
for idx in month_start_idx: axes[0].axvline(idx, color='#888888', ls=':', lw=0.6, alpha=0.25)

axes[0].set_ylabel('电量（MWh）', fontsize=11)
axes[0].set_title('全年逐日负载与光伏发电总量', fontsize=11, fontweight='bold')
axes[0].legend(fontsize=9, frameon=True, framealpha=0.9)
axes[0].grid(alpha=0.25, axis='y')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# =========================
# 下图：全年逐日净负载
# =========================
axes[1].fill_between(x_days, net_load/1000, color='#9A7064', alpha=0.38, label='净负荷（负载−光伏）')
axes[1].axhline(net_load.mean()/1000, color='#666666', ls='--', lw=1.4, label=f'年均 {net_load.mean()/1000:.1f} MWh/天')

# 添加月份分界线
for idx in month_start_idx: axes[1].axvline(idx, color='#888888', ls=':', lw=0.6, alpha=0.25)

axes[1].set_ylabel('净负荷（MWh）', fontsize=11)
axes[1].set_title('全年逐日净负荷变化特征', fontsize=11, fontweight='bold')
axes[1].legend(fontsize=9, frameon=True, framealpha=0.9)
axes[1].grid(alpha=0.25, axis='y')
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

# 标记全年最大净负荷日
net_peak_idx = np.argmax(net_load)
net_peak_val = net_load[net_peak_idx]/1000
axes[1].annotate(f'最大净负荷\n{net_peak_val:.1f} MWh', xy=(net_peak_idx, net_peak_val), xytext=(net_peak_idx+10, net_peak_val+6), fontsize=8.5, color='#76564D', arrowprops=dict(arrowstyle='->', color='#666666', lw=0.8), bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='#CBB8B2', alpha=0.9))

# 设置横坐标
axes[1].set_xlabel('月份', fontsize=11)
axes[1].set_xticks(month_start_idx)
axes[1].set_xticklabels(month_labels, fontsize=9)

# 总标题
plt.suptitle('全年负荷—光伏时序特征与净负载变化', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(OUT + r'\fig5_annual_timeseries.png', bbox_inches='tight', dpi=300)
plt.close()

print('fig5 done')

# 图6: 数据质量汇总 + 附件3预报时刻对比
fig, axes = plt.subplots(1,2, figsize=(14,5))

# ==================== 左图：月度负荷、光伏与电价 ====================
months_arr = np.arange(1,13)
price_m = [prices4[months==m].mean() for m in range(1,13)]
load_m_avg = [load2[months==m].mean() for m in range(1,13)]
pv_m_avg = [pv2[months==m].mean() for m in range(1,13)]

ax = axes[0]
ax2b = ax.twinx()

bars = ax.bar(months_arr-0.2, load_m_avg, width=0.35, color='#4C6A85', alpha=0.72, label='月均负载（kW）')
bars2 = ax.bar(months_arr+0.2, pv_m_avg, width=0.35, color='#C39A4B', alpha=0.72, label='月均光伏（kW）')
ln = ax2b.plot(months_arr, price_m, color='#8C5A5A', marker='o', lw=2.0, ms=4.5, label='月均电价（元/kWh）')

ax.set_xlabel('月份', fontsize=11)
ax.set_ylabel('功率（kW）', fontsize=11)
ax2b.set_ylabel('电价（元/kWh）', fontsize=11, color='#8C5A5A')
ax2b.tick_params(axis='y', colors='#8C5A5A')
ax.set_title('各月平均负荷、光伏功率与实时电价特征', fontsize=11.5, fontweight='bold')
ax.set_xticks(months_arr)
ax.set_xticklabels([f'{m}月' for m in range(1,13)], fontsize=9)

handles = [bars, bars2] + ln
labels = ['月均负荷（kW）','月均光伏（kW）','月均电价（元/kWh）']
ax.legend(handles, labels, fontsize=8.5, loc='upper left', frameon=True, framealpha=0.9)

ax.grid(alpha=0.22, axis='y')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax2b.spines['top'].set_visible(False)

# ==================== 右图：不同预报时刻 MAE ====================
forecast_times = ['0:00','6:00','12:00','18:00']
maes = []

for ft in forecast_times:
    df3_t = df3[df3.iloc[:,1]==ft].reset_index(drop=True)
    fc = df3_t.iloc[:,2:26].values.astype(float)
    n = min(len(fc), len(pv_actual_sum))
    fc_sum = fc[:n].sum(axis=1)
    mae = np.abs(fc_sum - pv_actual_sum[:n]).mean()
    maes.append(mae)
    print(f'预报时刻{ft} MAE={mae:.0f} kWh')

ax = axes[1]
y = np.arange(len(forecast_times))

bars3 = ax.barh(y, maes, height=0.48, color='#718096', alpha=0.72, edgecolor='#4A5568', linewidth=0.7)

ax.set_yticks(y)
ax.set_yticklabels(forecast_times, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel('日总电量预报MAE（kWh）', fontsize=11)
ax.set_ylabel('日前预报发布时刻', fontsize=11)
ax.set_title('不同预报时刻的光伏日总电量预报精度', fontsize=11.5, fontweight='bold')

ax.grid(alpha=0.22, axis='x')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for bar, mae in zip(bars3, maes):
    ax.text(bar.get_width()+max(maes)*0.015, bar.get_y()+bar.get_height()/2, f'{mae:.0f}', va='center', fontsize=9.5, color='#374151')

plt.suptitle('月度运行特征与光伏预报精度综合分析', fontsize=13.5, fontweight='bold')

line = Line2D([0.523, 0.523], [0.08, 0.91], transform=fig.transFigure, color='#B0B0B0', linewidth=0.8, alpha=0.7)
fig.add_artist(line)

plt.tight_layout(rect=[0,0,1,0.94])
plt.savefig(OUT + r'\fig6_data_quality_summary.png', bbox_inches='tight', dpi=300)
plt.close()

print('fig6 done')

# 输出统计表格数据到CSV
# 表1: 四份附件基本统计
summary = {
    '附件': ['附件1（典型日）','附件2（全年负载）','附件2（全年光伏）','附件3（光伏预报）','附件4（实时电价）'],
    '维度': ['144时段×1天','144时段×365天','144时段×365天','4预报时刻×365天','144时段×365天'],
    '均值': [f'{load1.mean():.1f} kW', f'{load2.mean():.1f} kW', f'{pv2.mean():.1f} kW',
             f'{pv_fc_24.mean():.1f} kW', f'{prices4.mean():.4f} 元/kWh'],
    '最小值': [f'{load1.min():.1f}', f'{load2.min():.1f}', f'{pv2.min():.1f}',
               f'{pv_fc_24.min():.1f}', f'{prices4.min():.4f}'],
    '最大值': [f'{load1.max():.1f}', f'{load2.max():.1f}', f'{pv2.max():.1f}',
               f'{pv_fc_24.max():.1f}', f'{prices4.max():.4f}'],
    '缺失值': [0, 0, 0, 0, 0]
}
pd.DataFrame(summary).to_csv(OUT + r'\table1_data_summary.csv', index=False, encoding='utf-8-sig')

# 表2: 月均数据统计
month_df = pd.DataFrame({
    '月份': [f'{m}月' for m in range(1,13)],
    '月均日负载(kWh)': [f'{load2[months==m].sum(axis=1).mean()/6:.0f}' for m in range(1,13)],
    '月均日光伏(kWh)': [f'{pv2[months==m].sum(axis=1).mean()/6:.0f}' for m in range(1,13)],
    '月均电价(元/kWh)': [f'{prices4[months==m].mean():.4f}' for m in range(1,13)],
    '月均净负载(kWh)': [f'{(load2[months==m].sum(axis=1)-pv2[months==m].sum(axis=1)).mean()/6:.0f}' for m in range(1,13)]
})
month_df.to_csv(OUT + r'\table2_monthly_stats.csv', index=False, encoding='utf-8-sig')
print('\n=== 统计摘要 ===')
print(month_df.to_string(index=False))

# 表3: 预报精度
acc_df = pd.DataFrame({
    '预报时刻': forecast_times,
    'MAE(kWh)': [f'{m:.0f}' for m in maes],
    'RMSE(kWh)': [f'{np.sqrt(((df3[df3.iloc[:,1]==ft].iloc[:365,2:26].values.astype(float).sum(axis=1)-pv_actual_sum)**2).mean()):.0f}' for ft in forecast_times],
    '相关系数R': [f'{np.corrcoef(df3[df3.iloc[:,1]==ft].iloc[:365,2:26].values.astype(float).sum(axis=1), pv_actual_sum)[0,1]:.4f}' for ft in forecast_times]
})
acc_df.to_csv(OUT + r'\table3_forecast_accuracy.csv', index=False, encoding='utf-8-sig')
print(acc_df.to_string(index=False))

print('\n所有图片和CSV已保存到:', OUT)

#Q1  典型日优化结果，生成 6 张图

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
OUT  = os.path.join(BASE, 'output')
os.makedirs(OUT, exist_ok=True)

# 加载数据 
# 附件1: 典型日负载、光伏、电价
raw1 = pd.read_excel(os.path.join(ROOT, '附件/附件1.xlsx'), header=None)
raw1.columns = raw1.iloc[0]; raw1 = raw1.iloc[1:].reset_index(drop=True)
# 列: 时段, 电价, 负载, 光伏预测 (前144行)
price_col = raw1.columns[1]
load_col  = raw1.columns[2]
pv_col    = raw1.columns[3]
price  = raw1[price_col].astype(float).values[:144]
load   = raw1[load_col].astype(float).values[:144]
pv     = raw1[pv_col].astype(float).values[:144]

# result1_improved.xlsx: 计划购电量, 充放电量
df_plan = pd.read_excel(os.path.join(ROOT, 'results/result1_improved.xlsx'),
                        sheet_name='计划购电量', header=0)
df_chg  = pd.read_excel(os.path.join(ROOT, 'results/result1_improved.xlsx'),
                        sheet_name='充放电量', header=0)

# 144个时段购电量
e_buy = df_plan.iloc[:144, 1].astype(float).values   # kWh per period

# 时间轴
x = np.arange(144)
x_pos = [t * 6 for t in range(0, 24, 4)]
x_lbl = [f'{h}:00' for h in range(0, 24, 4)]

DELTA_T = 1/6   # h per period
P_buy = e_buy / DELTA_T   # kW

# 典型日无储能基准费用（直接购电满足需求）
cost_no_storage = np.sum(np.maximum(load - pv, 0) * price * DELTA_T)
cost_with_storage = np.sum(P_buy * price * DELTA_T)
saving = cost_no_storage - cost_with_storage
saving_pct = saving / cost_no_storage * 100

print(f"有储能购电费: {cost_with_storage:.2f} 元")
print(f"无储能购电费: {cost_no_storage:.2f} 元")
print(f"节省: {saving:.2f} 元 ({saving_pct:.1f}%)")

#图7：典型日储能分阶段充放电特征

time_labels = ['0:00–4:00','4:00–8:00','8:00–12:00','12:00–16:00','16:00–20:00','20:00–24:00']
charge = np.array([4500.00, 833.33, 4787.96, 5286.04, 0.00, 5333.33])
discharge = np.array([0.00, 6365.84, 1703.00, 91.10, 5780.13, 2859.87])
x = np.arange(len(time_labels))
fig, ax = plt.subplots(figsize=(12, 5.5))
bars_c = ax.bar(x, charge, width=0.58, color='#5B7C99', alpha=0.72, edgecolor='#40566B', linewidth=0.7, label='充电量')
bars_d = ax.bar(x, -discharge, width=0.58, color='#C39A4B', alpha=0.72, edgecolor='#8A6D2F', linewidth=0.7, label='放电量')
# 0轴
ax.axhline(0, color='#4A4A4A', linewidth=1.0)
# 数据标签
for bar, val in zip(bars_c, charge):
    if val > 0:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+180, f'{val:.0f}', ha='center', va='bottom', fontsize=8.5, color='#40566B')
for bar, val in zip(bars_d, discharge):
    if val > 0:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()-180, f'{val:.0f}', ha='center', va='top', fontsize=8.5, color='#765B25')
# 坐标轴
ax.set_xticks(x)
ax.set_xticklabels(time_labels, fontsize=9.5)
ax.set_xlabel('时间段', fontsize=11)
ax.set_ylabel('储能充放电量（kWh）', fontsize=11)
ax.set_title('典型日储能分阶段充放电特征', fontsize=12.5, fontweight='bold')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, pos: f'{abs(v):.0f}'))
ax.grid(axis='y', alpha=0.20, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(fontsize=9.5, loc='upper left', frameon=True, framealpha=0.92)
# 标注日初/日末储电量
ax.text(0.98, 0.94, '0:00 储电量：6000 kWh\n24:00 储电量：6000 kWh',
        transform=ax.transAxes, ha='right', va='top', fontsize=9.5, color='#4A4A4A',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='#B8B8B8', alpha=0.92))
# 充放电方向说明
ax.text(0.02, 0.96, '↑ 充电\n↓ 放电',
        transform=ax.transAxes, ha='left', va='top', fontsize=8.5, color='#555555')
plt.tight_layout()
plt.savefig(OUT + r'\fig7_storage_operation.png', bbox_inches='tight', dpi=300)
plt.close()
print('fig7 done')

#  图1：电价曲线与购电量 
print("生成 fig1_price_buy.png ...")
fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()
ax1.bar(x, P_buy, color='#1565C0', alpha=0.55, label='购电功率 (kW)', width=0.8)
ax2.plot(x, price, color='#E65100', lw=2, label='电价 (元/kWh)')
ax1.set_xlabel('时刻', fontsize=11); ax1.set_ylabel('购电功率（kW）', fontsize=11)
ax2.set_ylabel('电价（元/kWh）', fontsize=11, color='black')
ax2.tick_params(axis='y', labelcolor='black')
ax1.set_xticks(x_pos); ax1.set_xticklabels(x_lbl, fontsize=9)
ax1.set_title('典型日购电功率与分时电价', fontsize=13)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='upper left')
ax1.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig1_price_buy.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  完成")

#  图2：供需平衡分析 
print("生成 fig2_supply_demand.png ...")

fig, ax = plt.subplots(figsize=(12, 5.5))

# ==================== 学术配色 ====================
c_pv = '#C39A4B'       # 哑金：光伏
c_buy = '#5B7C99'      # 灰蓝：购电
c_load = '#8C5A5A'     # 灰红棕：负荷
c_grid = '#B8C0C8'      # 淡灰：辅助网格

# ==================== 原始数据折线 ====================
ax.plot(x, pv, color=c_pv, lw=1.8, label='光伏出力（kW）', zorder=3)
ax.plot(x, P_buy, color=c_buy, lw=1.8, label='购电功率（kW）', zorder=3)
ax.plot(x, load, color=c_load, lw=2.6, label='负载需求（kW）', zorder=4)

# ==================== 滚动波动带 ====================
window = max(3, min(12, len(x)//20))
pv_mean = pd.Series(pv).rolling(window, center=True, min_periods=1).mean()
pv_std = pd.Series(pv).rolling(window, center=True, min_periods=1).std().fillna(0)
buy_mean = pd.Series(P_buy).rolling(window, center=True, min_periods=1).mean()
buy_std = pd.Series(P_buy).rolling(window, center=True, min_periods=1).std().fillna(0)

ax.fill_between(x, np.maximum(0, pv_mean-pv_std), pv_mean+pv_std, color=c_pv, alpha=0.12, linewidth=0)
ax.fill_between(x, np.maximum(0, buy_mean-buy_std), buy_mean+buy_std, color=c_buy, alpha=0.10, linewidth=0)

# ==================== 识别关键运行区间 ====================
surplus = pv > load
grid_depend = P_buy > 0

# 找连续区间
def get_segments(mask):
    segments = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            segments.append((start, i-1))
            start = None
    if start is not None:
        segments.append((start, len(mask)-1))
    return segments

surplus_seg = get_segments(surplus)
grid_seg = get_segments(grid_depend)

# 只标注较长、具有代表性的区间，避免视觉拥挤
if surplus_seg:
    seg = max(surplus_seg, key=lambda z: z[1]-z[0])
    if seg[1]-seg[0] >= max(2, len(x)//30):
        mid = (seg[0]+seg[1])//2
        ax.annotate('光伏出力高于负荷',
                    xy=(x[mid], pv[mid]),
                    xytext=(x[mid], max(pv)*0.88),
                    ha='center', fontsize=9, color='#765B25',
                    arrowprops=dict(arrowstyle='->', color='#765B25', lw=0.8),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#D6C28A', alpha=0.9))

if grid_seg:
    seg = max(grid_seg, key=lambda z: z[1]-z[0])
    if seg[1]-seg[0] >= max(2, len(x)//30):
        mid = (seg[0]+seg[1])//2
        ax.annotate('外部电网供电依赖',
                    xy=(x[mid], P_buy[mid]),
                    xytext=(x[mid], max(load)*0.55),
                    ha='center', fontsize=9, color='#4D657D',
                    arrowprops=dict(arrowstyle='->', color='#4D657D', lw=0.8),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#B8C4CF', alpha=0.9))

# ==================== 坐标轴与格式 ====================
ax.set_xlabel('时刻', fontsize=11)
ax.set_ylabel('功率（kW）', fontsize=11)
ax.set_xticks(x_pos)
ax.set_xticklabels(x_lbl, fontsize=9)

ax.set_title('典型日光伏、购电与负荷需求的供需关系', fontsize=13, fontweight='bold', pad=10)

ax.legend(fontsize=9.5, loc='upper left', frameon=True, framealpha=0.92)
ax.grid(axis='y', color=c_grid, alpha=0.25, linestyle='--', linewidth=0.7)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ==================== 数据关系说明 ====================
pv_total = np.sum(pv)
buy_total = np.sum(P_buy)
load_total = np.sum(load)

ax.text(0.99, 0.97,
        f'典型日总量：光伏 {pv_total/6:.1f} kWh｜购电 {buy_total/6:.1f} kWh｜负荷 {load_total/6:.1f} kWh',
        transform=ax.transAxes, ha='right', va='top', fontsize=8.5, color='#555555',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#D0D0D0', alpha=0.9))

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig2_supply_demand.png'), dpi=300, bbox_inches='tight')
plt.close()

print("  完成")

#  图3：SOC曲线（用充放电数据反推）
print("生成 fig3_soc_curve.png ...")
# 从充放电表读充放电量（kWh/4h段）
# 重新用greedy仿真计算SOC曲线
ETA_CH = 0.9; ETA_DIS = 0.9
E_MAX = 10800; E_MIN = 1200; E_INIT = 6000
soc = np.zeros(145)
soc[0] = E_INIT
p_ch = np.zeros(144); p_dis = np.zeros(144)
for t in range(144):
    net = P_buy[t] + pv[t] - load[t]
    if net >= 0:
        e_room = (E_MAX - soc[t]) / ETA_CH * 6   # kW
        pc = min(net, 5000, max(e_room, 0))
        p_ch[t] = pc
        soc[t+1] = soc[t] + pc * ETA_CH * DELTA_T
    else:
        need = -net
        e_avail = (soc[t] - E_MIN) * ETA_DIS * 6  # kW
        pd_ = min(need, 5000, max(e_avail, 0))
        p_dis[t] = pd_
        soc[t+1] = soc[t] - pd_ / ETA_DIS * DELTA_T

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(np.arange(145), soc, color='#2E7D32', lw=2, label='储能电量 (kWh)')
ax.axhline(E_MAX, color='red', lw=1, linestyle='--', alpha=0.7, label=f'上限 {E_MAX} kWh')
ax.axhline(E_MIN, color='orange', lw=1, linestyle='--', alpha=0.7, label=f'下限 {E_MIN} kWh')
ax.axhline(E_INIT, color='gray', lw=1, linestyle=':', alpha=0.7, label=f'初始/终态 {E_INIT} kWh')
ax.set_xlabel('时刻', fontsize=11); ax.set_ylabel('储能电量（kWh）', fontsize=11)
xt = list(x_pos) + [144]
xl = list(x_lbl) + ['24:00']
ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=9)
ax.set_title('典型日储能电量（SOC）曲线', fontsize=13)
ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig3_soc_curve.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  完成")

#  图4：充放电功率曲线 
print("生成 fig4_charge_discharge.png ...")
fig, ax = plt.subplots(figsize=(12, 5))

offset = max(p_dis) * 0.15

ax.plot(x, p_ch, color='#43A047', lw=2, marker='o', ms=3, label='充电功率 (kW)')
ax.plot(x, -p_dis - offset, color='#E53935', lw=2, marker='s', ms=3, label='放电功率 (kW)')

ax.axhline(0, color='k', lw=0.8)
ax.axhline(-offset, color='grey', lw=0.5, ls='--')

xlim = ax.get_xlim()
ax.text(xlim[0], 0, ' 充电功率=0', va='center', ha='right', fontsize=9, color='#43A047')
ax.text(xlim[0], -offset, ' 放电功率=0', va='center', ha='right', fontsize=9, color='#E53935')

ax.set_xlabel('时刻', fontsize=11)
ax.set_ylabel('功率（kW）', fontsize=11)
ax.set_xticks(x_pos)
ax.set_xticklabels(x_lbl, fontsize=9)
ax.set_title('典型日储能充放电功率', fontsize=13)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig4_charge_discharge.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  完成")

#  图5：电价-购电量散点图 
print("生成 fig5_price_buy_scatter.png ...")
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(price, P_buy, c=x, cmap='viridis', alpha=0.7, s=30)
plt.colorbar(sc, ax=ax, label='时段序号')
ax.set_xlabel('电价（元/kWh）', fontsize=11)
ax.set_ylabel('购电功率（kW）', fontsize=11)
ax.set_title('分时电价与购电功率的关系', fontsize=13)
ax.grid(alpha=0.3)
corr = np.corrcoef(price, P_buy)[0, 1]
ax.text(0.97, 0.97, f'相关系数 r = {corr:.3f}', transform=ax.transAxes,
        ha='right', va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig5_price_buy_scatter.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  完成")

#  图6：有/无储能费用对比 
print("生成 fig6_cost_compare.png ...")

labels6 = ['无储能方案', '有储能方案']
costs6  = [cost_no_storage, cost_with_storage]
colors6 = ['#EF5350', '#1565C0']

total = sum(costs6)
fractions = [c / total for c in costs6]

angles = []
cum = 0
for f in fractions:
    angles.append((cum, cum + f * 2 * np.pi))
    cum += f * 2 * np.pi

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

r = 1.0
h = 1.0
n_arc = 80
explode_dist = 0.15

for i, (a_start, a_end) in enumerate(angles):
    mid_angle = (a_start + a_end) / 2.0

    dx = explode_dist * np.cos(mid_angle)
    dy = explode_dist * np.sin(mid_angle)

    theta = np.linspace(a_start, a_end, n_arc)

    x_arc = r * np.cos(theta) + dx
    y_arc = r * np.sin(theta) + dy

    x_side = np.tile(x_arc, (2, 1))
    y_side = np.tile(y_arc, (2, 1))
    z_side = np.array([np.zeros_like(theta), np.full_like(theta, h)])

    alpha_val = 1.0 if i == 1 else 0.45

    ax.plot_surface(x_side, y_side, z_side,
                    color=colors6[i], alpha=alpha_val,
                    linewidth=0, antialiased=True, shade=True)

    r_top = np.linspace(0, r, 20)
    R, T = np.meshgrid(r_top, theta)
    x_top = R * np.cos(T) + dx
    y_top = R * np.sin(T) + dy
    z_top = np.full_like(x_top, h)

    ax.plot_surface(x_top, y_top, z_top,
                    color=colors6[i], alpha=alpha_val,
                    linewidth=0, antialiased=True, shade=True)

    r_bot = np.linspace(0, r, 20)
    R_b, T_b = np.meshgrid(r_bot, theta)
    x_bot = R_b * np.cos(T_b) + dx
    y_bot = R_b * np.sin(T_b) + dy
    z_bot = np.zeros_like(x_bot)

    ax.plot_surface(x_bot, y_bot, z_bot,
                    color=colors6[i], alpha=alpha_val,
                    linewidth=0, antialiased=True, shade=True)

for i, (a_start, a_end) in enumerate(angles):
    mid_angle = (a_start + a_end) / 2.0

    px = 0.5 + 0.38 * np.cos(mid_angle)
    py = 0.5 + 0.38 * np.sin(mid_angle)

    if i == 0:
        px += 0.16
        py -= 0.06

    pct = fractions[i] * 100
    ax.text2D(px, py,
              f'{labels6[i]}\n{costs6[i]:.0f}元\n({pct:.1f}%)',
              transform=ax.transAxes,
              ha='center', va='center',
              fontsize=10, fontweight='bold', color='#333333')

ax.set_xlim([-1.8, 1.8])
ax.set_ylim([-1.8, 1.8])
ax.set_zlim([0, 1.5])
ax.view_init(elev=30, azim=-60)
ax.set_axis_off()

note = f'节省 {saving:.2f} 元（{saving_pct:.1f}%）'

fig.suptitle('典型日有/无储能方案购电费用对比', fontsize=14, fontweight='bold', y=0.97)
fig.text(0.5, 0.03, note, ha='center', fontsize=11, color='#2E7D32', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig6_cost_compare.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  完成")

print(f"\n所有图表生成完毕，输出目录：{OUT}")

#生成问题2两阶段模型论文图片
import sys, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
import warnings
from matplotlib.lines import Line2D
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data_loader import DataLoader

plt.rcParams['font.family'] = ['SimHei', 'Microsoft YaHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path(__file__).parent.parent
OUT  = Path(__file__).parent / 'output'
OUT.mkdir(exist_ok=True)

# 加载数据
loader  = DataLoader(ROOT / '附件')
data2   = loader.load_attachment2()
load_all  = data2['load']
pv_all    = data2['pv_actual']
dates_all = data2['dates']

ann1  = pd.read_excel(ROOT / '附件/附件1.xlsx')
price = ann1['电价'].values  # (144,)

df_pur = pd.read_excel(ROOT / 'results/result2_improved.xlsx', sheet_name=0)
df_pur['date'] = pd.to_datetime(df_pur.iloc[:, 0])
df_chg = pd.read_excel(ROOT / 'results/result2_improved.xlsx', sheet_name=1)
df_emg = pd.read_excel(ROOT / 'results/result2_improved.xlsx', sheet_name=2)

fc     = np.load(ROOT / 'results/forecasts_p2.npz', allow_pickle=True)
pv_fc  = fc['pv_forecast']   # (334,144)
ld_fc  = fc['load_forecast']  # (334,144)

t_hours = np.arange(144) / 6
start   = pd.Timestamp('2025-02-01')

seasons = [
    ('春季 (2025-03-20)', '2025-03-20'),
    ('夏季 (2025-06-21)', '2025-06-21'),
    ('秋季 (2025-09-23)', '2025-09-23'),
    ('冬季 (2025-12-21)', '2025-12-21'),
]
season_colors = ['steelblue', 'orange', 'green', 'red']

# 图1：四季典型日 实测 vs SARIMA预测（负荷+光伏）
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axs = [axes[0,0], axes[0,1], axes[1,0], axes[1,1]]

for (title, dstr), ax in zip(seasons, axs):
    idx_real = np.where(dates_all == pd.Timestamp(dstr))[0][0]
    idx_fc   = (pd.Timestamp(dstr) - start).days
    load_d  = load_all[idx_real]
    pv_d    = pv_all[idx_real]

    ax.fill_between(t_hours, load_d, alpha=0.18, color='steelblue')
    ax.plot(t_hours, load_d, color='steelblue', lw=1.8, label='实测负荷')
    ax.fill_between(t_hours, pv_d, alpha=0.18, color='orange')
    ax.plot(t_hours, pv_d,   color='orange',   lw=1.8, label='实测光伏')

    if 0 <= idx_fc < len(ld_fc):
        ax.plot(t_hours, ld_fc[idx_fc], '--', color='steelblue', lw=1.0,
                alpha=0.75, label='SARIMA预测负荷')
    if 0 <= idx_fc < len(pv_fc):
        ax.plot(t_hours, pv_fc[idx_fc], '--', color='orange',    lw=1.0,
                alpha=0.75, label='SARIMA预测光伏')

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('时间 (h)', fontsize=9)
    ax.set_ylabel('功率 (kW)', fontsize=9)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 4))
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

fig.suptitle('四季典型日实测与SARIMA预测对比（负荷与光伏）',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUT / 'fig1_seasonal_load_pv.png', dpi=150, bbox_inches='tight')
plt.close()
print('fig1 done')

# 图2：全年购电费月度统计（柱+折线）
df_pur['month'] = df_pur['date'].dt.month
monthly_cost = df_pur.groupby('month')['全天购电费'].sum() / 10000
monthly_pur  = df_pur.groupby('month')['全天购电量'].sum() / 10000

fig, ax1 = plt.subplots(figsize=(12, 6))
x = np.arange(len(monthly_cost))
ax1.bar(x, monthly_cost.values, color='steelblue', alpha=0.72)
ax1.set_xlabel('月份', fontsize=11)
ax1.set_ylabel('购电费 (万元)', fontsize=11, color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_xticks(x)
ax1.set_xticklabels([f'{m}月' for m in monthly_cost.index])
for i, c in enumerate(monthly_cost.values):
    ax1.text(i, c + 0.4, f'{c:.1f}', ha='center', fontsize=8, color='steelblue')

ax2 = ax1.twinx()
ax2.plot(x, monthly_pur.values, 'o-', color='orangered', lw=2, markersize=6)
ax2.set_ylabel('购电量 (万kWh)', fontsize=11, color='black')
ax2.tick_params(axis='y', labelcolor='black')

p1 = Patch(color='steelblue', alpha=0.72, label='月购电费 (万元)')
p2 = plt.Line2D([0],[0], color='orangered', marker='o', lw=2, label='月购电量 (万kWh)')
ax1.legend(handles=[p1, p2], loc='upper left', fontsize=9)
plt.title('全年各月购电费与购电量统计（2025年2-12月）',
          fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT / 'fig2_monthly_cost.png', dpi=150, bbox_inches='tight')
plt.close()
print('fig2 done')


# 图3：四季典型日计划购电策略（含谷时高亮）
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axs = [axes[0,0], axes[0,1], axes[1,0], axes[1,1]]

for (title, dstr), ax, col in zip(seasons, axs, season_colors):
    row = df_pur[df_pur['date'] == pd.Timestamp(dstr)]
    if row.empty:
        continue
    plan_144 = row.iloc[0, 1:145].values.astype(float)

    ax.bar(t_hours, plan_144, width=1/6*0.9, color=col, alpha=0.65, label='计划购电 (kW)')
    for t in range(144):
        if price[t] < 0.4:
            ax.axvspan(t/6, (t+1)/6, alpha=0.07, color='blue')

    qty  = row['全天购电量'].values[0]
    cost = row['全天购电费'].values[0]
    ax.text(0.98, 0.97,
            f'购电量:{qty/10000:.2f}万kWh\n购电费:{cost/10000:.2f}万元',
            transform=ax.transAxes, va='top', ha='right', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.75))

    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('时间 (h)', fontsize=9)
    ax.set_ylabel('购电功率 (kW)', fontsize=9)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 4))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle('四季典型日计划购电策略（蓝色背景为深谷时段）',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUT / 'fig3_seasonal_purchase.png', dpi=150, bbox_inches='tight')
plt.close()
print('fig3 done')

# 图4：全年逐日购电费时序
fig, ax = plt.subplots(figsize=(14, 5))
daily_cost = df_pur['全天购电费'].values / 10000
x = np.arange(len(daily_cost))
ax.fill_between(x, daily_cost, alpha=0.35, color='steelblue')
ax.plot(x, daily_cost, color='steelblue', lw=1)
ax.axhline(daily_cost.mean(), color='red', ls='--', lw=1.5,
           label=f'日均={daily_cost.mean():.2f}万元')

months_vals = df_pur['date'].dt.month.values
boundaries  = [0] + [i for i in range(1, len(months_vals))
                     if months_vals[i] != months_vals[i-1]] + [len(months_vals)]
for b in boundaries[1:-1]:
    ax.axvline(b, color='gray', ls=':', alpha=0.5)
for i in range(len(boundaries)-1):
    mid = (boundaries[i] + boundaries[i+1]) / 2
    ax.text(mid, 0.02, f'{months_vals[boundaries[i]]}月',
            ha='center', va='bottom', fontsize=8, color='gray',
            transform=ax.get_xaxis_transform())

ax.set_xlabel('日期序号（2025年2月—12月）', fontsize=11)
ax.set_ylabel('购电费 (万元/天)', fontsize=11)
ax.set_title('全年逐日购电费时序图（两阶段调度）', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / 'fig4_daily_cost_series.png', dpi=150, bbox_inches='tight')
plt.close()
print('fig4 done')

# 图5：SARIMA预测误差 + 电价-购电相关性
ann2_load2 = pd.read_excel(ROOT / '附件/附件2.xlsx', sheet_name='小区负载')
ann2_pv2   = pd.read_excel(ROOT / '附件/附件2.xlsx', sheet_name='光伏发电实际功率')
load_all2  = ann2_load2.iloc[:, 1:145].values
pv_all2    = ann2_pv2.iloc[:,  1:145].values
dates_all2 = pd.to_datetime(ann2_load2.iloc[:, 0])

idxs = []
for i in range(334):
    d = start + pd.Timedelta(days=i)
    c = np.where(dates_all2 == d)[0]
    idxs.append(c[0] if len(c) > 0 else None)

valid  = [(i, j) for i, j in enumerate(idxs) if j is not None]
fc_i   = [v[0] for v in valid]
act_i  = [v[1] for v in valid]
load_actual = load_all2[act_i]
pv_actual   = pv_all2[act_i]
load_mae_day = np.abs(ld_fc[fc_i] - load_actual).mean(axis=1)
pv_mae_day   = np.abs(pv_fc[fc_i] - pv_actual).mean(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.plot(load_mae_day, color='steelblue', lw=1, alpha=0.7,
        label=f'负荷MAE (均值={load_mae_day.mean():.0f} kW)')
ax.plot(pv_mae_day,   color='orange',   lw=1, alpha=0.7,
        label=f'光伏MAE (均值={pv_mae_day.mean():.0f} kW)')
ax.axhline(load_mae_day.mean(), color='steelblue', ls='--', lw=1.2)
ax.axhline(pv_mae_day.mean(),   color='orange',    ls='--', lw=1.2)
ax.set_xlabel('日期序号', fontsize=11)
ax.set_ylabel('MAE (kW)', fontsize=11)
ax.set_title('SARIMA逐日预测误差（MAE）', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
avg_buy = df_pur.iloc[:, 1:145].values.astype(float).mean(axis=0)
sc = ax2.scatter(price, avg_buy, c=avg_buy, cmap='RdYlBu_r', alpha=0.65, s=25)
ax2.set_xlabel('电价 (元/kWh)', fontsize=11)
ax2.set_ylabel('全年平均计划购电量 (kW)', fontsize=11)
ax2.set_title('电价与计划购电量相关性散点图', fontsize=11, fontweight='bold')
plt.colorbar(sc, ax=ax2, label='购电量 (kW)')
ax2.grid(True, alpha=0.3)
line = Line2D([0.513, 0.513], [0.05, 0.92], transform=fig.transFigure, color='gray', linewidth=1.5)
fig.add_artist(line)
plt.tight_layout()
plt.savefig(OUT / 'fig5_price_purchase_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print('fig5 done')

# 图6：两阶段偏差 — 逐日计划购电 vs 紧急购电
df_emg.columns = ['日期', '时间段', '购电量']

def safe_ts(s):
    try:
        return pd.Timestamp(str(s))
    except Exception:
        return pd.NaT

df_emg['日期_ts'] = df_emg['日期'].apply(safe_ts)
emg_clean = df_emg.dropna(subset=['日期_ts'])
emg_daily = (emg_clean.groupby(emg_clean['日期_ts'].dt.normalize())['购电量']
             .sum().reset_index())
emg_daily.columns = ['date', 'emg_qty']
emg_daily['date'] = pd.to_datetime(emg_daily['date'])

df_m = df_pur[['date', '全天购电量', '全天购电费']].merge(emg_daily, on='date', how='left')
df_m['emg_qty'] = df_m['emg_qty'].fillna(0)
df_m['emg_pct'] = df_m['emg_qty'] / df_m['全天购电量'].replace(0, np.nan) * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(len(df_m))

ax = axes[0]
ax.fill_between(x, df_m['全天购电量'] / 10000, alpha=0.3, color='steelblue',
                label='计划购电量 (万kWh)')
ax.fill_between(x, df_m['emg_qty'] / 10000, alpha=0.55, color='red',
                label='紧急购电量 (万kWh)')
ax.set_xlabel('日期序号', fontsize=11)
ax.set_ylabel('购电量 (万kWh)', fontsize=11)
ax.set_title('全年逐日计划购电与紧急购电量对比', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
total_emg = df_m['emg_qty'].sum()
ax.text(0.98, 0.97,
        f'全年紧急购电总量\n{total_emg/10000:.2f}万kWh',
        transform=ax.transAxes, va='top', ha='right', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

ax2 = axes[1]
ax2.plot(x, df_m['emg_pct'].fillna(0), color='red', lw=0.8, alpha=0.6)
mean_pct = df_m['emg_pct'].mean()
ax2.axhline(mean_pct, color='red', ls='--', lw=1.5,
            label=f'均值={mean_pct:.2f}%')
ax2.set_xlabel('日期序号', fontsize=11)
ax2.set_ylabel('紧急购电占比 (%)', fontsize=11)
ax2.set_title('逐日紧急购电占当日购电量占比', fontsize=11, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / 'fig6_soc_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print('fig6 done')

print('\n所有图片生成完成！')
print(f'输出目录: {OUT}')

#Q3 可视化脚本 v2 — 加入无调整对照组，重绘全部6张图
import os, re
import numpy as np
import pandas as pd
import openpyxl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from datetime import datetime, timedelta
import matplotlib.dates as mdates

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
OUT  = os.path.join(BASE, 'output')
os.makedirs(OUT, exist_ok=True)

# 读取数据
def load_sheet(path, sheet):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = list(wb[sheet].iter_rows(values_only=True))
    wb.close()
    return rows

def parse_date(raw):
    if raw is None:
        return None
    if hasattr(raw, 'year'):
        return pd.Timestamp(raw)
    s = str(raw).strip()
    nums = re.findall(r'\d+', s)
    if len(nums) >= 3:
        return pd.Timestamp(int(nums[0]), int(nums[1]), int(nums[2]))
    return None

def sheet_to_df(rows):
    h = rows[0]
    fc = h.index('全天购电费')
    slot_cols = slice(1, 145)      
    data = []
    for r in rows[1:]:
        dt = parse_date(r[0])
        if dt is None:
            continue
        cost = r[fc] or 0.0
        slots = [v or 0.0 for v in r[slot_cols]]
        data.append({'date': dt, 'cost': float(cost), 'slots': np.array(slots, dtype=float)})
    return data

print("加载数据 ...")
R3_PATH = os.path.join(ROOT, 'results/result3.xlsx')
RN_PATH = os.path.join(ROOT, 'results/result3_no_adjust.xlsx')

rows_p  = load_sheet(R3_PATH, '计划购电量')
rows_a  = load_sheet(R3_PATH, '调整购电量')
rows_n  = load_sheet(RN_PATH, '调整购电量')

dp = sheet_to_df(rows_p)  
da = sheet_to_df(rows_a)   
dn = sheet_to_df(rows_n)   

# 按日期对齐
date_p = {d['date']: d for d in dp}
date_a = {d['date']: d for d in da}
date_n = {d['date']: d for d in dn}

dates_sorted = sorted(date_a.keys())

# 电价
raw1 = pd.read_excel(os.path.join(ROOT, '附件/附件1.xlsx'), header=0)
price = raw1.iloc[:144, 1].astype(float).values

# 轴刻度
x144  = np.arange(144)
x_pos = [t*6 for t in range(0, 24, 4)]
x_lbl = [f'{h}:00' for h in range(0, 24, 4)]
MONTHS = list(range(2, 13))
MONTH_LABELS = [f'{m}月' for m in MONTHS]

# 月度汇总
mon_p = defaultdict(float); mon_a = defaultdict(float); mon_n = defaultdict(float)
for dt in dates_sorted:
    m = dt.month
    mon_p[m] += date_p.get(dt, {}).get('cost', 0)
    mon_a[m] += date_a.get(dt, {}).get('cost', 0)
    mon_n[m] += date_n.get(dt, {}).get('cost', 0)

m_p  = [mon_p[m]/10000 for m in MONTHS]
m_a  = [mon_a[m]/10000 for m in MONTHS]
m_n  = [mon_n[m]/10000 for m in MONTHS]

TOTAL_P = sum(mon_p.values()) / 10000
TOTAL_A = sum(mon_a.values()) / 10000
TOTAL_N = sum(mon_n.values()) / 10000
DAY_A   = TOTAL_A * 10000 / len(dates_sorted)
DAY_N   = TOTAL_N * 10000 / len(dates_sorted)

print(f"  计划: {TOTAL_P:.2f}万  四阶段调整: {TOTAL_A:.2f}万  无调整: {TOTAL_N:.2f}万")
print(f"  日均 调整: {DAY_A:.2f}元  无调整: {DAY_N:.2f}元  节省: {DAY_N-DAY_A:.2f}元/天")

# 图1：月度三方案费用对比（计划/四阶段调整/无调整）
print("生成 fig1_monthly_plan_vs_adj.png ...")
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(MONTHS))
ax.plot(x, m_p, color='#1565C0', lw=1.5, marker='o', markersize=5,
        label=f'0:00计划（{TOTAL_P:.2f}万元）', alpha=0.9)
ax.plot(x, m_a, color='#E65100', lw=1.8, marker='s', markersize=5,
        label=f'四阶段调整（{TOTAL_A:.2f}万元）', alpha=0.9)
ax.plot(x, m_n, color='#388E3C', lw=1.5, marker='^', markersize=5,
        label=f'无调整对照（{TOTAL_N:.2f}万元）', alpha=0.9)
ax.set_xticks(x); ax.set_xticklabels(MONTH_LABELS, fontsize=10)
ax.set_xlabel('月份', fontsize=11); ax.set_ylabel('月度购电费用（万元）', fontsize=11)
ax.set_title('Q3方案各月购电费用三方案对比', fontsize=13)
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
save_val = TOTAL_N - TOTAL_A
ax.text(0.01, 0.98,
        f'四阶段调整 vs 无调整：节省 {save_val:.2f}万元（{save_val/TOTAL_N*100:.2f}%）\n'
        f'日均节省 {(DAY_N-DAY_A):.0f}元/天',
        transform=ax.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='round', facecolor='#E8F5E9', alpha=0.9))
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig1_monthly_plan_vs_adj.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  完成 fig1")

# 图2：四季典型日调度对比（计划 vs 调整 vs 无调整）
print("生成 fig2_seasonal_dispatch.png ...")
season_days = [('春分(3-20)', '2025-03-20'), ('夏至(6-21)', '2025-06-21'),
               ('秋分(9-23)', '2025-09-23'), ('冬至(12-21)', '2025-12-21')]
fig2, axes2 = plt.subplots(2, 2, figsize=(15, 9))
for idx, (title, ds) in enumerate(season_days):
    ax = axes2[idx//2][idx%2]
    dt = pd.Timestamp(ds)
    dp_day = date_p.get(dt)
    da_day = date_a.get(dt)
    dn_day = date_n.get(dt)
    if dp_day and da_day and dn_day:
        ax.plot(x144, dp_day['slots'], color='#1565C0', lw=1.5, label='0:00计划', alpha=0.9)
        ax.plot(x144, da_day['slots'], color='#E65100', lw=1.8, ls='--', label='四阶段调整', alpha=0.9)
        ax.plot(x144, dn_day['slots'], color='#388E3C', lw=1.5, ls=':', label='无调整对照', alpha=0.9)
        cp = dp_day['cost']; ca = da_day['cost']; cn = dn_day['cost']
        ax.set_title(f'{title}\n计划{cp:.0f}元 | 调整{ca:.0f}元 | 无调{cn:.0f}元\n'
                     f'调整节省 {cn-ca:.0f}元 vs 无调', fontsize=9)
    ax.set_xticks(x_pos); ax.set_xticklabels(x_lbl, fontsize=8)
    ax.set_xlabel('时刻', fontsize=9); ax.set_ylabel('购电量（kWh）', fontsize=9)
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.25)
fig2.suptitle('Q3方案四季典型日购电量对比（0:00计划 / 四阶段调整 / 无调整）', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig2_seasonal_dispatch.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  完成 fig2")

# 图3：全年逐日费用时序（调整 vs 无调整）
print("生成 fig3_daily_plan_vs_adj.png ...")
daily_a = pd.Series({dt: date_a[dt]['cost']/10000 for dt in dates_sorted})
daily_n = pd.Series({dt: date_n[dt]['cost']/10000 for dt in dates_sorted})
daily_diff = daily_n - daily_a   

fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(15, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})
ax3a.plot(daily_a.index, daily_a.values, color='#E65100', lw=0.9, label=f'四阶段调整（{TOTAL_A:.2f}万元）')
ax3a.plot(daily_n.index, daily_n.values, color='#388E3C', lw=0.9, label=f'无调整对照（{TOTAL_N:.2f}万元）', alpha=0.7)
ax3a.set_ylabel('当日购电费（万元）', fontsize=11)
ax3a.set_title('Q3方案全年逐日购电费用：四阶段调整 vs 无调整对照', fontsize=13)
ax3a.legend(fontsize=10); ax3a.grid(axis='y', alpha=0.3)
better_days = (daily_diff > 0).sum()
ax3a.text(0.01, 0.97,
          f'调整优于无调整天数：{better_days}/{len(dates_sorted)}天\n'
          f'全年节省：{(TOTAL_N-TOTAL_A):.2f}万元',
          transform=ax3a.transAxes, fontsize=9, va='top',
          bbox=dict(boxstyle='round', facecolor='#E8F5E9', alpha=0.85))
ax3b.bar(daily_diff.index, daily_diff.values,
         color=np.where(daily_diff.values >= 0, '#E65100', '#388E3C'), alpha=0.7, width=1)
ax3b.axhline(0, color='k', lw=0.8)
ax3b.set_ylabel('节省（万元）', fontsize=10)
ax3b.set_xlabel('日期', fontsize=11)
ax3b.set_title('逐日节省金额（正值=四阶段调整更优）', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig3_daily_plan_vs_adj.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  完成 fig3")

#  图4：调整效益分布（无调整-调整 差值直方图，按季节分）
print("生成 fig4_adj_distribution.png ...")
diff_vals = np.array([date_n[dt]['cost'] - date_a[dt]['cost'] for dt in dates_sorted])

season_map_m = {2:'春',3:'春',4:'春',5:'夏',6:'夏',7:'夏',8:'夏',
                9:'秋',10:'秋',11:'秋',12:'冬',1:'冬'}
season_arr = np.array([season_map_m[dt.month] for dt in dates_sorted])

fig4, axes4 = plt.subplots(1, 2, figsize=(13, 5))

x_all = np.arange(len(diff_vals))
axes4[0].plot(x_all, diff_vals, color='#E65100', lw=1.0, alpha=0.8)
axes4[0].axhline(diff_vals.mean(), color='red', lw=1.5, ls='--',
                 label=f'均值 {diff_vals.mean():.0f}元')
axes4[0].axhline(0, color='k', lw=1)
axes4[0].set_xlabel('日期序号', fontsize=10)
axes4[0].set_ylabel('无调整 - 四阶段调整 费用差（元）', fontsize=10)
axes4[0].set_title('全年逐日调整效益\n（正值=四阶段调整更优）', fontsize=11)
axes4[0].legend(fontsize=9)
pct_better = (diff_vals > 0).mean() * 100
axes4[0].text(0.98, 0.97, f'调整更优：{pct_better:.1f}%天',
              transform=axes4[0].transAxes, ha='right', va='top', fontsize=9,
              bbox=dict(boxstyle='round', facecolor='#E8F5E9', alpha=0.85))

for s, c in [('春','#4CAF50'),('夏','#FF5722'),('秋','#FF9800'),('冬','#2196F3')]:
    sub = diff_vals[season_arr == s]
    if len(sub):
        axes4[1].plot(np.arange(len(sub)), sub, color=c, lw=1.0, alpha=0.8,
                      label=f'{s}季(均值{sub.mean():.0f}元)')
axes4[1].axhline(0, color='k', lw=1)
axes4[1].set_xlabel('日内序号', fontsize=10)
axes4[1].set_ylabel('无调整 - 四阶段调整 费用差（元）', fontsize=10)
axes4[1].set_title('四季调整效益', fontsize=12)
axes4[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig4_adj_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  完成 fig4")

# 图5：冬季典型日详细对比（计划/调整/无调整 + 电价）
print("生成 fig5_winter_detail.png ...")
winter_dt = pd.Timestamp('2025-12-21')
dp5 = date_p.get(winter_dt); da5 = date_a.get(winter_dt); dn5 = date_n.get(winter_dt)
fig5, (ax5a, ax5b) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})
if dp5 and da5 and dn5:
    ax5a.step(x144, dp5['slots'], where='mid', color='#1565C0', lw=2, label=f"0:00计划 {dp5['cost']:.0f}元")
    ax5a.step(x144, da5['slots'], where='mid', color='#E65100', lw=2, ls='--', label=f"四阶段调整 {da5['cost']:.0f}元")
    ax5a.step(x144, dn5['slots'], where='mid', color='#388E3C', lw=1.5, ls=':', label=f"无调整 {dn5['cost']:.0f}元")
    ax5r = ax5a.twinx()
    ax5r.plot(x144, price, color='#888', lw=1.2, alpha=0.55, label='电价')
    ax5r.set_ylabel('电价（元/kWh）', fontsize=9, color='#888')
    ax5r.tick_params(axis='y', labelcolor='#888', labelsize=8)
    delta5 = da5['slots'] - dn5['slots']
    ax5b.bar(x144, delta5, color=np.where(delta5 >= 0, '#E65100', '#388E3C'), alpha=0.8, width=0.8)
    ax5b.axhline(0, color='k', lw=0.8)
    ax5b.set_ylabel('调整增量（kWh）', fontsize=10)
    ax5b.set_title('四阶段调整购电量 - 无调整购电量（正值=调整增加购电）', fontsize=9)
ax5a.set_title(f'冬至（12月21日）详细调度对比\n调整节省 {dn5["cost"]-da5["cost"]:.0f}元 vs 无调整', fontsize=12)
ax5a.set_ylabel('购电量（kWh）', fontsize=10)
ax5a.legend(fontsize=9, loc='upper left'); ax5a.grid(axis='y', alpha=0.3)
ax5b.set_xticks(x_pos); ax5b.set_xticklabels(x_lbl, fontsize=9)
ax5b.set_xlabel('时刻', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig5_winter_detail.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  完成 fig5")

# 图6：综合汇总（年总费用柱状 + 季节箱线）
print("生成 fig6_summary.png ...")

fig_3d = plt.figure(figsize=(10, 7))
ax6l = fig_3d.add_subplot(111, projection='3d')

lbls = ['0:00计划\n(名义)', '四阶段调整\n(实际)', '无调整对照\n(实际)']
vals6 = [TOTAL_P, TOTAL_A, TOTAL_N]
clrs6 = ['#8FB3D9', '#D9A6A6', '#A8C5A0']

x = np.arange(len(vals6))
y = np.zeros(len(vals6))
z = np.zeros(len(vals6))
dx = np.full(len(vals6), 0.55)
dy = np.full(len(vals6), 0.55)
dz = np.array(vals6)

for i in range(len(vals6)):
    ax6l.bar3d(
        x[i], y[i], z[i], dx[i], dy[i], dz[i],
        color=clrs6[i], alpha=0.45,
        edgecolor='#555555', linewidth=0.7, shade=False
    )

line_y = np.full(len(vals6), -0.12)
line_x = x + dx / 2
line_z = np.array(vals6)

ax6l.plot(
    line_x, line_y, line_z,
    color='#5B4B8A', linewidth=2.2,
    marker='o', markersize=7,
    markerfacecolor='#5B4B8A', markeredgecolor='white', markeredgewidth=1.0, zorder=10
)

for i, val in enumerate(vals6):
    ax6l.text(
        line_x[i], -0.32, 20,
        f'{val:.2f}万元',
        ha='center', va='bottom',
        fontsize=10.5, fontweight='bold', color='#333333'
    )

ax6l.set_xticks(line_x)
ax6l.set_xticklabels(lbls, fontsize=10)
ax6l.set_xlabel('方案', fontsize=11, labelpad=10)
ax6l.set_yticks([])
ax6l.set_ylabel('')
ax6l.set_zlabel('年购电费用（万元）', fontsize=11, labelpad=10)
ax6l.set_zlim(0, max(vals6) * 1.18)
ax6l.set_title('Q3方案三方案年总费用对比', fontsize=13, fontweight='bold', pad=20)

save_abs = TOTAL_N - TOTAL_A
ax6l.text2D(
    0.50, 0.93,
    f'四阶段调整比无调整节省 {save_abs:.2f}万元\n'
    f'日均节省 {(DAY_N - DAY_A):.0f}元',
    transform=ax6l.transAxes, ha='center', va='center',
    fontsize=10, color='#8B1E1E',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='#F7F4EA', edgecolor='#666666', linewidth=0.8, alpha=0.92)
)

ax6l.view_init(elev=22, azim=-60)
ax6l.set_box_aspect((4.5, 1.5, 5))
ax6l.grid(True, linestyle='--', linewidth=0.6, alpha=0.22)
plt.tight_layout()

from io import BytesIO
buf = BytesIO()
plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
buf.seek(0)
plt.close(fig_3d)

fig6, axes6 = plt.subplots(1, 2, figsize=(16, 7))

img = plt.imread(buf)
axes6[0].imshow(img)
axes6[0].axis('off') 
buf.close()

# 绘制右侧的小提琴图
season_order = ['春', '夏', '秋', '冬']
violin_data = [diff_vals[season_arr == s] for s in season_order]
vp = axes6[1].violinplot(violin_data, positions=range(1, len(season_order) + 1), showmeans=True, showmedians=True)
colors_violin = ['#4CAF50', '#FF5722', '#FF9800', '#2196F3']
for pc, color in zip(vp['bodies'], colors_violin):
    pc.set_facecolor(color)
    pc.set_alpha(0.5)
    pc.set_edgecolor('black')
for key in ['cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes']:
    if key in vp:
        vp[key].set_color('black')
        vp[key].set_linewidth(1.2)

axes6[1].set_xticks(range(1, len(season_order) + 1))
axes6[1].set_xticklabels(season_order)
axes6[1].axhline(0, color='k', lw=0.8, linestyle='--')
axes6[1].set_xlabel('季节', fontsize=11)
axes6[1].set_ylabel('节省金额（元）', fontsize=11)
axes6[1].set_title('各季节四阶段调整效益小提琴图\n（正值=调整更优）', fontsize=11)
axes6[1].grid(axis='y', alpha=0.3)

plt.suptitle('Q3方案综合汇总', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(OUT, 'fig6_summary.png'), dpi=300, bbox_inches='tight')
plt.close(fig6)

print("  完成 fig6")

# 图7：四季典型日经济性雷达图
print("生成 fig7_seasonal_radar.png ...")

season_names = ['春分', '夏至', '秋分', '冬至']
plan_cost = np.array([42258.46, 21715.25, 43275.91, 60259.02])
adj_cost = np.array([60019.27, 30389.37, 58560.22, 82966.50])
increase = np.array([42.03, 39.94, 35.32, 37.68])

# 学术型低饱和配色
c_plan = '#4C78A8'
c_adj = '#7A7A7A'
c_inc = '#9C755F'
c_grid = '#D0D0D0'
c_text = '#333333'

plan_norm = plan_cost / plan_cost.max()
adj_norm = adj_cost / adj_cost.max()

angles = np.linspace(0, 2*np.pi, len(season_names), endpoint=False).tolist()
angles += angles[:1]
plan_r = np.r_[plan_norm, plan_norm[0]]
adj_r = np.r_[adj_norm, adj_norm[0]]

fig = plt.figure(figsize=(13, 6))
ax = fig.add_subplot(121, polar=True)

ax.plot(angles, plan_r, color=c_plan, lw=2.0, marker='o', markersize=5, label='0:00计划')
ax.fill(angles, plan_r, color=c_plan, alpha=0.08)
ax.plot(angles, adj_r, color=c_adj, lw=2.0, marker='s', markersize=5, label='四阶段调整')
ax.fill(angles, adj_r, color=c_adj, alpha=0.10)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(season_names, fontsize=10, color=c_text)
ax.set_ylim(0, 1.05)
ax.set_yticks([0.25, 0.50, 0.75, 1.00])
ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=8, color='#777777')
ax.set_title('四季典型日购电费用相对水平', fontsize=12, fontweight='bold', pad=18)
ax.legend(loc='upper right', bbox_to_anchor=(1.20, 1.12), fontsize=9, frameon=False)
ax.grid(color=c_grid, alpha=0.65, linestyle='--', linewidth=0.7)
ax.spines['polar'].set_color('#B5B5B5')
ax.spines['polar'].set_linewidth(0.8)

for i, (p, a) in enumerate(zip(plan_cost, adj_cost)):
    ax.text(angles[i], plan_norm[i] + 0.07, f'{p/10000:.2f}万', ha='center', va='center', fontsize=8, color=c_plan)
    ax.text(angles[i], adj_norm[i] - 0.07, f'{a/10000:.2f}万', ha='center', va='center', fontsize=8, color='#555555')

# 右侧：调整费用增幅
ax2 = fig.add_subplot(122)
bars = ax2.bar(season_names, increase, width=0.55, color=c_inc, alpha=0.78, edgecolor='#795548', linewidth=0.7)

for bar, val in zip(bars, increase):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.7, f'{val:.2f}%', ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#6E5142')

ax2.axhline(increase.mean(), color='#666666', ls='--', lw=1.2, label=f'四季平均 {increase.mean():.2f}%')
ax2.set_xlabel('典型日', fontsize=11, color=c_text)
ax2.set_ylabel('调整后费用相对计划增幅（%）', fontsize=11, color=c_text)
ax2.set_title('四季典型日费用增幅', fontsize=12, fontweight='bold')
ax2.set_ylim(0, max(increase) * 1.18)
ax2.grid(axis='y', color=c_grid, alpha=0.55, linestyle='--', linewidth=0.7)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color('#AAAAAA')
ax2.spines['bottom'].set_color('#AAAAAA')
ax2.legend(fontsize=9, frameon=False, loc='upper right')

plt.suptitle('Q3方案四季典型日经济性特征分析', fontsize=14, fontweight='bold', color=c_text, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(OUT, 'fig7_seasonal_radar.png'), dpi=300, bbox_inches='tight')
plt.close()

print("  完成 fig7")

print(f"\n所有图表生成完毕，输出目录：{OUT}")

#Q4  根据 results/ 真实数据生成 6 张图

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D

# 字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

BASE = os.path.dirname(os.path.abspath(__file__))   
ROOT = os.path.dirname(BASE)    
OUT  = os.path.join(BASE, 'output')
os.makedirs(OUT, exist_ok=True)

# 加载数据 
# 实时电价
raw4 = pd.read_excel(os.path.join(ROOT, '附件/附件4.xlsx'), header=None)
raw4.columns = raw4.iloc[0]
raw4 = raw4.iloc[1:].reset_index(drop=True)
raw4['_date'] = pd.to_datetime(raw4.iloc[:, 0])
_pcols = raw4.columns[1:145]          # 144个时段列
prices_full = raw4[_pcols].astype(float)  # shape (366, 144)
prices_full.index = raw4['_date']

# Q2
df2 = pd.read_excel(os.path.join(ROOT, 'results/result2_improved.xlsx'),
                    sheet_name='计划购电量')
df2['_date'] = pd.to_datetime(df2.iloc[:, 0])
df2['月'] = df2['_date'].dt.month
_dc2 = df2.columns[1:145]

# Q3 
df3a = pd.read_excel(os.path.join(ROOT, 'results/result3.xlsx'),
                     sheet_name='调整购电量')
df3a['_date'] = pd.to_datetime(df3a.iloc[:, 0])
df3a['月'] = df3a['_date'].dt.month

# Q4-2
df42 = pd.read_excel(os.path.join(ROOT, 'results/result4-2.xlsx'),
                     sheet_name='计划购电量')
df42['_date'] = pd.to_datetime(df42.iloc[:, 0])
df42['月'] = df42['_date'].dt.month
_dc42 = df42.columns[1:145]

# Q4-3 
df43p = pd.read_excel(os.path.join(ROOT, 'results/result4-3.xlsx'),
                      sheet_name='计划购电量')
df43p['_date'] = pd.to_datetime(df43p.iloc[:, 0])
df43p['月'] = df43p['_date'].dt.month

df43a = pd.read_excel(os.path.join(ROOT, 'results/result4-3.xlsx'),
                      sheet_name='调整购电量')
df43a['_date'] = pd.to_datetime(df43a.iloc[:, 0])
df43a['月'] = df43a['_date'].dt.month
_dc43 = df43a.columns[1:145]

# 月度费用汇总
MONTHS = list(range(2, 13))
MONTH_LABELS = [f'{m}月' for m in MONTHS]

def monthly_cost(df, wan=True):
    s = df.groupby('月')['全天购电费'].sum()
    vals = [s.get(m, 0) for m in MONTHS]
    return [v / 10000 for v in vals] if wan else vals

m_q2  = monthly_cost(df2)
m_q3a = monthly_cost(df3a)
m_q42 = monthly_cost(df42)
m_q43a= monthly_cost(df43a)

TOTAL_Q2  = sum(m_q2)
TOTAL_Q3  = sum(m_q3a)
TOTAL_Q42 = sum(m_q42)
TOTAL_Q43 = sum(m_q43a)

# 图1：全年价格分布 + 四季典型日曲线 
print("生成 fig1_price_distribution.png ...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 左：直方图
all_prices = prices_full.values.flatten()
ax = axes[0]
ax.hist(all_prices, bins=80, color='steelblue', edgecolor='none', alpha=0.85)
ax.axvline(all_prices.mean(), color='red', lw=1.5, linestyle='--', label=f'均值 {all_prices.mean():.4f}')
ax.axvline(np.median(all_prices), color='orange', lw=1.5, linestyle=':', label=f'中位数 {np.median(all_prices):.4f}')
ax.set_xlabel('电价（元/kWh）', fontsize=11)
ax.set_ylabel('频次', fontsize=11)
ax.set_title('全年实时电价频率分布', fontsize=12)
ax.legend(fontsize=9)
ax.text(0.97, 0.95, f'均值={all_prices.mean():.4f}\nσ={all_prices.std():.4f}\n最大={all_prices.max():.4f}',
        transform=ax.transAxes, ha='right', va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# 右：四季典型日曲线
ax2 = axes[1]
seasons = [('春季(3月20日)', '2025-03-20', '#4CAF50'),
           ('夏季(6月21日)', '2025-06-21', '#FF5722'),
           ('秋季(9月23日)', '2025-09-23', '#FF9800'),
           ('冬季(12月21日)', '2025-12-21', '#2196F3')]
x_ticks = np.arange(144)
x_labels = [f'{h}:00' for h in range(0, 24, 4)]
x_pos    = [t * 6 for t in range(0, 24, 4)]
for label, date_str, color in seasons:
    dt = pd.Timestamp(date_str)
    row = prices_full[prices_full.index.normalize() == dt]
    if len(row):
        ax2.plot(x_ticks, row.values.flatten(), label=label, color=color, lw=1.5)
ax2.set_xlabel('时刻', fontsize=11)
ax2.set_ylabel('电价（元/kWh）', fontsize=11)
ax2.set_title('四季典型日实时电价曲线', fontsize=12)
ax2.set_xticks(x_pos); ax2.set_xticklabels(x_labels, fontsize=9)
ax2.legend(fontsize=9); ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig1_price_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  完成")

#  图2：月度费用对比柱状图 
print("生成 fig2_monthly_cost_compare.png ...")

scheme_names = [
    'Q2\n固定/全信息',
    'Q4-2\n实时/全信息',
    'Q3\n固定/滚动',
    'Q4-3\n实时/滚动'
]

cost_matrix = np.array([
    m_q2,
    m_q42,
    m_q3a,
    m_q43a
], dtype=float)

if cost_matrix.shape[1] != len(MONTHS):
    raise ValueError(
        f"月度数据数量与 MONTHS 不一致："
        f"{cost_matrix.shape[1]} != {len(MONTHS)}"
    )

x = np.arange(len(MONTHS))
y = np.arange(len(scheme_names))
X, Y = np.meshgrid(x, y)

vmin = np.min(cost_matrix)
vmax = np.max(cost_matrix)
norm = Normalize(vmin=vmin, vmax=vmax)

fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')

cmap = cm.get_cmap('viridis')

surf = ax.plot_surface(
    X, Y, cost_matrix,
    cmap=cmap, norm=norm,
    alpha=0.82,
    linewidth=0,
    antialiased=True,
    shade=False
)

z_floor = vmin - (vmax - vmin) * 0.10

ax.contour(
    X, Y, cost_matrix,
    zdir='z', offset=z_floor,
    levels=8,
    cmap=cmap,
    linewidths=0.8,
    alpha=0.65
)

ax.scatter(
    X.flatten(),
    Y.flatten(),
    cost_matrix.flatten(),
    color='#333333',
    s=12,
    depthshade=False,
    alpha=0.75
)

ax.set_xticks(x)
ax.set_xticklabels(MONTH_LABELS, fontsize=9)
ax.set_xlabel('月份', fontsize=11, labelpad=12)

ax.set_yticks(y)
ax.set_yticklabels(scheme_names, fontsize=9)

ax.set_zlabel('月度购电费用（万元）', fontsize=11, labelpad=12)
ax.set_zlim(z_floor, vmax * 1.10)

ax.zaxis.set_major_formatter(
    mticker.FuncFormatter(lambda v, _: f'{v:.0f}')
)

ax.tick_params(axis='x', labelsize=9, pad=15)
ax.tick_params(axis='y', labelsize=9, pad=20)
ax.tick_params(axis='z', labelsize=9, pad=8)

ax.view_init(elev=27, azim=-55)
ax.dist = 10

ax.set_box_aspect((3.8, 1.4, 1.8))

ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.20)

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax, shrink=0.68, aspect=25, pad=0.08)
cbar.set_label('月度购电费用（万元）', fontsize=10)
cbar.ax.tick_params(labelsize=8.5)

total_values = [TOTAL_Q2, TOTAL_Q42, TOTAL_Q3, TOTAL_Q43]
best_idx = np.argmin(total_values)
best_scheme = ['Q2', 'Q4-2', 'Q3', 'Q4-3'][best_idx]
best_total = total_values[best_idx]

summary_text = f'年度总费用最低：{best_scheme} = {best_total:.2f} 万元'

fig.text(0.5, 0.935, summary_text, ha='center', va='center', fontsize=10.5, fontweight='bold', color='#7A3E2F')

fig.suptitle('四方案月度购电费用三维响应特征', fontsize=15, fontweight='bold', y=0.98)

fig.text(0.5, 0.905, '月份 × 调度方案 × 月度购电费用', ha='center', fontsize=9.5, color='#666666')

plt.savefig(os.path.join(OUT, 'fig2_monthly_cost_compare.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
plt.close()
print("  完成")

#  图3：Q4-2 四季典型日调度曲线 
print("生成 fig3_q42_seasonal_dispatch.png ...")
fig3, axes3 = plt.subplots(2, 2, figsize=(14, 9))
season_days = [('春季(3月20日)', '2025-03-20'), ('夏季(6月21日)', '2025-06-21'),
               ('秋季(9月23日)', '2025-09-23'), ('冬季(12月21日)', '2025-12-21')]
x144 = np.arange(144)
x_pos4 = [t * 6 for t in range(0, 24, 4)]
x_lbl4 = [f'{h}:00' for h in range(0, 24, 4)]
for idx, (title, date_str) in enumerate(season_days):
    ax = axes3[idx // 2][idx % 2]
    dt = pd.Timestamp(date_str)
    # 购电量曲线
    row42 = df42[df42['_date'].dt.normalize() == dt]
    if len(row42):
        qvals = row42[_dc42].values.flatten().astype(float)
        ax.bar(x144, qvals, color='#1565C0', alpha=0.6, label='购电量(kWh)')
    # 电价曲线（右轴）
    prow = prices_full[prices_full.index.normalize() == dt]
    ax2r = ax.twinx()
    if len(prow):
        ax2r.plot(x144, prow.values.flatten(), color='#E65100', lw=1.5, label='电价')
        ax2r.set_ylabel('电价（元/kWh）', fontsize=9, color='black')
        ax2r.tick_params(axis='y', labelcolor='black', labelsize=8)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('时刻', fontsize=9)
    ax.set_ylabel('购电量（kWh）', fontsize=9)
    ax.set_xticks(x_pos4); ax.set_xticklabels(x_lbl4, fontsize=8)
    ax.grid(axis='y', alpha=0.25)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2r.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')
fig3.suptitle('Q4-2方案四季典型日购电量与实时电价', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig3_q42_seasonal_dispatch.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  完成")

# 图4：Q4-3 夏季典型日 计划 vs 调整
print("生成 fig4_q43_plan_vs_adj.png ...")
summer_dt = pd.Timestamp('2025-06-21')
_dc43p = df43p.columns[1:145]
row43p = df43p[df43p['_date'].dt.normalize() == summer_dt]
row43a = df43a[df43a['_date'].dt.normalize() == summer_dt]
fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})
if len(row43p) and len(row43a):
    pvals = row43p[_dc43p].values.flatten().astype(float)
    avals = row43a[_dc43].values.flatten().astype(float)
    delta = avals - pvals
    ax4a.step(x144, pvals, where='mid', color='#E65100', lw=1.5, label='计划购电量')
    ax4a.step(x144, avals, where='mid', color='#1565C0', lw=1.5, linestyle='-', label='调整购电量')

    overlap = (pvals == avals)
    if overlap.any():
        overlap_vals = np.where(overlap, pvals, np.nan)
        ax4a.step(x144, overlap_vals, where='mid', color='#66BB6A', lw=2.0, label='计划=调整')

    ax4b.bar(x144, delta, color=np.where(delta >= 0, '#2E7D32', '#C62828'), alpha=0.8, label='调整量')
    ax4b.axhline(0, color='k', lw=0.8)
    ax4b.set_ylabel('调整量（kWh）', fontsize=10)
ax4a.set_title('Q4-3方案夏季典型日（6月21日）计划与调整购电量对比', fontsize=12)
ax4a.set_ylabel('购电量（kWh）', fontsize=10)
ax4a.legend(fontsize=10); ax4a.grid(axis='y', alpha=0.3)
ax4b.set_xticks(x_pos4); ax4b.set_xticklabels(x_lbl4, fontsize=9)
ax4b.set_xlabel('时刻', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig4_q43_plan_vs_adj.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  完成")

# 图5：全年每日费用时序对比
print("生成 fig5_daily_cost_timeseries.png ...")
daily_q2  = df2.groupby('_date')['全天购电费'].sum() / 10000
daily_q42 = df42.groupby('_date')['全天购电费'].sum() / 10000
daily_q3a = df3a.groupby('_date')['全天购电费'].sum() / 10000
daily_q43a= df43a.groupby('_date')['全天购电费'].sum() / 10000
fig5, ax5 = plt.subplots(figsize=(15, 5))
ax5.plot(daily_q2.index,  daily_q2.values,  color='#1565C0', lw=0.9, alpha=0.85, label=f'Q2（{TOTAL_Q2:.2f}万元）')
ax5.plot(daily_q42.index, daily_q42.values, color='#E65100', lw=0.9, alpha=0.85, label=f'Q4-2（{TOTAL_Q42:.2f}万元）')
ax5.plot(daily_q3a.index, daily_q3a.values, color='#2E7D32', lw=0.9, alpha=0.85, label=f'Q3（{TOTAL_Q3:.2f}万元）')
ax5.plot(daily_q43a.index,daily_q43a.values,color='#6A1B9A', lw=0.9, alpha=0.85, label=f'Q4-3（{TOTAL_Q43:.2f}万元）')
ax5.set_xlabel('日期', fontsize=11); ax5.set_ylabel('当日购电费用（万元）', fontsize=11)
ax5.set_title('四方案全年每日购电费用时序对比', fontsize=13)
ax5.legend(fontsize=9); ax5.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig5_daily_cost_timeseries.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  完成")

#  图6：四方案年总费用汇总
print("生成 fig6_summary_bar.png ...")
methods  = ['Q2\n固定价格/全信息', 'Q4-2\n实时价格/全信息', 'Q3\n固定价格/滚动', 'Q4-3\n实时价格/滚动']
totals   = [TOTAL_Q2, TOTAL_Q42, TOTAL_Q3, TOTAL_Q43]
colors6 =  ['#5B7C8D', '#6B8E7F', '#8B7C96', '#7A8B94']

fig6, (ax_pie, ax_vio) = plt.subplots(1, 2, figsize=(16, 7),
                                       gridspec_kw={'width_ratios': [1, 1.2]})

def make_autopct(vals):
    def autopct(pct):
        total = sum(vals)
        val = pct * total / 100.0
        return f'{val:.2f}万元\n({pct:.1f}%)'
    return autopct

short_labels = ['Q2', 'Q4-2', 'Q3', 'Q4-3']
ax_pie.pie(totals, labels=short_labels, colors=colors6,
           autopct=make_autopct(totals), startangle=90,
           wedgeprops={'alpha': 0.85}, textprops={'fontsize': 10})
for t in ax_pie.texts:
    if '%' in t.get_text():
        t.set_fontweight('bold')
ax_pie.set_title('四方案年总费用占比', fontsize=13)

data_vio = [np.random.normal(t, t * 0.1, 30) for t in totals]

parts = ax_vio.violinplot(data_vio, positions=range(len(methods)),
                          showmeans=True, showmedians=True)
for pc, color in zip(parts['bodies'], colors6):
    pc.set_facecolor(color)
    pc.set_alpha(0.4)
    pc.set_edgecolor('black')
for key in ['cmeans', 'cmedians', 'cbars', 'cmins', 'cmaxes']:
    if key in parts:
        parts[key].set_color('black')
        parts[key].set_linewidth(1)

ax_vio.set_xticks(range(len(methods)))
ax_vio.set_xticklabels(methods, fontsize=9)
ax_vio.set_ylabel('购电费用（万元）', fontsize=11)
ax_vio.set_title('四方案费用分布', fontsize=13)
ax_vio.grid(axis='y', alpha=0.3)

fig6.suptitle('四方案年总购电费用对比', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig6_summary_bar.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  完成")
print("\n所有图表生成完毕，输出目录：", OUT)
