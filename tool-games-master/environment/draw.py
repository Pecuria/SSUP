import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.stats import pearsonr

# ==========================================
# 1. 加载并清洗数据 (你提供的新逻辑)
# ==========================================
df = pd.read_csv('human_data.csv')

# --- 筛选超时 ---
timeout_mask = df['Time'] > 120000
df_filtered = df[~timeout_mask].copy()

# --- 处理分组统计 ---
def process_group(group):
    success_trial = group['SuccessTrial'].iloc[0]
    ever_reached = group['SuccessPlace'].any()
    success_rows = group[group['SuccessPlace'] == True]
    if not success_rows.empty:
        total_attempts = success_rows['AttemptNum'].min() + 1
    else:
        total_attempts = group['AttemptNum'].max() + 1
    return pd.Series({
        'SuccessTrial': success_trial,
        'EverReachedSuccessPlace': ever_reached,
        'TotalAttempts': total_attempts
    })

grouped_df = df_filtered.groupby(['ID', 'Trial']).apply(process_group).reset_index()

# --- 排除矛盾数据 ---
# contradict_mask = (grouped_df['SuccessTrial'] != grouped_df['EverReachedSuccessPlace'])
# final_grouped_df = grouped_df[~contradict_mask].copy()

final_grouped_df = grouped_df.copy()  # 根据你的要求，不排除矛盾数据

# ==========================================
# 2. 准备 Model 数据 (CDF 反推)
# ==========================================
tasks = [
    "Basic", "Bridge", "Catapult", "Chaining", "Gap",
    "SeeSaw", "Unbox", "Unsupport", "Falling_A", "Falling_B",
    "Launch_A", "Launch_B", "Prevention_A", "Prevention_B", "Shafts_A",
    "Shafts_B", "Table_A", "Table_B", "Towers_A", "Towers_B"
]
model_raw = [[0., 42., 63., 76., 83., 92., 94., 96., 96., 97., 98., 98., 99., 99., 100., 100., 100., 100., 100., 100., 100., 100.], [0., 10., 18., 26., 35., 41., 45., 46., 54., 56., 62., 69., 70., 73., 77., 81., 84., 85., 88., 91., 92., 92.], [0., 9., 12., 20., 26., 35., 47., 50., 55., 58., 64., 67., 71., 77., 79., 81., 84., 84., 86., 88., 90., 90.], [0., 9., 11., 15., 18., 23., 27., 32., 33., 36., 37., 41., 45., 47., 48., 49., 50., 53., 57., 60., 62., 62.], [0., 57., 82., 90., 96., 98., 98., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100], [0., 1., 4., 6., 11., 13., 14., 14., 16., 17., 17., 18., 22., 23., 23., 25., 25., 25., 28., 28., 29., 29.], [0., 76., 96., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100.], [0., 16., 28., 35., 42., 51., 57., 62., 65., 69., 74., 80., 83., 84., 87., 87., 91., 92., 92., 94., 97., 97.], [0., 88., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100.], [0., 6., 17., 28., 35., 44., 49., 55., 63., 70., 75., 79., 82., 85., 88., 88., 89., 90., 91., 93., 93., 93.], [0., 10., 21., 28., 37., 46., 64., 75., 82., 87., 91., 96., 96., 96., 99., 100., 100., 100., 100., 100., 100., 100.], [0., 3., 10., 13., 19., 26., 32., 36., 40., 45., 53., 58., 61., 62., 65., 67., 68., 71., 74., 75., 76., 76.], [0., 5., 10., 16., 19., 22., 23., 26., 31., 33., 34., 36., 38., 40., 41., 44., 47., 51., 52., 53., 55., 55.], [0., 1., 4., 7., 12., 13., 16., 19., 24., 25., 26., 30., 30., 31., 33., 38., 41., 42., 44., 44., 44., 44.], [0., 75., 91., 97., 99., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100.], [0., 41., 58., 75., 83., 91., 94., 97., 99., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100.], [0., 5., 15., 23., 24., 31., 37., 40., 44., 48., 53., 55., 57., 61., 63., 64., 65., 66., 69., 73., 75., 75.], [0., 5., 15., 23., 24., 31., 37., 40., 44., 48., 53., 55., 57., 61., 63., 64., 65., 66., 69., 73., 75., 75.], [0., 55., 82., 91., 95., 97., 97., 98., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100.], [0., 49., 76., 89., 95., 98., 99., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100.]]

model_summary_list = []
for i, task in enumerate(tasks):
    cdf = np.array(model_raw[i]) / 100.0
    pdf = np.diff(cdf, prepend=0)
    # success_prob = cdf[-1]         # 总成功率
    # fail_prob = 1.0 - success_prob # 始终失败的概率
    avg_att = np.sum(pdf * np.arange(len(cdf))) # + (fail_prob * (len(cdf) - 1))

    model_summary_list.append({
        'Trial': task,
        'Model_AvgAttempts': avg_att,
        'Model_Accuracy': cdf[-1],
        'TaskID': i + 1
    })
df_model_summary = pd.DataFrame(model_summary_list)

print("Model Summary:")
print(df_model_summary)

# ==========================================
# 3. 计算 Human 汇总指标 (基于清洗后的 final_grouped_df)
# ==========================================
# 仅筛选 20 个目标任务
final_grouped_df = final_grouped_df[final_grouped_df['Trial'].isin(tasks)]

human_summary = final_grouped_df.groupby('Trial').agg(
    Human_AvgAttempts=('TotalAttempts', 'mean'),
    Human_Attempts_SE=('TotalAttempts', 'sem'),
    Human_Accuracy=('SuccessTrial', 'mean')
).reindex(tasks).reset_index()

print("Human Summary:")
print(human_summary)
# 合并用于图 A/B
df_corr = pd.merge(human_summary, df_model_summary, on='Trial')

# ==========================================
# 4. 绘图：相关性散点图 (Figure A & B)
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
sns.set_style("white")
scatter_kwargs = dict(color='black', alpha=0.8, s=45, edgecolors='white', linewidth=0.5, zorder=3)

# --- 图 A: Average Attempts ---
r_val_a, _ = pearsonr(df_corr['Model_AvgAttempts'], df_corr['Human_AvgAttempts'])
ax1.errorbar(df_corr['Model_AvgAttempts'], df_corr['Human_AvgAttempts'], 
             yerr=df_corr['Human_Attempts_SE'], fmt='none', ecolor='black', elinewidth=1, capsize=0, zorder=2)
ax1.scatter(df_corr['Model_AvgAttempts'], df_corr['Human_AvgAttempts'], **scatter_kwargs)
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlim(-0.5, 11); ax1.set_ylim(-0.5, 11)
ax1.plot([-1, 12], [-1, 12], '--', color='lightgray', linewidth=1, zorder=1)
for i, row in df_corr.iterrows():
    ax1.text(row['Model_AvgAttempts']+0.2, row['Human_AvgAttempts']+0.2, str(int(row['TaskID'])), fontsize=9)
ax1.set_xlabel('Full Model', fontsize=15, fontweight='bold')
ax1.set_ylabel('Human Attempts', fontsize=15, fontweight='bold')
ax1.set_title('A', loc='left', fontsize=18, fontweight='bold')
ax1.text(0.55, 0.1, f'r = {r_val_a:.2f}', transform=ax1.transAxes, fontsize=30)

# --- 图 B: Accuracy ---
r_val_b, _ = pearsonr(df_corr['Model_Accuracy'], df_corr['Human_Accuracy'])
ax2.scatter(df_corr['Model_Accuracy'], df_corr['Human_Accuracy'], **scatter_kwargs)
ax2.set_aspect('equal', adjustable='box')
ax2.set_xlim(-0.05, 1.05); ax2.set_ylim(-0.05, 1.05)
ax2.plot([-0.1, 1.2], [-0.1, 1.2], '--', color='lightgray', linewidth=1, zorder=1)
for i, row in df_corr.iterrows():
    ax2.text(row['Model_Accuracy']+0.02, row['Human_Accuracy']+0.02, str(int(row['TaskID'])), fontsize=9)
ax2.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax2.set_xlabel('Full Model', fontsize=15, fontweight='bold')
ax2.set_ylabel('Human Accuracy', fontsize=15, fontweight='bold')
ax2.set_title('B', loc='left', fontsize=18, fontweight='bold')
ax2.text(0.5, 0.1, f'r = {r_val_b:.3f}', transform=ax2.transAxes, fontsize=30)

# 绘制任务图例列表
for i, t in enumerate(tasks):
    col, row_idx = i % 5, i // 5
    fig.text(0.08 + col * 0.18, 0.18 - row_idx * 0.03, f"{i+1}. {t.replace('_', ' ')}", fontsize=12, color='#333333')

plt.tight_layout(rect=[0, 0.18, 1, 1])
plt.show()

# ==========================================
# 5. 绘图：累积成功曲线 (CDF Grid)
# ==========================================
# 准备模型 CDF 展平数据
model_cdf_list = []
for i, task in enumerate(tasks):
    for att, val in enumerate(model_raw[i]):
        model_cdf_list.append({'Attempts': att, 'SuccessRate': val/100.0, 'Task': f"{i+1}. {task}", 'Group': 'Full Model'})

# 准备人类 CDF 展平数据 (基于清洗后的 final_grouped_df)
human_cdf_list = []
for i, task in enumerate(tasks):
    subset = final_grouped_df[final_grouped_df['Trial'] == task]
    total_n = len(subset)
    # 仅统计真正成功的人在第几次成功
    success_attempts = subset[subset['EverReachedSuccessPlace'] == True]['TotalAttempts']
    for att in range(22):
        # 此时 TotalAttempts 是 1-based，cdf 统计 att 次内成功的比例
        count = (success_attempts <= att).sum()
        human_cdf_list.append({'Attempts': att, 'SuccessRate': count / total_n if total_n > 0 else 0, 
                               'Task': f"{i+1}. {task}", 'Group': 'Human'})

df_plot = pd.concat([pd.DataFrame(model_cdf_list), pd.DataFrame(human_cdf_list)], ignore_index=True)

sns.set_style("whitegrid", {'grid.color': '#e0e0e0'})
g = sns.FacetGrid(df_plot, col="Task", hue="Group", col_wrap=5, height=2.2, aspect=1.3, palette=['#e15759', '#4e79a7'])
g.map(plt.step, "Attempts", "SuccessRate", where='post', linewidth=1.5)

g.set_titles(col_template="{col_name}", size=10)
for ax in g.axes.flatten():
    ax.set_title(ax.get_title(), fontweight='bold')
    ax.set_ylim(-0.02, 1.05); ax.set_xlim(0, 21)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xlabel(""); ax.set_ylabel("")

g.fig.text(0.5, 0.06, 'Attempts', ha='center', fontsize=12, fontweight='bold')
g.fig.text(0.01, 0.5, 'Cumulative Solutions', ha='center', rotation='vertical', fontsize=12, fontweight='bold')
legend = g.fig.legend(handles=g.axes[0].get_legend_handles_labels()[0][::-1], 
                      labels=['Human', 'Full Model'], loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.01))
for text in legend.get_texts(): text.set_fontweight('bold')

plt.tight_layout(rect=[0.04, 0.1, 1, 0.98])
plt.subplots_adjust(hspace=0.4, wspace=0.25)
plt.show()