import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# 1. 模拟数据生成 (构造一个长格式 DataFrame)
np.random.seed(42)
tasks = [
    "1. Basic", "2. Bridge", "3. Catapult", "4. Chaining", "5. Gap",
    "6. SeeSaw", "7. Unbox", "8. Unsupport", "9. Falling (A)", "10. Falling (B)",
    "11. Launch (A)", "12. Launch (B)", "13. Prevention (A)", "14. Prevention (B)", "15. Shafts (A)",
    "16. Shafts (B)", "17. Table (A)", "18. Table (B)", "19. Towers (A)", "20. Towers (B)"
]
attempts = np.arange(0, 22)
groups = ['Model A']

data_list = []
for task in tasks:
    for group in groups:
        # 生成随机递增序列来模拟累积成功率
        difficulty = np.random.uniform(0.1, 0.5)
        vals = np.cumsum(np.random.rand(len(attempts)) * difficulty)
        vals = np.clip(vals / vals.max(), 0, 1.0) # 归一化到 0-1
        
        # 针对特定任务（如 Table/Towers）降低成功率，模拟原图中难度较大的情况
        if "Table" in task or "Towers" in task:
            vals *= 0.5
            
        vals = [0.,6.,17.,28.,35.,44.,49.,55.,63.,70.,75.,79.,82.,85.,88.,88.,89.,90.,91.,93.,93.,93.,]
        for i, att in enumerate(attempts):
            print(vals[i])
            data_list.append({
                'Attempts': att,
                'SuccessRate': vals[i]/100,
                'Task': task,
                'Group': group
            })

df = pd.DataFrame(data_list)

# 2. 绘图设置
sns.set_style("whitegrid", {'grid.color': '.9'}) # 设置白色网格背景
sns.set_context("notebook", font_scale=0.9)

# 创建分面网格 (4行5列)
g = sns.FacetGrid(
    df, 
    col="Task",      # 按任务分列
    hue="Group",     # 按组别区分颜色
    col_wrap=5,      # 每行显示5个图
    height=2.5,      # 每个子图的高度
    aspect=1.2,      # 宽高比
    palette=['#d62728', '#1f77b4'], # 蓝色和红色
    sharex=True, 
    sharey=True
)

# 使用 plt.step 绘制阶梯图
# where='post' 确保阶梯在尝试次数增加时向上跳变
g.map(plt.step, "Attempts", "SuccessRate", where='post', linewidth=1.5)

# 3. 细节美化
g.set_axis_labels("Attempts", "Cumulative Solutions")
g.set_titles(col_template="{col_name}", weight='bold')

for ax in g.axes.flatten():
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 21)
    # 将 Y 轴设置为百分比格式
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    # 设置 X 轴刻度间隔
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))

plt.tight_layout()
plt.show()