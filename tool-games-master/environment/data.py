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
model = []
model.append([0., 42., 63., 76., 83., 92., 94., 96., 96., 97., 98., 98., 99., 99., 100., 100., 100., 100., 100., 100., 100., 100.]) #basic
model.append([0., 10., 18., 26., 35., 41., 45., 46., 54., 56., 62., 69., 70., 73., 77., 81., 84., 85., 88., 91., 92., 92.]) #bridge
model.append([0., 9., 12., 20., 26., 35., 47., 50., 55., 58., 64., 67., 71., 77., 79., 81., 84., 84., 86., 88., 90., 90.]) #catapult
model.append([0., 9., 11., 15., 18., 23., 27., 32., 33., 36., 37., 41., 45., 47., 48., 49., 50., 53., 57., 60., 62., 62.]) #chaining
model.append([0., 57., 82., 90., 96., 98., 98., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100]) #gap
model.append([0., 1., 4., 6., 11., 13., 14., 14., 16., 17., 17., 18., 22., 23., 23., 25., 25., 25., 28., 28., 29., 29.]) #seesaw
model.append([0., 76., 96., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100.]) #unbox
model.append([0., 16., 28., 35., 42., 51., 57., 62., 65., 69., 74., 80., 83., 84., 87., 87., 91., 92., 92., 94., 97., 97.]) #unsupport
model.append([0., 88., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100.]) #falling_A
model.append([0., 6., 17., 28., 35., 44., 49., 55., 63., 70., 75., 79., 82., 85., 88., 88., 89., 90., 91., 93., 93., 93.]) #falling_B
model.append([0., 10., 21., 28., 37., 46., 64., 75., 82., 87., 91., 96., 96., 96., 99., 100., 100., 100., 100., 100., 100., 100.]) #launch_A
model.append([0., 3., 10., 13., 19., 26., 32., 36., 40., 45., 53., 58., 61., 62., 65., 67., 68., 71., 74., 75., 76., 76.]) #launch_B
model.append([0., 5., 10., 16., 19., 22., 23., 26., 31., 33., 34., 36., 38., 40., 41., 44., 47., 51., 52., 53., 55., 55.]) #prevention_A
model.append([0., 1., 4., 7., 12., 13., 16., 19., 24., 25., 26., 30., 30., 31., 33., 38., 41., 42., 44., 44., 44., 44.]) #prevention_B
model.append([0., 75., 91., 97., 99., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100.]) #shafts_A
model.append([0., 41., 58., 75., 83., 91., 94., 97., 99., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100.]) #shafts_B
model.append([0., 5., 15., 23., 24., 31., 37., 40., 44., 48., 53., 55., 57., 61., 63., 64., 65., 66., 69., 73., 75., 75.]) #table_A
model.append([0., 5., 15., 23., 24., 31., 37., 40., 44., 48., 53., 55., 57., 61., 63., 64., 65., 66., 69., 73., 75., 75.]) #table_B 这个有 bug 跑不起来不知道为什么
model.append([0., 55., 82., 91., 95., 97., 97., 98., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100.]) #tower_A
model.append([0., 49., 76., 89., 95., 98., 99., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100.]) #tower_B
val = []
val.append(model)


data_list = []
for a, task in enumerate(tasks):
    for b, group in enumerate(groups):
        vals = val[b][a]
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