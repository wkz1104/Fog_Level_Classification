import matplotlib.pyplot as plt

# ========== 关键：设置中文字体，解决乱码 ==========
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Zen Hei', 'Heiti SC']  # 适配Windows/Linux/Mac
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

labels = ['无雾', '轻雾', '浓雾']
sizes = [847, 580, 760]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

plt.figure(figsize=(7,7), dpi=300)
plt.axis('equal')

# 画饼图，只显示真实数字
wedges, texts, autotexts = plt.pie(
    sizes,
    labels=labels,
    colors=colors,
    autopct=lambda p: f'{int(round(sum(sizes)*p/100))}',
    startangle=90,
    textprops={'fontsize':18, 'weight':'bold', 'color':'black'}  # 标签字体颜色改为黑色
)

# 内部数字超大加粗，同时改成黑色
for t in autotexts:
    t.set_fontsize(26)
    t.set_weight('bold')
    t.set_color('black')  # 数字颜色改为黑色

plt.savefig('fog_pie.png', dpi=300, bbox_inches='tight')
plt.show()