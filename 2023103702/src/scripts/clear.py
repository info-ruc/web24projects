import matplotlib.pyplot as plt

# 随意绘制一个样图
plt.plot([1, 2, 3, 4, 3, 2, 3])

# plt.rcParams['font.family'] = ['serif']
# plt.rcParams['font.serif'] = ['SunTimes']
plt.title('shfoshdf')
# 保存图为svg格式，即矢量图格式
# plt.savefig("test.svg", dpi=300, format="svg")
plt.show()
plt.savefig("clear.svg",dpi=600,format="svg")