import pylab
from scipy import stats
from matplotlib import pyplot as plt
data1 = stats.gamma.rvs(2, loc=1.5, scale=2, size=10000) # 通过scipy生成服从gamma分布的10000个样本
# 拟合分布
from fitter import Fitter
plt.ion()
f1 = Fitter(data1, distributions=['gamma', 'rayleigh', 'uniform'], timeout =30)
f1.fit()  # 创建Fitter类
# f1.hist()
_ = pylab.hist(f1._data, bins=f1.bins, density=f1._density,alpha=0.5)
# pylab.grid(True)
# f1.plot_pdf(Nbest=1)
name = f1.df_errors.sort_values(by="sumsquare_error").index[0]
print(name)
pylab.plot(f1.x, f1.fitted_pdf[name], lw=2, label=name)

data1 = stats.gamma.rvs(2, loc=1.5, scale=1, size=10000) # 通过scipy生成服从gamma分布的10000个样本
# 拟合分布
from fitter import Fitter
plt.ion()
f1 = Fitter(data1, distributions=['gamma', 'rayleigh', 'uniform'], timeout =30)
f1.fit()  # 创建Fitter类
# f1.hist()
_ = pylab.hist(f1._data, bins=f1.bins, density=f1._density,alpha=0.5)
# pylab.grid(True)
# f1.plot_pdf(Nbest=1)
name = f1.df_errors.sort_values(by="sumsquare_error").index[0]
print(name)
pylab.plot(f1.x, f1.fitted_pdf[name], lw=2, label=name)


plt.show()
# f1.summary()  # 输出拟合结果
plt.savefig('test.jpg',bbox_inches = 'tight')