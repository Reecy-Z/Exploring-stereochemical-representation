import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition
from mpl_toolkits.mplot3d import Axes3D

titles = ['BHA','SMC','ANSA']
file = './PCA/BHA/MD.csv'
yield_path = './PCA/BHA/yields.csv'
data = pd.read_csv(file,header = None)
yields = pd.read_csv(yield_path)
yields = yields['x']
pca = decomposition.PCA(n_components=3)
pca.fit(data)
data = pca.transform(data)
print(pca.explained_variance_ratio_)
x = data[:,0]
y = data[:,1]
z = data[:,2]
fig = plt.figure(figsize=(20,20))
plt.style.use('ggplot')
ax = plt.axes(projection='3d')
pic = ax.scatter3D(x, y, z,c=yields, marker='.',cmap='jet',s = 400)
plt.xlabel('PC1(14.00%)',fontsize=40)
plt.ylabel('PC2(12.63%)', rotation=38,fontsize=40)
ax.set_zlabel('PC3(9.97%)',fontsize=40)
plt.title('Suzuki-Miyaura coupling', fontsize=40, fontweight='bold')
cbar_ax = fig.add_axes([0.91, 0.11, 0.02, 0.77])
cb = fig.colorbar(pic, cax=cbar_ax)
cb.ax.tick_params(labelsize=30)
fig.text(0.97, 0.5, 'yield%', ha='center', va='center', rotation='vertical', fontsize=40)
plt.savefig('./PCA/BHA/PCA.png', dpi = 300)
