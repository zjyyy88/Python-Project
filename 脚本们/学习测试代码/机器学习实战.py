# import kNN  # 移除不存在的模块导入
from numpy import *  # 导入numpy库的所有模块，用于科学计算：矩阵运算、数组操作等
import operator      # 导入运算符模块，主要用于排序操作
import os            # 导入操作系统相关模块 (本脚本暂时未用到，但常用于路径处理)
import matplotlib    # 导入绘图库 matplotlib
import matplotlib.pyplot as plt # 导入 matplotlib 的 pyplot 模块，用于绘制图形
import numpy as np   # 导入 numpy 库并简写为 np，方便后续调用

# 尝试强制使用 TkAgg 后端，这通常能解决弹不出窗口的问题
# 注意：这行代码最好在导入 pyplot 之前
'''
try:
    matplotlib.use('TkAgg')
except:
    pass
'''



def createDataset():
    """
    创建训练数据集和对应的标签
    """
    # 创建数据集数组 (4个样本，每个样本2个特征)
    group = array([[1,1.1],[1,1],[0,0],[0,0.1]])
    # 创建标签列表，对应上面的4个数据点
    labels = ['A','A','B','B']
    return group, labels

# 直接调用本文件中定义的函数，而不是 kNN.createDataset()
group, labels = createDataset() # 调用 createDataset 获取数据集和标签
print("Group:\n", group)        # 打印数据集
print("Labels:", labels)        # 打印标签列
def moviedataset():
    group = array([[3,104],[2,100],[1,81],[101,10],[99,5],[98,2]])
    labels=["爱情片","爱情片","爱情片","动作片","动作片","动作片"]
    return group,labels

def classify0(inX, dataset, labels, k):
    """
    k-近邻算法分类函数 (k-NN)：少数服从多数的思想，主要用于分类任务。
    参数:
    inX:     用于分类的输入向量 (测试数据 [x, y])
    dataset: 训练样本集 (NumPy数组)
    labels:  训练样本标签向量
    k:       选择最近邻居的数目 (整数)
    """
    # 1. 计算距离 (欧氏距离)
    datasetSize = dataset.shape[0]  # 获取数据集的行数，即样本数量
    
    # tile(inX, (datasetSize,1)) 将输入向量 inX 复制 datasetSize 行，构造成与 dataset 同维度的矩阵
    # 然后减去 dataset，得到向量差
    diffMat = tile(inX, (datasetSize,1)) - dataset
    
    sqDiffMat = diffMat**2          # 计算差值的平方
    sqDistances = sqDiffMat.sum(axis=1) # 对平方差按行求和 (axis=1)
    distances = sqDistances**0.5    # 开根号，得到最终的欧氏距离
    
    # argsort() 返回数组值从小到大的索引值，用于后续寻找最近的 k 个点
    sortedDistIndices = distances.argsort()
    
    classCount = {} # 定义一个字典，用于记录前 k 个最近邻的类别出现次数
    
    # 2. 选择距离最小的 k 个点
    for i in range(k):
        # 获取第 i 个最近邻的标签
        voteIlabel = labels[sortedDistIndices[i]]
        
        # 统计该标签出现的次数
        # get(key, default): 如果 key 不存在则返回 default (这里是0)，然后加 1
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        
    # 3. 排序
    # 对 classCount 字典进行排序
    # key=operator.itemgetter(1) 表示按照字典的值 (value，即出现次数) 进行排序
    # reverse=True 表示降序排序 (出现次数最多的排在前面)
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    
    # 4. 返回出现次数最多的类别标签
    # sortedClassCount[0] 是出现次数最多的 (label, count) 元组
    # 取其第 0 个元素即为标签
    return sortedClassCount[0][0]


# 测试分类1: [2, 1.5]
result1 = classify0(array([2,1.5]), group, labels, 3)
print(f"{result1}") #f用于包裹变量，如果不加f和大括号则无法输出变量值

# 测试分类2: [0, 0]
result2 = classify0(array([0,0]), group, labels, 3)
print(f"输入 [0, 0] 的分类结果: {result2}")

# 测试 moviedataset
movie_group, movie_labels = moviedataset()
result3 = classify0(array([100,20]), movie_group, movie_labels, 3)
print(f"电影推荐分类结果: {result3}")

# 定义数据文件绝对路径
#file=r'C:\Users\ZHANGJY02\PycharmProjects\PythonProject\.venv\111\Machine-Learning-in-Action-Python3-master\kNN_Project1\datingTestSet2.txt'
file=r'C:\Users\ZHANGJY02\PycharmProjects\PythonProject\脚本们\学习测试代码\Machine-Learning-in-Action-Python3-master\kNN_Project1\datingTestSet2.txt'
datingset= file # 将文件路径赋值给 datingset 变量


#############################################################################################
#1.把txt文本转化为矩阵
################################################
def file2matrix(filename):
    """
    将文本文件转换成NumPy矩阵
    参数:
        filename: 文件路径
    返回:
        returnMat: 特征矩阵 (NumPy matrix)
        classLabelVector: 类标签列表
    """
    fr = open(filename)             # 打开 filename 指定的文件
    arrayOLines = fr.readlines()    # 读取文件的所有行内容，并作为列表返回
    numberOfLines = len(arrayOLines)# 获取文件行数 (即样本总数)
    
    # 创建返回的矩阵: numberOfLines 行, 3 列 。训练集中有三列数据，3个特征标，最后一列是喜欢的程度，即分类的结果
    # zeros() 初始化全0矩阵，用于后续填入数据
    returnMat = zeros((numberOfLines,3)) 
    
    classLabelVector = []           # 创建一个空列表，用于存储每一行数据的分类标签
    index = 0                       # 初始化索引计数器，用于记录当前处理到矩阵的第几行
    
    for line in arrayOLines:        # 遍历文件的每一行
        line = line.strip()         #strip() 函数去除行首尾的不可见字符 (如换行符\n, 空格)
        listFromLine = line.split('\t') # 使用 split() 函数按制表符 '\t' 分割字符串，得到数据列表
        
        # 将分割后的前3个元素 (特征数据) 存入矩阵的第 index 行的所有列
        returnMat[index,:] = listFromLine[0:3] 
        
        # 将最后一个元素 (标签) 转换为 int 类型并添加到标签列表中
        # listFromLine[-1] 表示取列表的倒数第一个元素
        classLabelVector.append(int(listFromLine[-1])) 
        
        index += 1                  # 索引加1，准备处理下一行数据
    return returnMat, classLabelVector # 返回处理好的特征矩阵和标签列表

# 调用函数获取数据
# datingsetMat 接收返回的特征矩阵，classLabelVector 接收返回的标签列表
datingsetMat, classLabelVector = file2matrix(datingset)
#####################################################################################################


############################################################################################################
#不同的变量之间值差的太大，用knn计算距离时影响很大,把上面处理好的矩阵的数据，进一步转化为归一化的数值，最后重新给datingset赋值
############################################
def autoNorm(dataSet):
    minVals = dataSet.min(0)   # 获取每列的最小值 (axis=0 表示按列操作)
    maxVals = dataSet.max(0)   # 获取每列的最大值
    ranges = maxVals - minVals  # 计算每列的范围 (最大值 - 最小值)
    normDataSet = zeros(shape(dataSet)) # 创建一个与 dataSet 形状相同的全零矩阵，用于存储归一化后的数据
    m = dataSet.shape[0]       # 获取数据集的行数 (样本数量)
    normDataSet = dataSet - tile(minVals, (m,1)) # 每个元素减去对应列的最小值
    normDataSet = normDataSet / tile(ranges, (m,1)) # 每个元素除以对应列的范围，实现归一化
    return normDataSet, ranges, minVals # 返回归一化后的数据集、范围和最小值

# 调用 autoNorm 进行归一化
normDataSet, ranges, minVals = autoNorm(datingsetMat)

##############################################################################################################


########################################################################################################
#2.数据可视化
########################################################################
 # 在图形窗口中添加一个子图 (Axes)。111 表示 1行1列的第1个子图。
fig =   plt.figure()    # 创建一个新的图形窗口 (Figure)
ax  =  fig.add_subplot(111)
# 绘制散点图 scatter
# datingsetMat[:,0]: 取矩阵的第1列数据作为 x 轴坐标 (飞行里程)
# datingsetMat[:,1]: 取矩阵的第2列数据作为 y 轴坐标 (玩游戏时间)
# s=... : 设置点的大小 (Size)。这里用标签值乘以15，使得不同类别的点大小不同，方便区分
# c=... : 设置点的颜色 (Color)。这里直接用标签值作为颜色映射，不同类别颜色不同
ax.scatter(datingsetMat[:,0], datingsetMat[:,1],
           s=15.0*array(classLabelVector), c=array(classLabelVector))
plt.show()  # 显示图形窗口。注意：如果配置不对可能弹不出窗口。
fig =   plt.figure()    # 创建一个新的图形窗口 (Figure)
ax1  =  fig.add_subplot(111)
ax1.scatter(normDataSet[:,0], normDataSet[:,2],
           s=15.0*array(classLabelVector), c=array(classLabelVector))
plt.show()  # 显示图形窗口。注意：如果配置不对可能弹不出窗口。

# 打印结果
print(f"特征矩阵 (前5行):\n{datingsetMat[0:5]}") # 打印特征矩阵的前5行，方便检查数据是否正确
print(f"标签向量 (前20个): {classLabelVector[0:20]}") # 打印标签列表的前20个元素
############################################################################################################################

def datingClassTest():
    hoRatio = 0.10      # 测试集所占比例 (10%)
    datingDataMat, datingLabels = file2matrix(datingset) # 获取数据矩阵和标签列表
    normMat, ranges, minVals = autoNorm(datingDataMat)    # 归一化数据矩阵
    m = normMat.shape[0]   # 获取样本总数
    numTestVecs = int(m * hoRatio) # 计算测试集的样本数量
    errorCount = 0.0        # 初始化错误计数器
    
    for i in range(numTestVecs):  # 遍历测试集样本
        # 使用 classify0 函数进行分类，传入测试样本、训练样本、标签和 k 值 (这里 k=3)
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:],
                                     datingLabels[numTestVecs:m], 3)
        
        print(f"分类结果: {classifierResult}, 真实类别: {datingLabels[i]}") # 打印分类结果和真实类别
        
        if (classifierResult != datingLabels[i]): # 如果分类错误，错误计数器加1
            errorCount += 1.0
            
    print(f"错误率: {errorCount/float(numTestVecs)}") # 计算并打印错误率

# 调用测试函数
datingClassTest()