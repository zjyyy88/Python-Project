#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
'''
import keyword
keyword.kwlist
x  = "时代少年团"
print (x[0:] * 3)
print (x[0:5] + "马嘉祺")
print (x[0:3] )
print (x[-4:-1],end="" )
print(type(x))
'''
import sys
'''
################################################
###  \n 是换行   前面加r，取消换行的命令，变为普通字符
###   print 输出是默认换行输出，若要不换行，使用‘，’连续
for i in sys.argv:
    print (i)
print ('\n python 路径为',sys.path)
a=1 ;b=2 ;c=3
if a==b:
    print("adengyub")
else:
    print("buder\n dsab\n","dhsifwef")
if a!=b:
    print('-b')
else:
    print('cb')
##################################################
'''
#####################语句########################
#if 这种，以关键字开始，冒号结束：。比如上面每个关键字的一句后面都有冒号
#################################################
'''
#####################赋值操作###################
#每个变量在使用前都必须赋值，变量赋值以后该变量才会被创建。
#等号 = 用来给变量赋值。
#a=1   等号左边a是变量，等号右边1是存储在变量中的值
zjy = "yanjiusheng"
print(zjy)
zjy,hgl,wr=2002,"post",100.000
print(zjy,hgl,wr)
###############################################
'''

################数据类型（等号右边的）###############
#Python有五个标准的数据类型：Numbers（数字）String（字符串）List[列表] Tuple（元组）Dictionary{字典}
#Numbers数字：int（有符号整型）long（长整型，也可以代表八进制和十六进制）float（浮点型）complex（复数）
#String（字符串）：由数字、字母、下划线组成的一串字符。
##从左到右索引默认0开始的，最大范围是字符串长度少1；从右到左索引默认-1开始的，最大范围是字符串开头。冒号':'可以用于截取片段，星号用于重复'*'，加号用于'+'连接
str = 'you are pig!'
print(str) # 输出完整字符串
print(str[0])  # 输出字符串中的第一个字符
print(str[2:5])  # 输出字符串中第三个至第六个之间的字符串
print(str[2:])  # 输出从第三个字符开始的字符串
print(str * 2)  # 输出字符串两次
print(str + "TEST")  # 输出连接的字符串
list1= ["you","are","pig"]
#######[0####,1####,2####]
list2= ['no','i','am','not']
list3=['ddhewf','hcwefhwe','whduiwqhf','ff','fewh',24134,453,'we2']
print(list1)
print(list2*3)
print(list1[1])
print(list1[1:])
print(list1[0:2])
print(list1[-3:])
print(list1+list2)
print(list3)
print(list3[1:6:2])#前面截取，最后一个是步长
list3.append('CATL')####对list增加，append
print(list3)
##################################列表是有序的对象集合，字典是无序的对象集合
# 字典 键到值的映射########## 键是唯一的，值可以重复
房间号={'L5121':'张佳颖，王靓','L5122':'马嘉祺，黄明浩'}
xuehao={'202340100943':'张佳颖','202340100942':'进眼里'}
print (xuehao['202340100943'])
print(房间号['L5122'])

"""
from sympy.codegen import Print

"""
passowrd = 123456
if passowrd == 2108:
    print('登录成功')
elif passowrd ==0:
    print('请输入验证码')
#elif passowrd >0 and passowrd <96999999:
#    print("你很快就找到密码了")
#elif passowrd >0 or passowrd >99999:
#    print("再找找看")
elif (passowrd>0 and passowrd<9999) or (passowrd== 123456):
    print('ok')
"""
#############for循环######################
#核心是实现 “遍历有序序列” 的功能，以下分核心概念、语法、执行流程、示例四部分解释：Python 的for循环是迭代循环，可以依次遍历任意有序序列（如列表、字符串、元组等）中的每一个元素，自动完成 “取元素→执行代码→取下一个元素” 的过程，无需手动控制循环次数。
###################################
#for iterating_var in sequence:   #
#    statements(s)                #
###################################
#iterating_var：循环变量，每次循环会自动接收序列中的 “当前元素”；
#sequence：待遍历的序列（如列表[1,2,3]、字符串"abc"等）；
#statements(s)：循环体代码块，每次遍历元素时执行的代码（需缩进，Python 用缩进来标识代码块）。
#
###############执行流程（对应流程图）##################
#1.从 “序列suquence” 中取出第一个元素，赋值给iterating_var；    #
#2.执行statements(s)（循环体代码）；                  #
#3.从序列中取出下一个元素，重复步骤 2；                 #
#4.直到序列中没有剩余元素，循环结束。                   #
####################################################
'''
elements = ['Li' ,'In' ,'Cl']
for element in elements:
    print(f"当前元素是:  {element}" )####此处为f-string 一定要用{}包裹变量
fruits = [1,2,3,4,5,6,7,8]
for num in fruits :
    print(num)
    print(f"iphone{num}")
NSWS = [1,2,3,...,199,200]
for NSW in NSWS:
    print(f"当前离子步为{NSW}")

NSWSS= range(1,200)
for NSW in NSWSS:
    print(f"例子步数为：{NSW}")
'''
##############while循环语句########################
# while  条件语句condition：
#        执行语句statements
# 条件语句为真（非零，非空非null都为真），执行statements
# 条件语句为假，中止执行
#
#################################################
NSW=1
while NSW<10:               #为真
    print('当前离子步为',NSW) #执行此state
    NSW  += 1               #执行此state
print(NSW)                  #
print('离子步数已经达到最大值')