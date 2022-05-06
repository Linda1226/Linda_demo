a = [(1, 2), (4, 1), (9, 10), (13, -3)]

a.sort(key=lambda x: x[0])

print(a)
# list=("A","B","C","A")
# print(list[-2])
# print(str(list["A"][-1]))

#print(0)
# def std(nums):
#     n = len(nums)
#     avg = sum(nums) / n
#     return (sum(map(lambda e: (e - avg) * (e - avg), nums)) / n) ** 0.5
# import numpy as np

# arr=[2,6,2,3,2]
# # arr_var=np.var(arr)
# print(std(arr))

# import urllib.request as request
# print(request.urlopen('http://www.baidu.com').read())
# class Person(object):
#     def __new__(cls,*arg,**args):
#         obj=super().__new__(cls)
#         return obj 
#     def __init__(self,name,age):
#         self.name=name
#         self.age=age

# s=Person("linda",23)
# print("")


# from re import S

# class Ob2:
#     def fun(self):
#         print("this is Ob2 fun!")
# class Ob:
#     def fun(self):
#         print("this is Ob fun!")

# class Student(Ob2,Ob):

#     def fun(self):
#         super().fun()
#         print("this is fun!")
    
#     @classmethod
#     def f(cls):
#         print("this is f !")

#     @staticmethod
#     def s():
#         print("this is static method!")
    
#     def __init__(self,name,age) -> None:
#         self.name=name
#         self.age=age

# s=Student("linda",29)
# # print(s.name,s.age) 
# s.fun()
# print(s.__doc__)
# s.s()
# s.f()

# def f(n):
#     if n <0:
#         return "false"
#     if n==1 or n==2:
#         return 1
#     else:
#         return f(n-1)+f(n-2)
        
    
# print(f(6))


# name="linda"
# age=24

# print("i'm %s,this year %d"% (name,age))

# print("i'm {0},{1}".format(name,age))



# s="hello,world"
# print(s.center(3,"*"))

# print(ord("林"))
# print(chr(26519))
# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
# import numpy as np
 
# # arr = np.array(np.arange(1, 10)).reshape([3, 3])
# arr = np.array([  [  [   [1, 2, 3], [4, 5, 6] ] , [   [7, 8, 9],    [10, 11, 12] ]  ]  ,[  [   [13, 14, 25],    [16, 17, 18] ] ,[   [19, 20, 21],    [22, 23, 24] ]  ]])
# print("原始")
# print(arr)
# b = tf.pad(arr, paddings=[[1, 1], [0, 0],[0,0],[0,0]])
# print("第一列加0")
# print(b.shape)
# # b = tf.pad(arr, paddings=[[1, 1], [1, 1],[1,1],[1,1]])
# # print("第一列加0")
# # print(b.shape)
# sess = tf.compat.v1.InteractiveSession()

# print(b.eval())






# #讲解abs_mean
# import tensorflow as tf
# import numpy as np
# # 降维
# # X = np.array([[0, 1, 2], [3, 4, 5]])
# # Y = tf.cast(X, tf.float32)

# # mean_all = tf.reduce_mean(Y)

# # mean_0 = tf.reduce_mean(Y, axis=0)
# # mean_1 = tf.reduce_mean(Y, axis=1)
# # print(mean_all)
# # print(mean_0)
# # print(mean_1)
# # 不降维


# X = np.array([[[[0, 1, 2], [3, 4, 5]],[[10, 11, 22], [13, 14, 15]]],[[[0, 1, 2], [3, 4, 5]],[[10, 11, 22], [13, 14, 15]]]])
# print("===============X.shape===============")
# print(X.shape)
# Y = tf.cast(X, tf.float32)

# #mean_all = tf.reduce_mean(Y, keepdims=True)

# mean_0 = tf.reduce_mean(Y, axis=0, keepdims=True)
# mean_1 = tf.reduce_mean(Y, axis=1, keepdims=True)
# mean_2 = tf.reduce_mean(Y, axis=2, keepdims=True)
# mean_3 = tf.reduce_mean(Y, axis=3, keepdims=True)
# # print("===============mean_all===============")
# # print(mean_all)
# print("===============mean_0===============")
# print(mean_0)
# print("===============mean_1===============")
# print(mean_1)
# print("===============mean_2===============")
# print(mean_2)
# print("===============mean_3===============")
# print(mean_3)
# #讲解abs_mean




# mean_all=tf.reduce_mean(xx, keepdims=False)
# mean_0=tf.reduce_mean(xx, axis=0, keepdims=False)
# mean_1=tf.reduce_mean(xx, axis=1, keepdims=False)
 
 
# with tf.Session() as sess:
#     m_a,m_0,m_1=sess.run([mean_all, mean_0, mean_1])
 
# print(m_a)    # output: 2.0
# print(m_0)    # output: [ 1.  2.  3.]
# print(m_1)    #output:  [ 2.  2.]
# s1=[10,20,30,40]
# # s2={20,30}
# # print(s1.intersection(s2))

# print()



# t=("a","b",2)
# print(t)
# print(type(t))
# items={"A","B","C"}
# price={1,2,3}

# d={item.lower():p for item,p in zip(items,price)}
# print(d)

# fp = open("D:/text.txt","a+")
# print("this is test file",file=fp)
# import keyword

# print(keyword.kwlist)
# name = "linda"

# print("wo shi ",name)
# n1=29
# print(n1,type(n1))

# from decimal import Decimal

# n1=3.1
# n2=2.2
# print(Decimal('3.1')+Decimal('2.2'))

# age=20

# name=str(age)
# print(type(name))


#输入函数测试

# v=input("please input v")


# print(v,type(v))

#整除
# print(11//2)


# a=True
# print(not a)

# s="hello world!"
# print('h' in s)
# num=int(input("请输入一个数字："))
# num2=int(input("请输入一个数字："))

# print( ("第一个数字",num,"大") if num > num2 else ("第二个数字"+str(num2)+"大") )
# # if num  >=90 and num  <= 100:
# #     print("number is ",num)
# # elif num >=80 and num <90:
# #     print("number is ",num)
# # else:
# #     print("number is ",num)
    
# r=range(0,101,2)
# print(list(r))
# i=0
# sum=0
# while i<100:
#     if i%2==0:
#         sum+=i
#     i=i+1

# print(sum)

# for item in r:
#     print(item)
# scores={'linda':10,"lise":20}

# for item in scores:
#     print(item,scores[item])

# list1=['hello','world',15]
# lis1=list(["heool","2",1])
# print(list1[-1])
# from turtle import shape
# import tensorflow as tf
# import numpy as np
# nb_blocks=1
# # [0,1)
# for i in range(nb_blocks):
#     print(i)
# # #x是 多少个集合  y = 每个集合中有几个集合 z = 最后的集合我多少行 j = 多少列
# x=np.random.random((50000, 32, 32, 3))*0.1 
# print(x)
# from tflearn.datasets import cifar10
# (X, Y), (testX, testY) = cifar10.load_data()
# #print(X

# Y = tflearn.data_utils.to_categorical(Y,10)

# img_prep = tflearn.ImagePreprocessing()
# print(img_prep)

# def a(shape=None, placeholder=None,
#                data_preprocessing=None, data_augmentation=None,
#                name="InputData"):
#     print(shape,data_preprocessing,data_augmentation)

# a(shape=1,data_preprocessing=2,data_augmentation=23)
# a = np.array(np.arange(1,10))
# a = a.reshape((3,3))
# b = tf.pad(a,[[0, 0], [0, 0]],"CONSTANT")

# print(b)

# list=[1,2,3,4,5,6]
# list2=[True,False]
# list[1:]=list2
# print(list)

# l=[i for i in range(1,11,2)]
# print(l)
# with tf.compat.v1.variable_scope('V1',reuse=None):  
#     a1 = tf.compat.v1.get_variable(name='a1', shape=[1,3], initializer=tf.constant_initializer(1))  
#     a2 = tf.compat.v1.Variable(tf.compat.v1.random_normal(shape=[2,3], mean=0, stddev=1), name='a2')  
# with tf.compat.v1.variable_scope('V2',reuse=True):  
#     a3 = tf.compat.v1.get_variable(name='a1', shape=[1],initializer=tf.constant_initializer(1))  
#     a4 = tf.compat.v1.Variable(tf.compat.v1.random_normal(shape=[2,3], mean=0, stddev=1), name='a2')  
    
# with tf.compat.v1.Session() as sess:  
#     sess.run(tf.compat.v1.initialize_all_variables())  
#     print (a1.name)  
#     print (a2.name)   
#     print (a3.name)   
#     print (a4.name)
# c=np.array([[2,3,4,5,6,7,8,9],[5,6,7,6,7,8,9,0]])
# c=c.reshape(8,2,1,1)
# print(c)
# import tensorflow as tf
# with tf.variable_scope('V1',reuse=None):
#     a1 = tf.get_variable(name='a1', shape=[1], initializer=tf.constant_initializer(1))
#     a2 = tf.Variable(tf.random_normal(shape=[2,3], mean=0, stddev=1), name='a2')
# with tf.variable_scope('V1',reuse=True):
#     a3 = tf.get_variable(name='a1', shape=[1],initializer=tf.constant_initializer(1))
#     a4 = tf.Variable(tf.random_normal(shape=[2,3], mean=0, stddev=1), name='a2')

# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     print (a1.name)
#     print (a2.name)
#     print (a3.name)
#     print (a4.name)
#     结果为：
# V1/a1:0
# V1/a2:0
# V1/a1:0
# V1_1/a2:0

