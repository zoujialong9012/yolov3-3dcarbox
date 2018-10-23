#!/usr/bin/python3
 
 # 可写函数说明
def changeme( mylist ):
    "修改传入的列表"
    aa = [1, 2, 3]
    bb = [4, 5, 6]
    mylist.append(aa)
    mylist.append(bb)
    print ("函数内取值: ", mylist)
    return
              
# 调用changeme函数
mylist = []
changeme( mylist )
print ("函数外取值: ", mylist)
