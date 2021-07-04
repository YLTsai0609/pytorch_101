import torch
import numpy as np
from torch.autograd import Variable
'''
* Variable記得要大寫
使用PyTorch建構神經網路, 梯度計算是透過torch.autograd來完成,
當我們進行一系列計算, 並且想要獲取變量之間的梯度訊息, 需要 :
1. 建構計算圖, 用Variable將Tensor包裝起來, 行為計算圖中的節點, 
而Variable之間進行各種運算就算Tensor之間運算一樣, Variable支援幾乎所有的Tensor運算
2. 進行一系列運算之後, 可以執行backward()方法來計算出所有需要的梯度
3. 針對某個變量執行x.grad獲得想要的梯度值
https://zhuanlan.zhihu.com/p/29904755
'''

tensor = torch.FloatTensor([[1,2],[3,4]])
variable = Variable(tensor, requires_grad=True)
# requires_grad = True, 該variable涉及梯度計算, False 不涉及

print(tensor,type(tensor), 
      variable, type(variable), sep='\n')

t_out = torch.mean(tensor*tensor)
v_out = torch.mean(variable*variable)

print(t_out, v_out,
     type(t_out), type(v_out), sep='\n')
# v_out會包含計算圖的訊息

v_out.backward()
print(variable)
print(variable.data)
# print(variable.numpy()) 會噴錯, variable是計算圖, variable.data才是tensor
print(variable.grad)
print('change dtype : ',variable.data.numpy())
'''
v_out是一張計算圖，其中包含了variable,因此call反向傳播方法時,
variable的grad屬性會進行變化
d(v_out) / d(var) = 1/4*2*variable = 1/2*variable
'''
print(dir(Variable))
# Variable中有的屬性及方法?