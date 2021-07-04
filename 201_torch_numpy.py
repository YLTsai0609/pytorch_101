import torch
import numpy as np

np_data = np.arange(6).reshape(2,-1)

torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()

print(np_data, torch_data,
     tensor2array,
     type(torch_data), torch.__version__, sep='\n')

# 就像numpy依樣進行操作
data = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data)
data2d = [[1, 2], [3,4]]
array2d = np.array(data)
tensor2d = torch.FloatTensor(data2d)

print(tensor.abs(),
     torch.sin(tensor),
     torch.mean(tensor), sep='\n')

# mm -> matrix-multiple
print(np.dot(array2d, array2d),
      torch.mm(tensor2d, tensor2d),sep='\n') 

# 物件式, 使用dot
# torch的dot只接受1D array
print(
    np.array(data).dot(np.array(data)),
    # tensor2d.dot(tensor2d), will return error
    torch.dot(torch.tensor([2, 1]), torch.tensor([2, 1])),
    sep='\n'
)

