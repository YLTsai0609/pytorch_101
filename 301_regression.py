import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2*torch.rand(x.shape)
# check shape using shape/size
# unsqueeze -> reshape

x, y = Variable(x), Variable(y)
# if want to check the data
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

'''
繼承torch.nn.Module
修改init, forward
繼承torch.nn.Module中的init : 
super(Net, self).__init__()
回歸在output layer直接相等, 不用寫
'''
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature,n_hidden)
        self.predict = torch.nn.Linear(n_hidden,n_output)
    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

'''
input dimentions, hidden nerons, output
直接使用print(net)來確認網路結構 <--> model.summary() in keras 
'''
net = Net(1, 10, 1)
print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss_func = torch.nn.MSELoss()

plt.ion()   # something about plotting (interactive on)

for t in range(100):
    prediction = net(x)
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 5 == 0:
    # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
'''
初始化梯度為 0 
optimizer.zero_grad()
計算梯度
loss.backward()
更新梯度
optimizer.step()
prediction在前, true value在後
'''

