import torch
x = torch.randn(3, 4)
print(len(x))


soft = torch.nn.Softmax(dim=1)
x2 = soft(x)
print(x2)
y = torch.chunk(x2, 2, dim=1)
y = y
print(y[0])
print(y[1])
print(torch.sum(y[0], dim=1))
print(torch.sum(y[1], dim=1))
c1 = torch.sum(y[0],dim=1).unsqueeze(1)
c2 = torch.sum(y[1],dim=1).unsqueeze(1)
c = torch.cat((c1,c2),1)
print(c)
c3 = torch.log(c)
print(c3)


print('#####################')
# x2 =
y = torch.chunk(soft(x), 2, dim=1)
c = torch.cat((torch.sum(y[0],dim=1).unsqueeze(1),torch.sum(y[1],dim=1).unsqueeze(1)),1)




# import torch
#
#
# def function():
#     data1 = torch.rand([1, 3])
#     print("data1_shape: ", data1.shape)
#     print("data1: ", data1)
#
#     data2 = torch.sum(data1, dim=1)
#     print("data2_shape: ", data2.shape)
#     print("data2: ", data2)
#
#
# if __name__ == '__main__':
#     function()


