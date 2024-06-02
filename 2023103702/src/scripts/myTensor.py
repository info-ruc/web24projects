import torch
# 1*2*3*2
x1 = torch.tensor([[[[1,2],[3,4],[5,6]],[[7,8],[9,10],[11,12]]]])
x2 = torch.tensor([[[[13,14],[15,16],[17,18]],[[19,20],[21,22],[23,24]]]])

feature_size = x1.size(2)*x1.size(3)
x1 = x1.view(1,2,x1.size(2)*x1.size(3))
print('x1 ',x1.size())
x2 = x2.view(1,2,x2.size(2)*x2.size(3))

y = torch.transpose(x2,1,2)
z = torch.bmm(x1,y)
print('z ',z)
print(z.size())

a = (z/feature_size).view(1,-1)
print('a ',a)
print(a.size())
a = torch.nn.functional.normalize(torch.sign(a)*torch.sqrt(torch.abs(a)+1e-10))

print('a ',a)
print(a.size())