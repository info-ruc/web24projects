import torch
c = torch.tensor(
    [[[[[0,1,2],
        [3,4,5]]]]]
)
c = c.squeeze()
print(c)