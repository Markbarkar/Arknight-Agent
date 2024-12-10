import torch

actions = torch.tensor([[0, 1, 3],
                        [0, 5, 5],
                        [7, 7, 2]])
a = 2
b = 1
res = a or b
print(res)
