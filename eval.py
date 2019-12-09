import torch as th
import numpy as np

sellerPath = './models/seller.ckpt'
buyerPath = './models/buyer.ckpt'

model = th.load(sellerPath)
model.eval()
FloatTensor = th.FloatTensor
res = model(th.tensor([0,0,0,0,0,0]).type(FloatTensor).unsqueeze(0))
print(res)