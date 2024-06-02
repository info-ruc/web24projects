#coding: utf-8
import pynvml
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7" 
import torch
gpu_memory = 6 #GPU剩余显存阈值，如果达到这个数字就抢夺，单位是GB
pynvml.nvmlInit()
polling_list = [0,5,6,7] # 需要巡检的GPU ID编号，可自定义
while 1:
    for i in polling_list:
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print('第'+str(i)+'块GPU剩余显存'+str(meminfo.free/(1024**3))+'GB')
        if meminfo.free/(1024**2)>=gpu_memory*1024:
            a = torch.zeros((800, 800, 300), dtype=torch.float64, requires_grad=True).cuda(i)+200
            b = torch.randn((800, 300, 200), dtype=torch.float64, requires_grad=True).cuda(i)+100
            z = torch.matmul(a, b)
            print('GPU has been grabbed!')
        else:
            print("不符合剩余"+str(gpu_memory)+"GB显存需求")