# from advanced_model import CleanU_Net
from network import UNet
from dataset import *
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from PIL import Image
from modules import *
from save_history import *


if __name__ == "__main__":
   pass

def run(img_path, model_path):

   # '../data/test/images/'
   SEM_test = SEMDataTest(img_path)
   
   SEM_test_load = \
     torch.utils.data.DataLoader(dataset=SEM_test,
                                 num_workers=3, batch_size=1, shuffle=False)
   
   # Test
   print("generate test prediction")

   # "../test/latest.pwf"
   test_model(model_path, SEM_test_load, 0, "../result/")
