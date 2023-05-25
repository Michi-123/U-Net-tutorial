from advanced_model import CleanU_Net
from dataset import *
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from PIL import Image
from modules import *
from save_history import *


if __name__ == "__main__":
    # Dataset begin
    SEM_train = SEMDataTrain(
        '../data/train/images', '../data/train/masks')

    # TO DO: finish test data loading
    SEM_test = SEMDataTest(
        '../data/test/images/')
    SEM_val = SEMDataVal(
        '../data/val/images', '../data/val/masks')
    # Dataset end

    # Dataloader begins
    SEM_train_load = \
        torch.utils.data.DataLoader(dataset=SEM_train,
                                    num_workers=16, batch_size=2, shuffle=True)
    SEM_val_load = \
        torch.utils.data.DataLoader(dataset=SEM_val,
                                    num_workers=3, batch_size=1, shuffle=True)

    SEM_test_load = \
        torch.utils.data.DataLoader(dataset=SEM_test,
                                    num_workers=3, batch_size=1, shuffle=False)

    # Dataloader end

    # Model
    model = torch.load(model_path)
    model = torch.nn.DataParallel(model, device_ids=list(
        range(torch.cuda.device_count()))).cuda()
        

# Test
print("generate test prediction")
test_model("../history/latest.pwf",
           SEM_test_load, 120, "../history/")
