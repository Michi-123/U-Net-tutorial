
from dataset import *
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from PIL import Image
from modules import train_model, get_loss_train, validate_model, test_model, save_prediction_image, polarize
from save_history import *
import sys


if __name__ == "__main__":
    args = sys.argv
    call(args[1])
    pass
    
def run(model, num_epochs):

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
                                    num_workers=2, batch_size=2, shuffle=True) # ワーカー数をColab用に設定
    SEM_val_load = \
        torch.utils.data.DataLoader(dataset=SEM_val,
                                    num_workers=2, batch_size=2, shuffle=True)

    SEM_test_load = \
        torch.utils.data.DataLoader(dataset=SEM_test,
                                    num_workers=2, batch_size=1, shuffle=False)

    # Dataloader end
    
    model = torch.nn.DataParallel(model, device_ids=list(
        range(torch.cuda.device_count()))).cuda()

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizerd
    optimizer = torch.optim.RMSprop(model.module.parameters(), lr=0.001)

    # Parameters
    epoch_start = 0
    epoch_end = num_epochs # 推奨 2000
    
    if epoch_end < 10:
        save_frequency = 1
    elif epoch_end < 100:
        save_frequency = 10
    else:
        save_frequency = 100
    
    # Saving History to csv
    header = ['epoch', 'train loss', 'train acc', 'val loss', 'val acc']
    save_file_name = "../history/RMS/history_RMS3.csv"
    save_dir = "../history/RMS"

    # Saving images and models directories
    model_save_dir = "../history/RMS/saved_models3"
    image_save_path = "../history/RMS/result_images3"

    # Train
    print("Initializing Training!")
    for i in range(epoch_start, epoch_end):
        # train the model
        train_model(model, SEM_train_load, criterion, optimizer)
        train_acc, train_loss = get_loss_train(model, SEM_train_load, criterion)

        #train_loss = train_loss / len(SEM_train)
        print('Epoch', str(i+1), 'Train loss:', train_loss, "Train acc", train_acc)

        # Validation  (Error code)
        if False:
            val_acc, val_loss = validate_model(
                model, SEM_val_load, criterion, i+1, True, image_save_path)
            print('Val loss:', val_loss, "val acc:", val_acc)
            values = [i+1, train_loss, train_acc, val_loss, val_acc]
            export_history(header, values, save_dir, save_file_name)

        # save model every save frequency epoch
        if (i+1) % save_frequency == 0:
            print('Saved.')
            save_models(model, model_save_dir, i+1)

"""
# Test
print("generate test prediction")
test_model("../history/RMS/saved_models/model_epoch_440.pwf",
           SEM_test_load, 440, "../history/RMS/result_images_test")
"""

def debug():
    for batch, (images_v, masks_v, original_msk) in enumerate(SEM_val):
        print('batch', batch)
        print('images_v', images_v.shape)
        Image.fromarray(np.uint8(images_v[0].numpy().reshape(572,572)*255))
