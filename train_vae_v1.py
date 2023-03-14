# Author Ziyu Zhan
# Creation Data : 2021/12/27
# amplitude MNIST VAE
#  ===========================
import torch
from torchvision import transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import math
import time
import VAE_D2NN
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./ResultRepo')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# global parameters
batch_size = 1000
lr = 1.0e-3
num_classes = 21
num_layers = 5
wd = 0.8e-2  # weight decay
step_size = 15  # lr
gamma = 0.8
padding = 75
model_name = 'VAEd2NN_layers=%d_classes=%d_lr=%.3f_wd=%.3f_ss=%d_gamma=%.3f' % (num_layers, num_classes, lr, wd, step_size, gamma)
print('model:{}'.format(model_name))
model_dir = "Models/" + model_name
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
#------------path for saved models : save 3 models
model_val1 = "Models/" + model_name + "/best_model1.pth"
path_best1 = os.path.abspath(model_val1)
model_val2 = "Models/" + model_name + "/best_model2.pth"
path_best2 = os.path.abspath(model_val2)
model_val3 = "Models/" + model_name + "/best_model3.pth"
path_best3 = os.path.abspath(model_val3)
model_train1 = "Models/" + model_name + "/best_train1.pth"
path_train1 = os.path.abspath(model_train1)
model_train2 = "Models/" + model_name + "/best_train2.pth"
path_train2 = os.path.abspath(model_train2)
model_train3 = "Models/" + model_name + "/best_train3.pth"
path_train3 = os.path.abspath(model_train3)
path_best = [path_best1, path_best2, path_best3]
path_train = [path_train1, path_train2, path_train3]
# DATA LOADER
transform = transforms.Compose([transforms.Resize(size=(50, 50)), transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST("./DATA", train=True, transform=transform, download=True)
data_train = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = torchvision.datasets.MNIST("./DATA", train=False, transform=transform, download=True)
data_val = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

model = VAE_D2NN.Net(num_layers=num_layers)
model.cuda()
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(device)
    model.cuda()
else:
    print('no gpu available')
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)
min_loss, min_train = float('inf'), float('inf')
idx_best, idx_train = 0, 0
num_epochs = 10000
check = 300
iters = 0
for epoch in range(num_epochs):
    start0 = time.time()
    running_loss = 0.0
    for i, (train_input, _) in enumerate(data_train, 1):
        train_input = train_input.cuda()  # b h w
        train_input = torch.squeeze(train_input)
        train_images = F.pad(train_input, pad=[padding, padding, padding, padding])
        optimizer.zero_grad()
        pedueso_imgae, mu, sigma = model(train_images)
        kl_div = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
        loss = criterion(pedueso_imgae, train_input) + kl_div
        loss.backward()
        optimizer.step()
        # for p in model.parameters():
        #     p.data.clamp_(0.5*math.pi, 1.5*math.pi)
        running_loss += loss.item()
        if i % check == 0:
            total = 0
            val_loss = 0.0
            model.eval()
            for j, (val_input,_) in enumerate(data_val, 1):
                val_input = val_input.cuda()
                train_input = torch.squeeze(train_input)
                train_images = F.pad(train_input, pad=[padding, padding, padding, padding])
                with torch.no_grad():
                    pedueso_imgae_val, mu, sigma = model(val_input)
                    kl_div2 = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
                    loss2 = criterion(pedueso_imgae_val, train_input) + kl_div
                    val_loss += loss2.item()
            validation_loss = val_loss / len(data_val)
            training_loss = running_loss / check
            print('[{}, {}] train_loss = {:.5f} val_loss = {:.5f}'.format(epoch + 1, i, training_loss, validation_loss))
            writer.add_scalar('train_loss', training_loss, iters)
            writer.add_scalar('val_loss', validation_loss, iters)

            if validation_loss < min_loss:
                print('saving a lowest loss model: best_model')
                min_loss = validation_loss
                torch.save({
                    'Model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'min_loss': min_loss,
                    'min_train': min_train,
                    'idx_best': idx_best
                }, path_best[idx_best])
                idx_best = idx_best + 1
                idx_best = idx_best % 3
            if training_loss < min_train:
                print('saving a lowest train_loss model: best_train')
                min_train = training_loss
                torch.save({
                    'Model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'min_loss': min_loss,
                    'min_train': min_train,
                    'idx_train': idx_train
                }, path_train[idx_train])
                idx_train = idx_train + 1
                idx_train = idx_train % 3
            iters += 1
            running_loss = 0.0
    print('one epoch time %.2f sec' % (time.time() - start0))
    f1 = model.phase1[4]
    f1 = f1.squeeze()
    f2 = f1.detach()
    f2 = 10 * math.pi * (torch.sin(2 * f2) + 1)
    f2 = f2 % (2 * math.pi)
    fig = plt.figure()
    plt.imshow(f2.cpu(), cmap=plt.cm.hsv, vmax=2 * math.pi, vmin=0 * math.pi)
    writer.add_figure('phase1 plate 5', figure=fig, global_step=epoch)
    f11 = model.phase2[4]
    f11 = f11.squeeze()
    f22 = f11.detach()
    f22 = 10 * math.pi * (torch.sin(2 * f22) + 1)
    f22 = f22 % (2 * math.pi)
    fig = plt.figure()
    plt.imshow(f22.cpu(), cmap=plt.cm.hsv, vmax=2 * math.pi, vmin=0 * math.pi)
    writer.add_figure('phase2 plate 5', figure=fig, global_step=epoch)
