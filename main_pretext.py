from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import argparse
import os
from utils import metrics
from collections import OrderedDict
from torchvision.utils import save_image

from torch.utils.data import DataLoader

from models_pretext import *

from utils import logger
from utils import loss

import torch.nn as nn
import torch

import datasets_loading.dataset_pretext_loading


Mist = False
Sockeye = False
Local = True

MultiGPUs = False

os.makedirs("images/training", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)
os.makedirs("images/validation", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")

if Local:
    parser.add_argument("--dataset_path", type=str, default="D:\\ssl_data\\hack2_selected", help="path of the dataset")
    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
    parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
elif Mist:
    parser.add_argument("--dataset_path", type=str, default="/gpfs/fs0/scratch/u/uanazodo/uanazodo/dominic/us2mri/data/registered", help="path of the dataset")
    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
    parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
elif Sockeye:
    parser.add_argument("--dataset_path", type=str,
                        default="/scratch/st-zjanew-1/donzhang/us2mri/data/hack2",
                        help="path of the dataset")
    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
    parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
else:
    raise ValueError("no machine selected")

if MultiGPUs:
    parser.add_argument("--dist", type=bool, default=1, help="distribute or regular")
    parser.add_argument("--local_rank", default=-1, type=int)

parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--checkpoint_interval", type=int, default=1000, help="batch interval between model checkpoints")
parser.add_argument("--residual_blocks", type=int, default=6, help="number of residual blocks in the generator")
parser.add_argument("--lambda_adv", type=float, default=5e-3, help="adversarial loss weight")
parser.add_argument("--lambda_pixel", type=float, default=1e-2, help="pixel-wise loss weight")
opt = parser.parse_args()
print(opt)


if not MultiGPUs:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    torch.distributed.init_process_group(backend="nccl")

    world_size = torch.cuda.device_count()
    if world_size > 1:
        multi_gpu = True
    else:
        multi_gpu = False

    print("GPU num: ", world_size)
    print("torch distribution", torch.distributed.is_available())

    torch.cuda.set_device(opt.local_rank)
    device = torch.device("cuda", opt.local_rank)


ssl_model = SSLModel().to(device)


if opt.epoch != 0:
    # Load pretrained models
    ssl_model.load_state_dict(torch.load("saved_models/extractor_%d.pth" % opt.epoch))



if MultiGPUs:
    ssl_model = DistributedDataParallel(ssl_model, device_ids=[opt.local_rank], output_device=opt.local_rank, broadcast_buffers=False)

# Losses
criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
criterion_content = torch.nn.L1Loss().to(device)
criterion_pixel = loss.MSE_blur_loss().to(device)
criterion_seg = loss.TverskyLoss().to(device)
cross_entropy_loss = nn.BCEWithLogitsLoss().to(device)
criterion_info_nce = InfoNCE(temperature=0.5, negative_mode='unpaired').to(device)

# Optimizers
optimizer_M = torch.optim.Adam(ssl_model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

train_dataset = datasets_loading.dataset_pretext_loading.data_set(opt.dataset_path, mode='train')
val_dataset = datasets_loading.dataset_pretext_loading.data_set(opt.dataset_path, mode='val')
if MultiGPUs:
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=opt.local_rank)
    dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=opt.n_cpu, pin_memory=True, sampler=train_sampler)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=dist.get_world_size(), rank=opt.local_rank)
    val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=opt.n_cpu, pin_memory=True, sampler=val_sampler)
else:
    dataloader = DataLoader(train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu, pin_memory=True)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu, )

# ----------
#  Training
# ----------

log_train = logger.Train_Logger(os.getcwd(), "train_log")
log_val = logger.Train_Logger(os.getcwd(), "val_log")

def main():
    for epoch in range(opt.epoch, opt.n_epochs):
        # generator.train()
        ssl_model.train()

        # epoch_D_loss = metrics.LossAverage()
        epoch_Total_loss = metrics.LossAverage()
        epoch_content_loss = metrics.LossAverage()
        # epoch_segment_loss = metrics.LossAverage()
        # epoch_G_loss = metrics.LossAverage()
        epoch_pixel_loss = metrics.LossAverage()

        for i, imgs in enumerate(dataloader):

            batches_done = epoch * len(dataloader) + i

            # Configure model input
            # imgs_MR = imgs["MRI"]
            # imgs_MR = imgs_MR.float().to(device)

            imgs_US = imgs["US"]
            imgs_US = imgs_US.float().to(device)

            # print("imgs_US: ", imgs_US.shape)
            # s=3
            b, s, c, h, w = imgs_US.shape
            imgs_US = imgs_US.view(b*s, c, h, w)

            # training segmentor
            optimizer_M.zero_grad()

            proj, gens_US, random_US, saliency_US, reminder_US = ssl_model(imgs_US)
            proj = proj.view(b, s, -1)

            query = proj[:, 0, :]
            positive_key = proj[:, 1, :]
            negative_keys = proj[:, 2, :]

            loss_contrastive = criterion_info_nce(query, positive_key, negative_keys)
            loss_restore = criterion_pixel(gens_US, imgs_US)
            loss = loss_contrastive + loss_restore
            loss.backward()
            optimizer_M.step()

            epoch_Total_loss.update(loss.item(), imgs_US.shape[0])
            epoch_content_loss.update(loss_contrastive.item(), imgs_US.shape[0])
            epoch_pixel_loss.update(loss_restore.item(), imgs_US.shape[0])

            print(
                "[Epoch %d/%d] [Batch %d/%d][Total loss: %f, Contrastive: %f, Restore: %f]"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss.item(),
                    loss_contrastive.item(),
                    loss_restore.item(),
                )
            )

            if batches_done % opt.sample_interval == 0:
                img_grid = torch.cat((imgs_US, random_US, saliency_US, reminder_US, gens_US), -1)
                img_grid = datasets_loading.dataset_pretext_loading.denormalize(img_grid)
                # img_grid = torch.cat((img_grid, saliency + imgs_US), -1)
                save_image(img_grid, "images/training/%d.png" % batches_done, nrow=1, normalize=False)

        if MultiGPUs:
            torch.save(ssl_model.module.state_dict(), "saved_models/extractor_%d.pth" % epoch)
        else:
            torch.save(ssl_model.state_dict(), "saved_models/extractor_%d.pth" % epoch)

        temp_log_train = OrderedDict({'Total loss': epoch_Total_loss.avg, 'Contrastive loss': epoch_content_loss.avg, 'Restore loss': epoch_pixel_loss.avg})
        log_train.update(epoch, temp_log_train)

        ssl_model.eval()
        epoch_val_contrastive_loss = metrics.LossAverage()
        epoch_val_restore_loss = metrics.LossAverage()
        epoch_val_psnr = metrics.LossAverage()

        os.makedirs("images/validation/%d" % epoch, exist_ok=True)

        for i, imgs in enumerate(val_dataloader):
            # Configure model input
            imgs_US = imgs["US"]
            imgs_US = imgs_US.float().to(device)
            b, s, c, h, w = imgs_US.shape
            imgs_US = imgs_US.view(b * s, c, h, w)

            with torch.no_grad():
                proj, gens_US, _, _, _ = ssl_model(imgs_US)
                proj = proj.view(b, s, -1)

                query = proj[:, 0, :]
                positive_key = proj[:, 1, :]
                negative_keys = proj[:, 2, :]

                loss_contrastive = criterion_info_nce(query, positive_key, negative_keys)

                loss_restore = criterion_pixel(gens_US, imgs_US)
            epoch_val_contrastive_loss.update(loss_contrastive.item(), imgs_US.shape[0])
            epoch_val_restore_loss.update(loss_restore.item(), imgs_US.shape[0])

            img_grid = datasets_loading.dataset_pretext_loading.denormalize(torch.cat((imgs_US, gens_US), -1))
            save_image(img_grid, "images/validation/%d/%d.png" % (epoch, i), nrow=1, normalize=False)

            imgs_US = datasets_loading.dataset_pretext_loading.denormalize(imgs_US.detach())
            gens_US = datasets_loading.dataset_pretext_loading.denormalize(gens_US.detach())

            gens_US = gens_US.cpu().numpy() * 255
            imgs_US = imgs_US.cpu().numpy() * 255
            for ii in range(gens_US.shape[0]):
                gen_US_ = gens_US[ii, 0, :, :]
                img_US_ = imgs_US[ii, 0, :, :]
                epoch_val_psnr.update(metrics.calculate_psnr(gen_US_, img_US_), 1)

        val_log = OrderedDict(
            {'Val Con Loss': epoch_val_contrastive_loss.avg, 'Val Res Loss': epoch_val_restore_loss.avg,  'Val PSNR': epoch_val_psnr.avg})
        log_val.update(epoch, val_log)

if __name__ == '__main__':
    main()
