from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from tqdm import tqdm
import argparse
from utils import metrics
from collections import OrderedDict
from torchvision.utils import save_image
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from models import *
from utils import logger
from utils import loss
import torch.nn as nn
import torch
import datasets_loading.dataset_loading

Sockeye = False
Local = True

MultiGPUs = False

os.makedirs("images/training", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)
os.makedirs("images/validation", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--fold", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")

if Local:
    parser.add_argument("--dataset_path", type=str, default="E:\\registered_V2", help="path of the dataset")
    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
    parser.add_argument("--sample_interval", type=int, default=10, help="interval between saving image samples")
elif Sockeye:
    parser.add_argument("--dataset_path", type=str, default="/arc/project/st-zjanew-1/donzhang/us2pdff/registered_V2", help="path of the dataset")
    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
    parser.add_argument("--sample_interval", type=int, default=10, help="interval between saving image samples")
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
parser.add_argument("--lambda_adv", type=float, default=5e-2, help="adversarial loss weight")
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

hr_shape = (256, 256)


# Initialize generator and discriminator

generator = US2MRI('saved_models_pretrained/extractor_40.pth').to(device)
discriminator = Discriminator((opt.channels * 2, *hr_shape)).to(device)


if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("saved_models/generator_%d.pth" % opt.epoch))
    discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth" % opt.epoch))


if MultiGPUs:
    generator = DistributedDataParallel(generator, device_ids=[opt.local_rank], output_device=opt.local_rank, broadcast_buffers=False)
    discriminator = DistributedDataParallel(discriminator, device_ids=[opt.local_rank], output_device=opt.local_rank, broadcast_buffers=False)

# Losses
criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
criterion_content = torch.nn.L1Loss().to(device)
criterion_pixel = torch.nn.L1Loss().to(device)
criterion_seg = loss.TverskyLoss().to(device)
cross_entropy_loss = nn.BCEWithLogitsLoss().to(device)
criterion_gaussian = loss.MSE_blur_loss(dim=2).to(device)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

train_dataset = datasets_loading.dataset_loading.data_set(opt.dataset_path, mode='train', fold_index=opt.fold)
val_dataset = datasets_loading.dataset_loading.data_set(opt.dataset_path, mode='val', fold_index=opt.fold)

if MultiGPUs:
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=opt.local_rank)
    dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=opt.n_cpu, pin_memory=True, sampler=train_sampler)


    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=dist.get_world_size(),
                                                                  rank=opt.local_rank)
    val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=opt.n_cpu, pin_memory=True,
                                sampler=val_sampler)
else:
    dataloader = DataLoader(train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, num_workers=opt.n_cpu, pin_memory=True)

# ----------
#  Training
# ----------

log_train = logger.Train_Logger(os.getcwd(), "train_log")
log_val = logger.Train_Logger(os.getcwd(), "val_log")

def main():
    generator.eval()
    print("start to build prototype")

    if os.path.exists("features_all.npy"):
        features_all = np.load("features_all.npy")
    else:
        features_all = []

        for epoch in range(5):
            print("epoch: ", epoch)
            for i, imgs in tqdm(enumerate(dataloader)):
                imgs_US = imgs["US"]
                imgs_US = imgs_US.float().to(device)
                with torch.no_grad():
                    hidden_states_out = generator.generator.swinViT(imgs_US, generator.generator.normalize)
                    dec4 = generator.generator.encoder10(hidden_states_out[4])
                    dec3 = generator.generator.decoder5(dec4, hidden_states_out[3])
                b, c, h, w = dec3.shape
                us_features = dec3.view(b, c * h * w)
                features_all.append(us_features.cpu())

        features_all = torch.cat(features_all, dim=0)
        print("features_all", features_all.shape)
        features_all = features_all.numpy()
        np.save("features_all.npy", features_all)

    kmeans = KMeans(n_clusters=64)
    kmeans.fit(features_all)

    print("finished building prototype")
    prototype = torch.tensor(kmeans.cluster_centers_)

    print("prototype shape: ", prototype.shape)

    generator.YNet.prototype.prototypes_us.data.copy_(prototype.to(generator.YNet.prototype.prototypes_us.device))
    generator.YNet.prototype.prototypes_pdff.data.copy_(prototype.to(generator.YNet.prototype.prototypes_us.device))


    for epoch in range(opt.epoch, opt.n_epochs):
        generator.train()

        epoch_D_loss = metrics.LossAverage()
        epoch_Total_loss = metrics.LossAverage()
        epoch_G_loss = metrics.LossAverage()
        epoch_pixel_guidance_loss = metrics.LossAverage()
        epoch_pixel_loss = metrics.LossAverage()
        epoch_seg_loss = metrics.LossAverage()

        for i, imgs in enumerate(dataloader):

            batches_done = epoch * len(dataloader) + i

            # # Configure model input
            # imgs_T1 = imgs["T1"]
            # imgs_T1 = imgs_T1.float().to(device)

            imgs_T2 = imgs["T2"]
            imgs_T2 = imgs_T2.float().to(device)

            imgs_PDFF = imgs["PDFF"]
            imgs_PDFF = imgs_PDFF.float().to(device)

            imgs_US = imgs["US"]
            imgs_US = imgs_US.float().to(device)

            labels = imgs["label"]
            labels = labels.float().to(device)

            # imgs_T1 = imgs_T1 * labels
            imgs_PDFF_liver = imgs_PDFF * labels
            imgs_US_liver = imgs_US * labels

            imgs_MR = imgs_T2

            discriminator.eval()

            # Adversarial ground truths
            valid = np.ones([imgs_US.size(0), 1, int(hr_shape[0] / 2 ** 4), int(hr_shape[1] / 2 ** 4)])
            valid = torch.from_numpy(valid).to(device)
            fake = np.zeros([imgs_US.size(0), 1, int(hr_shape[0] / 2 ** 4), int(hr_shape[1] / 2 ** 4)])
            fake = torch.from_numpy(fake).to(device)


            optimizer_G.zero_grad()
            restored_US, restored_PDFF, gens_PDFF_liver, gens_PDFF, masks = generator(imgs_US, imgs_MR)
            loss_pixel_PDFF = criterion_pixel(restored_PDFF, imgs_PDFF) + criterion_gaussian(restored_PDFF, imgs_PDFF)
            loss_pixel_US = criterion_pixel(restored_US, imgs_US) + criterion_gaussian(restored_US, imgs_US)

            loss_guidance = loss_pixel_US + loss_pixel_PDFF

            loss_pixel_target = criterion_pixel(gens_PDFF_liver, imgs_PDFF_liver) + criterion_gaussian(gens_PDFF_liver, imgs_PDFF_liver)
            loss_seg = criterion_seg(masks, labels)

            # Extract validity predictions from discriminator
            pred_real = discriminator(torch.cat((imgs_PDFF_liver, imgs_US), 1)).detach()
            pred_fake = discriminator(torch.cat((gens_PDFF_liver, imgs_US), 1))

            # Adversarial loss (relativistic average GAN)
            loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)


            # Total generator losssave_path
            loss_G = opt.lambda_adv * loss_GAN + loss_pixel_target + loss_seg + 2 * loss_guidance

            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            discriminator.train()

            US_augs = imgs['US_augs']
            US_augs = US_augs.float().to(device)

            angles_augs = imgs['angles'].numpy()
            translations_augs = imgs['trans'].numpy()

            optimizer_D.zero_grad()

            pred_real = discriminator((torch.cat((imgs_PDFF_liver, imgs_US), 1)))

            uncertainty_aug_seg = []
            uncertainty_aug_pred = []

            with torch.no_grad():
                for j in range(len(imgs_US)):
                    imgs_us_aug = US_augs[j, ...]
                    imgs_us_aug = torch.unsqueeze(imgs_us_aug, dim=1)
                    _, gen_PDFF_aug, masks_aug = generator(imgs_us_aug)
                    gen_PDFF_aug = gen_PDFF_aug.detach().cpu().numpy()
                    masks_aug = masks_aug.detach().cpu().numpy()

                    for jj in range(masks_aug.shape[0]):
                        gen_PDFF_aug[jj, 0, ...] = datasets_loading.dataset_loading.inverse_transform(gen_PDFF_aug[jj, 0, ...],
                                                                                      angles_augs[j, jj],
                                                                                      translations_augs[j, jj, 0],
                                                                                      translations_augs[j, jj, 1],
                                                                                      flags=True)
                        masks_aug[jj, 0, ...] = datasets_loading.dataset_loading.inverse_transform(masks_aug[jj, 0, ...],
                                                                                   angles_augs[j, jj],
                                                                                   translations_augs[j, jj, 0],
                                                                                   translations_augs[j, jj, 1],
                                                                                   flags=False)

                    seg_unct = np.var(masks_aug, axis=0)
                    seg_unct = torch.from_numpy(seg_unct).float().to(device)

                    pdff_unct = np.var(gen_PDFF_aug, axis=0)
                    pdff_unct = torch.from_numpy(pdff_unct).float().to(device)

                    seg_unct_weight = seg_unct / (seg_unct.max() + 1e-6)
                    pdff_unct_weight = pdff_unct / (pdff_unct.max() + 1e-6)

                    epsilon_seg = torch.sigmoid(torch.randn_like(masks[j, ...]).float().to(device))  # 仍然使用随机噪声
                    seg_aug = masks.detach()[j, ...] + seg_unct_weight * epsilon_seg * torch.sqrt(seg_unct)

                    epsilon_pred = torch.randn_like(gens_PDFF[j, ...]).float().to(device)
                    pdff_aug = gens_PDFF.detach()[j, ...] + pdff_unct_weight * epsilon_pred * torch.sqrt(pdff_unct)

                    uncertainty_aug_seg.append(seg_aug)
                    uncertainty_aug_pred.append(pdff_aug)

            uncertainty_aug_seg = torch.stack(uncertainty_aug_seg, dim=0)
            uncertainty_aug_pred = torch.stack(uncertainty_aug_pred, dim=0)

            # uncertainty_aug_pred_liver = apply_gaussian_smoothing(uncertainty_aug_pred, 3, 1) * uncertainty_aug_seg

            uncertainty_aug_pred_liver = uncertainty_aug_pred * uncertainty_aug_seg

            pred_fake = discriminator((torch.cat((uncertainty_aug_pred_liver.detach(), imgs_US), 1)))
            pred_fake2 = discriminator((torch.cat((gens_PDFF_liver.detach(), imgs_US), 1)))

            # Adversarial loss for real and fake images (relativistic average GAN)
            loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid) + criterion_GAN(pred_real - pred_fake2.mean(0, keepdim=True), valid)
            loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake) + criterion_GAN(pred_fake2 - pred_real.mean(0, keepdim=True), fake)

            # Total loss
            loss_D = (loss_real + loss_fake) / 2

            loss_D.backward()
            optimizer_D.step()

            # --------------
            #  Log Progress
            # --------------

            epoch_D_loss.update(loss_D.item(), imgs_US.shape[0])
            epoch_Total_loss.update(loss_G.item(), imgs_US.shape[0])
            epoch_seg_loss.update(loss_seg.item(), imgs_US.shape[0])
            epoch_pixel_guidance_loss.update(loss_guidance.item(), imgs_US.shape[0])
            epoch_G_loss.update(loss_GAN.item(), imgs_US.shape[0])
            epoch_pixel_loss.update(loss_pixel_target.item(), imgs_US.shape[0])

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, Guidance loss: %f] [G loss: %f, adv: %f, pixel: %f, seg: %f]"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_guidance.item(),
                    loss_G.item(),
                    loss_GAN.item(),
                    loss_pixel_target.item(),
                    loss_seg.item()
                )
            )

            if batches_done % opt.sample_interval == 0:
                img_grid = torch.cat((imgs_MR, restored_US, restored_PDFF, imgs_US, imgs_PDFF, imgs_PDFF_liver, gens_PDFF_liver), -1)
                img_grid = torch.cat((img_grid, labels, masks), -1)
                save_image(img_grid, "images/training/%d.png" % batches_done, nrow=1, normalize=False)

        if MultiGPUs:
            state_generator = {'epoch': epoch, 'model': generator.module.state_dict()}
            torch.save(state_generator, "saved_models/generator_%d.pth" % epoch)

            state_discriminator = {'epoch': epoch, 'model': discriminator.module.state_dict()}
            torch.save(state_discriminator, "saved_models/discriminator_%d.pth" % epoch)
        else:
            torch.save(generator.state_dict(), "saved_models/generator_%d.pth" % epoch)
            torch.save(discriminator.state_dict(), "saved_models/discriminator_%d.pth" % epoch)

        temp_log_train = OrderedDict({'Discriminator loss': epoch_D_loss.avg, 'Total loss': epoch_Total_loss.avg, 'GAN loss': epoch_G_loss.avg, 'Pixel Guidance loss': epoch_pixel_guidance_loss.avg,
                                      'Pixel loss': epoch_pixel_loss.avg, 'Seg loss': epoch_seg_loss.avg, })
        log_train.update(epoch, temp_log_train)

        generator.eval()

        epoch_val_mae = metrics.LossAverage()
        epoch_val_lpe = metrics.LossAverage()
        epoch_val_ssim = metrics.LossAverage()
        epoch_val_psnr = metrics.LossAverage()
        epoch_val_dice = metrics.DiceAverage(class_num=1)
        epoch_val_HD = metrics.HausdorffDistance()


        os.makedirs("images/validation/%d" % epoch, exist_ok=True)

        with torch.no_grad():
            for i, imgs in enumerate(val_dataloader):
                # Configure model input
                imgs_US = imgs["US"]
                imgs_US = imgs_US.float().to(device)

                imgs_PDFF = imgs["PDFF"]
                imgs_PDFF = imgs_PDFF.float().to(device)

                labels = imgs["label"]
                labels = labels.float().to(device)

                imgs_PDFF_liver = imgs_PDFF * labels


                with torch.no_grad():
                    gens_PDFF_liver, _, masks = generator(imgs_US)

                epoch_val_dice.update(masks, labels)
                epoch_val_HD.update(masks, labels, gens_PDFF_liver.shape[0])

                loss_mae = criterion_pixel(gens_PDFF_liver, imgs_PDFF_liver)
                epoch_val_mae.update(loss_mae.item(), 1)

                img_grid = torch.cat((imgs_US, imgs_PDFF, imgs_PDFF_liver, gens_PDFF_liver), -1)
                img_grid = torch.cat((img_grid, labels, masks), -1)
                save_image(img_grid, "images/validation/%d/%d.png" % (epoch, i), nrow=1, normalize=False)

                imgs_PDFF_liver = imgs_PDFF_liver.detach() / 2
                gens_PDFF_liver = gens_PDFF_liver.detach() / 2

                gens_PDFF_liver = gens_PDFF_liver.cpu().numpy()
                imgs_PDFF_liver = imgs_PDFF_liver.cpu().numpy()
                masks = masks.cpu().numpy()
                labels = labels.cpu().numpy()

                for ii in range(gens_PDFF_liver.shape[0]):
                    mask = masks[ii, 0, :, :]
                    label = labels[ii, 0, :, :]
                    gen_PDFF = gens_PDFF_liver[ii, 0, :, :]
                    img_PDFF = imgs_PDFF_liver[ii, 0, :, :]

                    lpe = np.mean(np.abs((crop_center_region(mask, gen_PDFF) - crop_center_region(label, img_PDFF))))
                    epoch_val_lpe.update(lpe, 1)

                    epoch_val_ssim.update(metrics.ssim(gen_PDFF * 255, img_PDFF * 255), 1)
                    epoch_val_psnr.update(metrics.calculate_psnr(gen_PDFF * 255, img_PDFF * 255), 1)
            val_log = OrderedDict(
                {'Val MAE': epoch_val_mae.avg, 'Val LPE': epoch_val_lpe.avg, 'Val SSIM': epoch_val_ssim.avg,
                 'Val PSNR': epoch_val_psnr.avg, 'Val Dice': epoch_val_dice.avg, 'Val HD': epoch_val_HD.avg})
            log_val.update(epoch, val_log)


def crop_center_region(segmentation_mask, image, crop_size=(6, 6)):
    # Step 1: Get the foreground (non-zero) indices
    foreground_coords = np.nonzero(segmentation_mask)

    if len(foreground_coords[0]) == 0:
        center_coord = [127, 127]
    else:
        center_coord = np.mean(foreground_coords, axis=1).astype(int)

    # Step 3: Calculate the crop boundaries (keeping the crop size within image bounds)
    h, w = segmentation_mask.shape  # Height and width of the mask
    crop_h, crop_w = crop_size  # Desired crop size

    start_h = max(0, center_coord[0] - crop_h // 2)
    start_w = max(0, center_coord[1] - crop_w // 2)

    # Make sure the crop doesn't go out of bounds
    end_h = min(h, start_h + crop_h)
    end_w = min(w, start_w + crop_w)

    # Step 4: Perform the crop
    cropped_region = image[start_h:end_h, start_w:end_w]

    return cropped_region

if __name__ == '__main__':
    main()
