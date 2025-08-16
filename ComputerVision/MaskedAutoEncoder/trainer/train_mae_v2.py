import sys
import os
import argparse
import math
import torchvision
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm

from model import *


# =================================
# References: https://github.com/IcarusWizard/MAE/blob/main/mae_pretrain.py
# ================================

def create_params_table(hyperparams, vals) -> None:
    # Create table data
    params_table = [[key, value] for key, value in zip(hyperparams, vals)]

    # Create a new figure
    fig, ax = plt.subplots(figsize=(6, len(hyperparams) * 0.5))  # adjust size based on number of parameters
    ax.axis('off')  # Hide axes
    ax.table(cellText=params_table, colLabels=["Hyperparameter", "Value"],
             cellLoc='center', loc='center')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'../plots/mae/{exp_name}/hyperparams_mae_{exp_name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--max_device_batch_size', type=int, default=512)
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--total_epoch', type=int, default=2000)
    parser.add_argument('--warmup_epoch', type=int, default=200)
    parser.add_argument('--model_path', type=str, default='vit-t-mae.pt')
    parser.add_argument('--exp_name', type=str, required=True, help='Nome esperimento')

    args = parser.parse_args()

    # Set reproducibility
    # System
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # or ":4096:8" for more memory usage
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    g = torch.Generator()
    g.manual_seed(0)

    exp_name = args.exp_name
    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    train_dataset = torchvision.datasets.CIFAR10('../data', train=True, download=True,
                                                 transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    val_dataset = torchvision.datasets.CIFAR10('../data', train=False, download=True,
                                               transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, load_batch_size, shuffle=False, num_workers=4)
    writer = SummaryWriter(os.path.join('logs', 'cifar10', 'mae-pretrain'))

    model = MAE_ViT(mask_ratio=args.mask_ratio).to(device)

    # Number of trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95),
                              weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8),
                                0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    step_count = 0
    optim.zero_grad()
    avg_train_loss = []
    avg_test_loss = []
    for e in range(args.total_epoch):
        model.train()
        train_losses = []
        for img, label in tqdm(iter(train_dataloader)):
            step_count += 1
            img = img.to(device)
            predicted_img, mask = model(img)
            loss = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            train_losses.append(loss.item())
        lr_scheduler.step()
        avg_loss = sum(train_losses) / len(train_losses)
        avg_train_loss.append(avg_loss)
        print(avg_train_loss)
        writer.add_scalar('mae_loss', avg_loss, global_step=e)
        print(f'In epoch {e}, average traning loss is {avg_loss}.')

        ''' visualize the first 16 predicted images on val dataset'''
        model.eval()
        with torch.no_grad():
            test_losses = []
            for img, label in tqdm(iter(val_dataloader)):
                img = img.to(device)
                predicted_img, mask = model(img)
                loss = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio
                test_losses.append(loss.item())
            avg_loss = sum(test_losses) / len(test_losses)
            avg_test_loss.append(avg_loss)
            print(avg_test_loss)

            val_img = torch.stack([val_dataset[i][0] for i in range(16)])
            val_img = val_img.to(device)
            predicted_val_img, mask = model(val_img)
            predicted_val_img = predicted_val_img * mask + val_img * (1 - mask)
            img = torch.cat([val_img * (1 - mask), predicted_val_img, val_img], dim=0)
            img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)
            writer.add_image('mae_image', (img + 1) / 2, global_step=e)

        ''' save model '''
        torch.save(model.state_dict(), args.model_path)

    # Create a table for hyperparameters
    values = [model.image_size, model.patch_size, model.emb_dim, model.emb_dim * 4, model.encoder_layer,
              model.encoder_head, model.decoder_layer, model.decoder_head, model.mask_ratio, args.total_epoch,
              args.batch_size, args.base_learning_rate, args.warmup_epoch, args.weight_decay, total_params]
    keys = ['image_size', 'patch_size', 'emb_dim', 'forward_expansion', 'encoder_layer', 'encoder_head',
            'decoder_layer', 'decoder_head', 'mask_ratio', 'epochs', 'batch_size', 'learning_rate_start',
            'warmup_steps', 'weight_decay', 'trainable_params']

    create_params_table(keys, values)

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [15, 10]
    plt.plot(avg_train_loss, label='Train Loss', color="red")
    plt.plot(avg_test_loss, label='Test Loss', color="green")
    plt.xlim(1, args.total_epoch)
    plt.xticks(np.arange(1, args.total_epoch + 1, 100))
    plt.ylabel('Model Loss')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.legend(['Train', 'Test'], loc='best')
    plt.title('Loss function - Masked Autoencoder')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig(f'../plots/mae/{exp_name}/MAE_loss_{exp_name}.png', dpi=300, bbox_inches='tight')
    plt.show()
