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
# References: https://github.com/IcarusWizard/MAE/blob/main/train_classifier.py
# ================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_device_batch_size', type=int, default=256)
    parser.add_argument('--base_learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--total_epoch', type=int, default=100)
    parser.add_argument('--warmup_epoch', type=int, default=5)
    parser.add_argument('--pretrained_model_path', type=str, default='vit-t-mae.pt')
    parser.add_argument('--output_model_path', type=str, default='vit-t-classifier-pretrained.pt')
    parser.add_argument('--exp_name', type=str, required=True, help='Nome esperimento')

    args = parser.parse_args()

    # Set reproducibility
    # System
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # or ":4096:8" for more memory usage
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(1337)
    torch.use_deterministic_algorithms(True)
    g = torch.Generator()
    g.manual_seed(0)

    exp_name = args.exp_name
    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True,
                                                 transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True,
                                               transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, load_batch_size, shuffle=False, num_workers=4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.pretrained_model_path is not None:
        model = torch.load(args.pretrained_model_path, map_location='cpu', weights_only=False)
        writer = SummaryWriter(os.path.join('logs', 'cifar10', 'pretrain-cls'))
    else:
        model = MAE_ViT()
        writer = SummaryWriter(os.path.join('logs', 'cifar10', 'scratch-cls'))
    model = ViT_Classifier(model.encoder, num_classes=10).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    acc_fn = lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())

    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256,
                              betas=(0.9, 0.999), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8),
                                0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    best_val_acc = 0
    step_count = 0
    optim.zero_grad()
    train_losses, train_acc = [], []
    test_losses, test_acc = [], []
    for e in range(args.total_epoch):
        model.train()
        losses = []
        acces = []
        for img, label in tqdm(iter(train_dataloader)):
            step_count += 1
            img = img.to(device)
            label = label.to(device)
            logits = model(img)
            loss = loss_fn(logits, label)
            acc = acc_fn(logits, label)
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
            acces.append(acc.item())
        lr_scheduler.step()
        avg_train_loss = sum(losses) / len(losses)
        avg_train_acc = sum(acces) / len(acces)
        train_losses.append(avg_train_loss)
        train_acc.append(avg_train_acc)
        print(f'In epoch {e}, average training loss is {avg_train_loss}, average training acc is {avg_train_acc}.')

        model.eval()
        with torch.no_grad():
            losses = []
            acces = []
            for img, label in tqdm(iter(val_dataloader)):
                img = img.to(device)
                label = label.to(device)
                logits = model(img)
                loss = loss_fn(logits, label)
                acc = acc_fn(logits, label)
                losses.append(loss.item())
                acces.append(acc.item())
            avg_val_loss = sum(losses) / len(losses)
            avg_val_acc = sum(acces) / len(acces)
            test_losses.append(avg_val_loss)
            test_acc.append(avg_val_acc)
            print(f'In epoch {e}, average validation loss is {avg_val_loss}, average validation acc is {avg_val_acc}.')

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            print(f'saving best model with acc {best_val_acc} at {e} epoch!')
            torch.save(model, args.output_model_path)

        writer.add_scalars('cls/loss', {'train': avg_train_loss, 'val': avg_val_loss}, global_step=e)
        writer.add_scalars('cls/acc', {'train': avg_train_acc, 'val': avg_val_acc}, global_step=e)

    # Create figure
    fig, [ax0, ax1] = plt.subplots(1, 2, figsize=(15, 6))

    # Plot Loss
    ax0.plot(train_losses, label='Train Loss')
    ax0.plot(test_losses, label='Test Loss')
    ax0.set_ylabel('Model Loss')
    ax0.set_xlabel('Epoch')
    ax0.grid(True)
    ax0.legend(['Train', 'Test'], loc='best')
    ax0.set_title('Loss function - ViT-pretrained with LR Adaptation')

    # Plot Accuracy
    ax1.plot(train_acc, label='Training Accuracy')
    ax1.plot(test_acc, label='Test Accuracy')
    ax1.set_ylabel('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.grid(True)
    ax1.legend(['Train', 'Test'], loc='best')
    ax1.set_title('Accuracy function - ViT-pretrained with LR Adaptation')
    values = [model.image_size, model.patch_size, model.emb_dim, model.emb_dim * 4, model.encoder_layer,
              model.encoder_head, args.total_epoch, args.batch_size, args.base_learning_rate, args.warmup_epoch]
    keys = ['image_size', 'patch_size', 'emb_dim', 'forward_expansion', 'encoder_layer', 'encoder_head', 'epochs',
            'batch_size', 'learning_rate_start', 'warmup_steps']

    # Convert dictionary to table format
    table_data = [[k, v] for k, v in zip(keys, values)]
    table = plt.table(cellText=table_data, colLabels=["Hyperparameter", "Value"],
                      cellLoc='center', loc='upper right', bbox=[1.05, 0, 0.4, 0.3])
    # Adjust layout to prevent overlap
    plt.tight_layout()
    # Show the plot
    plt.savefig(f'../plots/vit-pretrained/{exp_name}/ViT_loss_accuracy_pretrained.png', dpi=300, bbox_inches='tight')
    plt.show()
