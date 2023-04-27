import os
import argparse
import torch
from torch import nn
import torchmetrics
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from models.resnet18_cifar_reparam import resnet18
import re
import glob


def get_all_task_checkpoints(ckpt_dir):
    task_checkpoints = {}
    for task_ckpt in glob.glob(os.path.join(ckpt_dir, "*/*.ckpt")):
        task_id = re.search(r'task(\d+)', os.path.basename(task_ckpt)).group(1)
        task_checkpoints[task_id] = task_ckpt
    return task_checkpoints

class EmotionClassifier(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = resnet18()
        self.encoder.fc = nn.Identity()
        self.fc = nn.Linear(512, 7)
        self.loss = nn.CrossEntropyLoss()
        self.metrics_acc = torchmetrics.Accuracy()
        self.load_weight(self.args.ckpt_path)

    def load_weight(self, ckpt_path):
        print(f"Loading previous task checkpoint {ckpt_path}...")
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        for k in list(state_dict.keys()):
            if "encoder" in k:
                state_dict[k.replace("encoder.", "")] = state_dict[k]
            del state_dict[k]
        self.encoder.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        with torch.no_grad():
            z = self.encoder(x)
        y = self.fc(z)
        return y

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        acc = self.metrics_acc(y_hat, y)
        self.log('train_loss', loss)
        self.log("train_acc", acc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        acc = self.metrics_acc(y_hat, y)
        self.log('val_loss', loss)
        self.log('val_acc', acc, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            momentum=0.9
        )
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=5,
            max_epochs=self.args.max_epochs,
            warmup_start_lr=0.01 * self.args.lr,
            eta_min=0.01 * self.args.lr,
        )
        return [optimizer], [scheduler]


def parse_args():
    parser = argparse.ArgumentParser(description="Emotion Classification on RAF-DB")
    parser.add_argument("--run_name", type=str, default="simclr", help="Name of the Weights & Biases run")
    parser.add_argument("--project", type=str, default="RAF-DB", help="Name of the Weights & Biases project")
    parser.add_argument("--entity", type=str, default='pigpeppa',
                        help="Name of the Weights & Biases entity (team or user)")
    parser.add_argument("--offline", action="store_true", help="Run Weights & Biases logger in offline mode")
    parser.add_argument("--max_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training and testing")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for the optimizer")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the ckpt")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    pl.seed_everything(5)
    train_mean = [0.57520399, 0.44951904, 0.40121641]
    train_std = [0.20838688, 0.19108407, 0.18262798]
    test_mean = [0.57697346, 0.44934572, 0.40011644]
    test_std = [0.2081054, 0.18985509, 0.18132337]

    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(test_mean, test_std)
    ])

    train_data = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=train_transform)
    test_data = datasets.ImageFolder(os.path.join(args.data_path, 'test'), transform=test_transform)


    task_checkpoints = get_all_task_checkpoints(args.ckpt_path)
    for task_id, task_ckpt in task_checkpoints.items():
        args.ckpt_path = task_ckpt
        model = EmotionClassifier(args)

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=4)

        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        wandb_logger = WandbLogger(name=f"{args.run_name}_task_{task_id}", project=args.project, entity=args.entity,
                                   offline=args.offline)
        trainer = pl.Trainer(max_epochs=args.max_epochs, gpus=1, logger=wandb_logger, callbacks=[lr_monitor])
        trainer.fit(model, train_loader, test_loader)
