import torch
import torch.nn as nn

from data.dataset import build_dataloaders
from model import build, train_model
from rich import print
from argparse import ArgumentParser
import os

def main(args):
    # Build model (or load from checkpoint)
    model = build(backbone=args.backbone)
    if args.load: model.load_state_dict(torch.load(f'./checkpoints/{args.load}.pth'))

    # Build dataloaders
    train_loader, val_loader = build_dataloaders(batch_size=args.batch_size, 
                                                 num_workers=args.num_workers,
                                                 reload_dataset=args.load_dataset)
    
    # Build optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Build loss function
    criterion = nn.CrossEntropyLoss()
    # Train model
    train_model(model, optimizer, criterion, train_loader, val_loader,
                args.device, epochs=args.epochs, loss_weights=args.loss_weights)

    if not args.no_save:
        if not os.path.exists('./checkpoints'):
            os.mkdir('./checkpoints')
        print(f'[green]Saving model to ./checkpoints/{args.name}.pth')
        torch.save(model.state_dict(), f'./checkpoints/{args.name}.pth')

if __name__ == '__main__':
    parser = ArgumentParser(
                    prog='Training model for document segmentation',
                    description='Training')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of workers for dataloader')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial rate for training')
    parser.add_argument('--backbone', type=str, default='mobilenetv3', help='Backbone to use for model')
    parser.add_argument('--loss_weights', type=float, nargs='+', default=[1.,0.], help='Weights for loss function')
    parser.add_argument('--no-save', action='store_true', help='Do not save model')
    parser.add_argument('--name', type=str, default='model', help='Name of model')
    parser.add_argument('--load', type=str, default=None, help='Load model from checkpoint')
    parser.add_argument('--load-dataset', action='store_true', help='Load a dataset from online')
    args = parser.parse_args()

    main(args)