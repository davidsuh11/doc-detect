"""
Defines the training loop for the model.

Author: David Suh
Website: david-suh.pages.dev
Email: suhdavid11 (at) gmail (dot) com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from rich import print
from rich.progress import Progress

def evaluate_val(model, val_loader, criterion, device):
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for x,y in val_loader:
            x,y = x.to(device), y.to(device)
            output = model(x)
            val_loss += criterion(output['out'], y).item()
        val_loss /= len(val_loader)
    return val_loss


def train_one_epoch(model, optimizer, criterion, data_loader, device, 
                    epoch, prog, task, loss_weights = [1.,0.], scheduler=None):
    model.train()

    for t, (x,y) in enumerate(data_loader):
        optimizer.zero_grad()

        x,y = x.to(device), y.to(device)
        output = model(x)
        loss_main = criterion(output['out'], y)
        loss_aux = output['aux'].mean()
        loss = loss_main * loss_weights[0] + loss_aux * loss_weights[1]

        prog.update(task, advance=1, 
                    description=f"[red]Training... Epoch {epoch+1} Loss: {loss.item():.4f}")

        loss.backward()
        optimizer.step()
    
    if scheduler is not None:
        scheduler.step()

# define main training loop
def train(model, optimizer, criterion, data_loader, val_loader, device, 
          epochs=10, loss_weights = [1.,0.], scheduler=None):
    model = model.to(device)

    with Progress() as prog:
        task = prog.add_task("[red]Training...", total=len(data_loader))
        for epoch in range(epochs):
            prog.update(task, description=f"[red]Training... Epoch {epoch+1}/{epochs}")
            train_one_epoch(model, optimizer, criterion, data_loader,
                            device, epoch, prog, task,
                            loss_weights=loss_weights, scheduler=scheduler)
            prog.reset(task)
            val_loss = evaluate_val(model, val_loader, criterion, device)
            print(f"[green]Validation loss: {val_loss:.4f}")

        prog.remove_task(task)
    
    print("[green]Training complete!")