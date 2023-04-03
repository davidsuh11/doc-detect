from .build import build_model
from .train import train

def build(backbone='mobilenetv3', pretrained=True):
    return build_model(backbone, pretrained)

def train_model(model, optimizer, criterion, data_loader,val_loader, device, 
                epochs=10, loss_weights = [1.,0.], scheduler=None):
    train(model, optimizer, criterion, data_loader, val_loader, device, 
          epochs, loss_weights=loss_weights, scheduler=scheduler)