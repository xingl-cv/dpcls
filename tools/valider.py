from tqdm import tqdm

def validate(valid_loader, model, criterion, device):
    '''
    Function for the validation step of the training loop
    '''
   
    model.eval()
    running_loss = 0
    
    for x, y_true in tqdm(valid_loader, ascii=False, desc='Validating'):
    
        x = x.to(device)
        y_true = y_true.to(device)

        # Forward pass and record loss
        y_pred = model(x) 
        loss = criterion(y_pred, y_true) 
        running_loss += loss.item() * x.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)
        
    return model, epoch_loss