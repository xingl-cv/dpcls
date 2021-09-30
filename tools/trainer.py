from tqdm import tqdm

def trainerize(train_loader, model, criterion, optimizer, device):
    '''
    Function for the training step of the training loop
    '''

    model.train()
    running_loss = 0
    
    for x, y_true in tqdm(train_loader, ascii=False, desc='Trainning'):

        optimizer.zero_grad()
        
        x = x.to(device)
        y_true = y_true.to(device)
    
        # Forward pass
        y_pred = model(x) 
        loss = criterion(y_pred, y_true) 
        running_loss += loss.item() * x.size(0)

        # Backward pass
        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss
