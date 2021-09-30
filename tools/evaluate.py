from tqdm import tqdm
import torch
import torch.nn.functional as F

def get_accuracy(model, data_loader, device, type='valid'):
    '''
    Function for computing the accuracy of the predictions over the entire data_loader
    '''
    
    correct_pred = 0 
    n = 0
    
    with torch.no_grad():
        model.eval()
        for x, y_true in tqdm(data_loader, ascii=False, desc=f'Calculate {type} Accuracy'):

            x = x.to(device)
            y_true = y_true.to(device)

            y_pred = model(x)
            y_prob = F.softmax(y_pred, dim=1)
            _, predicted_labels = torch.max(y_prob, 1)

            n += y_true.size(0)
            correct_pred += (predicted_labels == y_true).sum()

    return correct_pred.float() / n