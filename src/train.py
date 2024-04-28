import torch
from torch.nn import functional as F
from transformer_model.model.transformer import Transformer

def src_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def train(model, train_loader, criterion, optimizer, num_epoch, w, test_loader, scheduler):
    model.train()
    mask = src_mask(193)
    perf = torch.inf
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for input_ids, labels in train_loader:
            labels = labels.to(device)
            input_ids = input_ids.to(device)
            
            outputs = model(input_ids, input_ids[:,:-1])
            loss = criterion(outputs[:,-1,:], labels[:,-1])
            epoch_loss += loss.item()
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}')

        labels.to("cpu")
        outputs.to("cpu")
        input_ids.to("cpu")
        del labels, outputs, input_ids
        torch.cuda.empty_cache()

        perf_tmp = test(model, test_loader, criterion)
        scheduler.step(perf_tmp)
        if perf_tmp < perf:
            perf = perf_tmp
            torch.save(model.state_dict(), "model.pt")
    return model

def test(model, test_loader, criterion = None, verbose = False):
    model.eval()
    scr = 0.0
    full = 0.0
    
    mask = src_mask(193)
    full_loss = 0.0
    with torch.no_grad():
        for input_ids, labels in test_loader:
            labels = labels[:,-1].to(device)
            input_ids = input_ids.to(device)
            outputs = model(input_ids, input_ids[:,:-1])#,src_mask=mask)
            if criterion :
                loss = criterion(outputs[:,-1,:], labels)
                full_loss += loss.item()
            scr += sum(torch.argmax(outputs[:,-1,:],dim=1) == labels).cpu().item()
            full += labels.shape[0]
    if verbose:
        print(scr, "/", full)
    return full_loss / len(test_loader)
    
if __name__=="__main__":

    # Sample hyperparameters
    batch_size = 64
    num_epochs = 20
    learning_rate = 1e-3
    
    # Create model, loss function and optimizer
    
    
    from dataset import MahjongDataset
    md = MahjongDataset("tmp2", cache_dir="tmp2")
    w = md.words
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Transformer(
        src_pad_idx=md.word_count-1,
        trg_pad_idx=md.word_count-1,
        #trg_sos_idx=md.word_count + 1,
        d_model=512,
        enc_voc_size=md.word_count,
        dec_voc_size=md.word_count,
        max_len=197,
        ffn_hidden=512,
        n_head=8,
        n_layers=6,
        drop_prob=0.1,
        device=device).to(device)
    torch.save(model.state_dict(), "model.pt")
        
    criterion = torch.nn.CrossEntropyLoss(ignore_index=md.word_count-1)
    
    init_lr = 1e-5
    factor = 0.9
    patience = 10
    warmup = 100
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                     verbose=True,
                                                     factor=factor,
                                                     patience=patience)
    
    
    # Create sample dataloader
    
    
    # Define the size of the validation set (e.g., 20%)
    val_size = 0.2
    
    # Calculate the sizes of training and validation sets
    num_samples = len(md)
    num_val = int(num_samples * val_size)
    num_train = num_samples - num_val
    
    # Use random_split to split the dataset into train and validation sets
    train_dataset, val_dataset = torch.utils.data.random_split(md, [num_train, num_val])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader =torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    test(model, test_loader, verbose=True)
    model = train(model, train_loader, criterion, optimizer, num_epochs, w, test_loader, scheduler)
    test(model, test_loader, verbose=True)
    
