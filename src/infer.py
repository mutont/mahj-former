import torch
from torch.nn import functional as F
from transformer_model.model.transformer import Transformer
import sys, json

if __name__=="__main__":
    import argparse
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-c", "--config", help="Configuration file. JSON format", default="default_config.json")
    argParser.add_argument("-i", "--input_file", help="Input file", required=True)

    args = argParser.parse_args()
    conf_path = args.config
    input_file = args.input_file
    with open(conf_path) as f:
        config = json.load(f)
    
    
    # Sample hyperparameters
    batch_size =config["batch_size"]
    num_epochs = config["num_epoch"]
    learning_rate = config["learning_rate"]
    dataset_path = config["dataset_path"]
    cache_path = config["cache_path"]
    d_model = config["d_model"]
    max_len =  config["max_len"]
    ffn_hidden =  config["ffn_hidden"]
    n_head =  config["n_head"]
    n_layers =  config["n_layers"]
    drop_prob = config["drop_prob"]
    model_path = config["model_path"]

    init_lr = config["init_lr"]
    factor = config["factor"]
    patience = config["patience"]
    warmup = config["warmup"]

    from dataset import MahjongDataset
    md = MahjongDataset(input_file, cache_dir=cache_path)
    w = md.words
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Transformer(
        src_pad_idx=md.word_count-1,
        trg_pad_idx=md.word_count-1,
        #trg_sos_idx=md.word_count + 1,
        d_model=d_model,
        enc_voc_size=md.word_count,
        dec_voc_size=md.word_count,
        max_len=max_len,
        ffn_hidden=ffn_hidden,
        n_head=n_head,
        n_layers=n_layers,
        drop_prob=drop_prob,
        device=device).to(device)
    model.load_state_dict(torch.load(model_path))
    
    for i in md:
        with torch.no_grad():
            npt, trg = i
            npt = npt.unsqueeze(0).to(device)
            
            opt = model(npt, npt[:,:-1])
            input_txt =  [w[x] for x in npt[0,:] if w[x] != "NONE"]
            print(" ".join(input_txt))
            print("Target: ", w[trg[-1]], " Model Pred:", w[
                torch.argmax(opt[:,-1,:],dim=1)])
