import numpy as np
import torch
from util import read_tenhou_game, construct_words
import glob, os
import json

class MahjongDataset(torch.utils.data.Dataset):

    def __init__(self,path, ratio=1, cache_dir=None):
        self.files = glob.glob(path + "/*")[::ratio]
        self.words = construct_words()
        self.games = []
        len_files = len(self.files)
        for n, f in enumerate(self.files):
            print(f, n, "/", len_files)
            json_filename = f.split("/")[-1] + ".json"
            if os.path.exists(cache_dir + "/" + json_filename):
                with open(cache_dir + "/" + json_filename) as f:
                    game = json.load(f)
            else:
                try:
                    game = read_tenhou_game(f)
                except IndexError:
                    print("Failure")
                    continue
                with open(cache_dir + "/" + json_filename, "w") as f:
                    json.dump(game, f)
            for i in game:
                for j in i:
                    self.games.append(j)
        self.len_games = len(self.games)
        self.cut_off_points = [[n for n, y in enumerate(x) if len(y)==2 ] for x in self.games]
        self.games = [[self.words.index(y) for y in x] for x in self.games]
        
    def __len__(self):
        return self.len_games

    def __getitem__(self, ndx):
        sample = self.games[ndx]
        cut_off_points = self.cut_off_points[ndx]
        cut_off_point = np.random.choice(cut_off_points)
        game, output = sample[:cut_off_point], sample[cut_off_point]
        return torch.Tensor.long(game), torch.Tensor.long(output)

    def print_game(self, game):
        return [self.words[x] for x in game]

if __name__=="__main__":

    md = MahjongDataset("discard_datasets/2009", cache_dir="tmp")
    print(len(md))
