import numpy as np
import torch
from util import read_tenhou_game, construct_words
import glob, os
import json

class MahjongDataset(torch.utils.data.Dataset):

    def __init__(self,path, ratio=1, cache_dir=None):
        if os.path.isdir(path):
            self.files = glob.glob(path + "/*")[::ratio]
        else:
            self.files = [path]
        self.words = construct_words()[0] + (["NONE"])
        self.word_count = len(self.words)
        self.games = []
        len_files = len(self.files)
        for n, f in enumerate(self.files):
            print(f, n, "/", len_files)
            json_filename = f
            if not f.endswith(".json"):
                json_filename = cache_dir + "/" + f.split("/")[-1] + ".json"
            if os.path.exists(json_filename):
                
                with open(json_filename) as f:
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
        #self.cut_off_points = [[n for n, y in enumerate(x) if len(y)==2 and n > 18 ] for x in self.games]
        self.cut_off_points = [[n for n, y in enumerate(x) if "P0" in y and n > 18 ] for x in self.games]
        self.games = [[self.words.index(y) for y in x] for x in self.games]
        tmp_a = []
        tmp_b = []
        for i,j in zip(self.cut_off_points, self.games):
            if not len(i):
                continue
            tmp_a.append(i)
            tmp_b.append(j)
        self.cut_off_points = tmp_a
        self.games = tmp_b
        self.len_games = len(self.games)
        self.max_len = max([len(x) for x in self.games])
        
    def __len__(self):
        return self.len_games

    def __getitem__(self, ndx):
        sample = self.games[ndx]
        cut_off_points = self.cut_off_points[ndx]
        #cut_off_point = np.random.choice(cut_off_points)
        cut_off_point = cut_off_points[-1]
        game = torch.LongTensor(sample[:cut_off_point])
        #output = torch.LongTensor([sample[cut_off_point]])
        output = torch.LongTensor(sample[1:cut_off_point+1])
        game = torch.nn.functional.pad(game, (self.max_len - len(game),0), "constant",self.word_count-1)
        output = torch.nn.functional.pad(output, (self.max_len - len(output),0), "constant",self.word_count-1)
        #print(output, game)
        return game, output

    def print_game(self, game):
        return [self.words[x] for x in game]

if __name__=="__main__":

    md = MahjongDataset("tmp", cache_dir="tmp")
    print(len(md))
    x,y = md[9]
    print(x)
    print(x.shape, y)
