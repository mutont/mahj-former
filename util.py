import numpy as np
import scipy
from collections import Counter

from mahjong.hand_calculating.hand import HandCalculator
from mahjong.tile import TilesConverter
from mahjong.shanten import Shanten
from mahjong.hand_calculating.hand_config import HandConfig
from mahjong.meld import Meld

def list_difference(list1, list2, small_portion=False):
    # Use Counter to get the occurrences of elements in both lists
    counter1 = Counter(list1)
    counter2 = Counter(list2)

    # Calculate the difference by subtracting the counts
    #result_counter = max(counter1,counter2) - min(counter1,counter2)
    if small_portion:
        result_counter = counter1 - counter2 if len(counter1) < len(counter2) else counter2 - counter1
    else:
        result_counter = counter1 - counter2 if len(counter1) > len(counter2) else counter2 - counter1
    
    # Convert the result back to a list
    result = list(result_counter.elements())
    return result


TILES = ["m1","m2","m3","m4","m5","m6","m7","m8","m9","p1","p2","p3","p4","p5","p6","p7","p8","p9","s1","s2","s3","s4","s5","s6","s7","s8","s9","w1","w2","w3","w4", "d1","d2","d3"]

def construct_words(v=1):
    suits = ["s", "p", "m"]
    numbers = [y+str(x) for y in suits for x in range(1,10)]

    dragons = ["d" + str(x) for x in range(1,4)] #HAKU HATSU CHUN
    winds = ["w" + str(x) for x in range(1,5)] #EAST SOUTH WEST NORTH

    PLAYERS = ["P0", "P1", "P2", "P3"]

    CHIS = [z+"".join([str(x+y)for x in range(0,3)]) for y in range(1,8) for z in suits]

    CALLS_WO_CHI = [ "DISCARD", "PON", "KAN", "RIICHI", "TSUMO", "RON"]
    CALLS = [ "CHI", "DISCARD", "PON", "KAN", "RIICHI", "TSUMO", "RON"]

    if v== 1:
        words = []
        tiles = numbers + (dragons) + (winds)
        words += ([x+"_"+y for x in CALLS_WO_CHI for y in tiles])
        words += (["CHI_"+x for x in CHIS])
        words = [x+"_"+y for x in PLAYERS for y in words]
        words += (tiles)
        words += (["WIND_"+str(x) for x in range(4)]) #range(1,5)])
        words += (["DORA_"+x for x in tiles])
        words.append("DRAW")
    else:
        words = []
        tiles = suits + (dragons) + (winds)
        words += tiles
        words += (PLAYERS)
        words += (CHIS)
        words += (CALLS)
        
    return words, tiles

def shuffle_tiles(tiles):
    new_tiles = tiles.copy()
    np.random.shuffle( new_tiles )
    return new_tiles

########### WE ARE JUGGLING WITH 4 DIFFERENT TYPES OF MAHJONG HAND REPRESENTATIONS THESE ARE
########### LST E.G ['m4', 'd2', 's3', ... ], THE EASIEST TO HANDLE IMO
########### STR E.G 'm334s444d222...', THE MOST READABLE IMO
########### VEC E.G [0,0,3,0,0,4,...] WHERE INDICES REFER TO CERTAIN TILE TYPE AND ENTRIES TO HOW MANY OF THESE TILES ARE IN HAND, THE INDEXES CORRESPOND TO THE 'TILES' ARRAY ABOVE. USED BY TENHOU DATASET
########### DICT E.G {'sou':'22333455', 'pin':'234',...}, USED BY THE MAHJONG- PYTHON LIBRARY

def lst_to_str(lst):
    txt = ""
    suit = None
    for i in lst:
        s,n = i[0],int(i[1])
        if s == suit:
            txt += str(n)
        else:
            suit = s
            txt += str(s)+str(n)

    return txt
            
def str_to_lst(txt):
    lst = []
    suit = None
    for i in txt:
        if not i.isnumeric():
            suit = i
        else:
            lst.append(suit+str(int(i)))
    return lst

def vec_to_lst(vec):
    
    result = []
    for i in range(len(vec)):
        for j in range(vec[i]):
            result.append(TILES[i])
    return result

def lst_to_vec(lst):
    vec = [0 for i in tiles]
    
    for i in lst:
        j = TILES.index(i)
        vec[j] += 1
    return vec

def vec_to_str(vec):
    return lst_to_str(vec_to_lst(vec))

def str_to_vec(txt):
    return str_to_lst(lst_to_vec(txt))

def lst_to_dct(lst):
    dct = {"pin":"","sou":"","man":"","honors":""}
    for i in lst:
        if i[0] == "p":
            dct["pin"] += i[1]
        
        elif i[0] == "s":
            dct["sou"] += i[1]
            
        elif i[0] == "m":
            dct["man"] += i[1]

        elif i[0] == "w":
            dct["honors"] += i[1] #HONORS CONTAIN BOTH WINDS AND DRAGONS. 1-4 ARE WINDS, 5-7 ARE DRAGONS

        elif i[0] == "d":
            dct["honors"] += str( int(i[1]) + 4 )

        else:
            raise Exception
    return dct

def dct_to_lst(dct):
    lst = []
    for p in dct["pin"]:
        lst.append("p"+p)
    for s in dct["sou"]:
        lst.append("s"+p)
    for m in dct["man"]:
        lst.append("m"+p)
    for h in dct["honors"]:
        if int(h) < 5:
            lst.append("w"+h)
        else:
            lst.append("d"+str(int(h)-4))
    return lst
    
def readable(hand):
    ### MOSTLY FOR DEBUGGING AND READABILITY REASONS WE CONVERT THE TENHOU DATASET ENTRIES INTO THIS REPRESENTATION AND ONLY THEN TO THEIR FINAL REPRESENTATION
    result =  {}
    #winds = ["east", "south", "west", "north"]
    winds = ["w" + str(x) for x in range(1,5)]
    result["wind"] = hand[0]#winds[hand[0]]
    result["round"] = hand[1]#winds[hand[1]]
    result["player"] = hand[2]
    left =   str((hand[2] - 1) % 4)
    right =  str((hand[2] + 1) % 4)
    center = str((hand[2] + 2) % 4)
    result["honba"] = hand[3]
    result["riichis"] = hand[4]
    result["wall"] = hand[5]
    result["scores"] = [x for x in hand[6:10]]
    result["scores"] = result["scores"][-result["player"]:] + (result["scores"][:-result["player"]])
    result[str(hand[2])+"_riichi"] = False #bool(hand[11])
    result[right+"_riichi"] = bool(hand[11])
    result[center+"_riichi"] = bool(hand[12])
    result[left+"_riichi"] = bool(hand[13])
    result["round_no"] = hand[32]
    result["inv_tiles"] = hand[33]
    result["doras"] = vec_to_str([x for x in hand[34:68]])
    result[str(hand[2])+"_hand"] = vec_to_str([x for x in hand[68:102]])
    result[str(hand[2])+"_calls"] = vec_to_str([x for x in hand[102:136]])
    result[right+"_calls"] = vec_to_str([x for x in hand[136:170]])
    result[center+"_calls"] = vec_to_str([x for x in hand[170:204]])
    result[left+"_calls"] = vec_to_str([x for x in hand[204:238]])
    result[str(hand[2])+"_discards"] = vec_to_str([x for x in hand[238:272]])
    result[right+"_discards"] = vec_to_str([x for x in hand[272:306]])
    result[center+"_discards"] = vec_to_str([x for x in hand[306:340]])
    result[left+"_discards"] = vec_to_str([x for x in hand[340:374]])
    result[str(hand[2])+"_discard"] = TILES[hand[-1]]
    result["player"] = str(result["player"])
    return result

def read_tenhou_game(path):

    data = np.load(path)
    data = np.array(scipy.sparse.csr_matrix((data['data'], data['indices'], data['indptr']), shape=data['shape']).toarray())
    return tenhou_to_custom(data)
    

def count_shanten(dct):
    shanten = Shanten()
    tiles = TilesConverter.string_to_34_array(sou=dct["sou"], pin=dct["pin"], man=dct["man"], honors=dct["honors"])
    #print(tiles)
    #exit()
    return shanten.calculate_shanten(tiles)

def count_price(dct, wintile):
    ### WORK IN PROGRESS. CURRENTLY ALL SCORES ARE 0
    calculator = HandCalculator()

    tiles = TilesConverter.string_to_136_array(man='123', pin='123678', sou='55', honors='222')
    win_tile = TilesConverter.string_to_136_array(pin='6')[0]




    result = calculator.estimate_hand_value(tiles, win_tile)

    if str(result) == "no_yaku" or str(result) == "hand_not_correct":
        return 0
    return result.cost['main']

def get_wait(lst):
    dct = lst_to_dct(lst)
    wait = []
    scr = []
    for i in TILES:
        
        tmp = lst + ([i])
        if len([x for x in tmp if x == i]) > 4: # ITS POSSIBLE THAT PLAYER IS HOLDING 4 SAME TILES WHICH WILL CAUSE AN ERROR HERE
            continue
        tmp = lst_to_dct(tmp)
        
        if count_shanten(tmp) == Shanten.AGARI_STATE:
            wait.append(i)
            scr.append( count_price( tmp, lst_to_dct([i]) ) )
        
    return wait, scr

class Game:
    ### CLASS FOR MAKING ADDING MOVES TO MULTIPLE PLAYERS EASIER
    def __init__(self):
        self.game = [[],[],[],[]]

    def add_winds(self, rnd, seat):
        
        for i in range(len(self.game)):
            self.game[i].append( "WIND_" + str(rnd) )
            self.game[i].append( "WIND_" + str((seat + i) % 4) )

    def add(self, x, turn=None, ndx=None):
        for i in range(len(self.game)):

            if turn in [0,1,2,3] and i != turn:
                continue
            tmp = x
            if "P" in x: #IF move is relative for player
                pos = str( (int( tmp.split("_")[0][1] ) - i) % 4 )
                tmp = "P"+"_".join([pos] + (tmp.split("_")[1:]))
            if ndx:
                self.game[i].insert(ndx, tmp)
            else:
                self.game[i].append(tmp)

    #def insert(self, ndx, x, turn=None):
        

    def __repr__(self):
        txt = ""
        for n, i in enumerate(self.game):
            txt += "\n" + str(n) + ": "
            for j in i:
                txt += str(j) + " "
        return txt

    def list(self):
        return self.game

def process_calls_and_discs(new_game, hands, calls, discs, turn, doras, riichi=False, call_tile=None, auto_discard_was_called_from=None):
    if len(set(calls)) == 1:
        
        if len(calls) == 3:
            new_game.add("P"+str(turn)+"_PON_"+calls[0])
            hands[int(turn)] = list_difference(hands[int(turn)], calls)
        elif len(calls) == 4 or len(calls) == 1:

            new_game.add("P"+str(turn)+"_KAN_"+calls[0])
            #new_dora = list_difference(str_to_lst(rg["doras"]), str_to_lst(rg2["doras"]))[0]
            new_game.add("DORA_" + doras[0])
            hands[int(turn)] = list_difference(hands[int(turn)], calls)
        if auto_discard_was_called_from is not None:
            
            new_game.add(call_tile[0],  turn=auto_discard_was_called_from,ndx=-1)
            new_game.add("P"+str(auto_discard_was_called_from)+"_DISCARD_"+call_tile[0], ndx=-1)
    elif len(set(calls)) == 3:
        print(calls)
        new_game.add("P"+str(turn)+"_CHI_"+lst_to_str(calls))
        if auto_discard_was_called_from is not None:
            new_game.add(call_tile[0],  turn=auto_discard_was_called_from, ndx=-1)
            new_game.add("P"+str(auto_discard_was_called_from)+"_DISCARD_"+ call_tile[0], ndx=-1)
        hands[int(turn)] = list_difference(hands[int(turn)], calls)

    if len(discs) == 1:
        if riichi:
            new_game.add("P"+str(turn)+"_RIICHI_"+discs[0])
            
        else:
            new_game.add("P"+str(turn)+"_DISCARD_"+discs[0])

    
def tenhou_to_custom(game):
    words, tiles = construct_words()
    tl_game = []
    tmp_games = []
    result = []
    ###FOR EASIER READABILITY WE DO TRANSLATION IN MULTIPLE RUNS

    ###SEPARATE SESSIONS INTO SEPARATE ROUNDS
    for n, i in enumerate(game):
        if not n:
            tmp = []
        rg = readable(i)
        if n and rg["wall"] == 69:
            tmp_games.append(tmp)
            tmp = [i]
        else:
            tmp.append(i)
    tmp_games.append(tmp)

    translated_games = []
    for game_no, tmp_game in enumerate([tmp_games[2]]):
        if len(tmp_game) <= 4:
            continue
        if game_no == len(tmp_games)-1:
            ### DUE TO THE NATURE OF THE DATASET, WE CAN'T FULLY TRANSLATE THE LAST GAME
            break

        ### TENHOU DATASET DOESNT HOLD THE HAND OF A DOUBLE RIICHI PLAYER SO WE WILL IGNORE THESE GAMES
        tmp = readable(tmp_game[0])
        if tmp["0_riichi"] or tmp["1_riichi"] or tmp["2_riichi"] or tmp["3_riichi"]:
            continue
        REMAINING_TILES = [x for _ in range(4) for x in TILES]
        hands = {}
        melds = []
        tenpais = {"0":{"wait":[], "scr":[]},"1":{"wait":[],"scr":[]},"2":{"wait":[],"scr":[]},"3":{"wait":[],"scr":[]}}
        used_tiles = []
        first_tile = [True, True, True, True]
        new_game = Game()
        for i in tmp_game:
            ### THIS RUN WE USE TO INITIALIZE HANDS
            rg = readable(i)
            if len(hands) < 4 and not rg["player"] in hands:
                hands[int(rg["player"])]= str_to_lst(rg[rg["player"]+"_hand"])
        for n, i in enumerate(tmp_game):
            ### FINALLY WE GO THROUGH ALL THE HANDS IN THE ROUND
            rg = readable(i)
            print(rg)
            hand = str_to_lst(rg[rg["player"] + "_hand"]) ### TENHOU DATASET HOLDS THE MOMENTARY 14 TILE HAND
            disc = rg[rg["player"] + "_discard"]
            hand = [x for n,x in enumerate(hand) if n != hand.index(disc)] ### WE ARE INTERESTED IN THE 13 TILE HAND
            wait, scr = get_wait(hand)
            
            ### WE HAVE TO KEEP TRACK OF POSSIBLE WAITS HERE BECAUSE TENHOU DATASET DOESNT EXPLICITLY TELL WHO WAS WAITIN WHAT AND WHO WON
            if len(wait):
                tenpais[rg["player"]]["wait"] = wait
                tenpais[rg["player"]]["scr"] = scr

            ### FIRST ROUND WE TAKE WINDS AND DORA
            if not n:
                new_game.add_winds(rg["wind"], rg["round"])
                new_game.add("DORA_"+ str_to_lst(rg["doras"])[0])
                for m, j in hands.items():
                    for k in j:
                        new_game.add(k, turn=m)
                ### FOR WHATEVER DAMN REASON, IF PLAYER CALLS THE VERY FIRST DISCARD, THE FIRST DISCARD CHOICE IS NOT RECORDED
                turn = str(rg["round"])
                calls = str_to_lst(rg[rg["player"]+"_calls"])
                if len(calls):
                    new_game.add("P"+str(turn)+"_DISCARD_"+calls[0]) # WE CANT REALLY NOW WHAT THE DISCARD WAS SO WE JUST PICK ONE
                    doras = []
                    discs = []
                    if len(doras) > 1:
                        doras = [str_to_lst(rg["doras"])[0]]
                    tmp = hands[int(turn)]
                    process_calls_and_discs(new_game, hands, calls, discs, turn, doras)
                    hands[int(turn)] = tmp
                    

            if n != len(tmp_game) - 1:
                ### TENHOU DATASET ONLY HAS A STATE OF THE TABLE AND A CORRESBONDING DISCARD. IT DOESN'T EXPLICITLY TELL WHAT CALLS AND AUTODISCARDS ANYBODY MAKES SO WE HAVE TO INFER THIS BY COMPARING TWO SEQUENTIAL STATES
                rg2 = readable(tmp_game[n+1])
                #print(rg2)
                if not first_tile[int(rg["player"])]:#len(hands[int(rg["player"])]) < 14:
                    
                    draw = list_difference( hands[int(rg["player"])], hand + ([disc]) )#[0]
                    
                    if len(draw) > 1: # IN CASE OF KANS YOU MIGHT HAVE MORE THAN ONE DRAW
                        for d in range(len(draw)):
                            
                            if draw[d] in str_to_lst( rg[rg["player"] + "_calls" ]):
                                new_game.add(draw.pop(d), turn=int(rg["player"]), ndx=-2)
                                break
                        else:
                            new_game.add(draw.pop(0), turn=int(rg["player"]), ndx=-2)
                        
                        #else:
                    if len(draw):
                        draw = draw[0]
                    
                        ### DRAW OF THE CURRENT PLAYER
                        new_game.add(draw, turn=int(rg["player"]))
                first_tile[int(rg["player"])] = False
                hands[int(rg["player"])] = hand
                call_tile = None
                auto_discard_was_called_from = None
                tile_was_called = False
                for j in range(5): 
                    turn =  str((int(rg["player"]) + j) % 4 )
    
                    ### THIS SECTION ADDS CALLS AND AUTODISCARDS
                    discs = list_difference(str_to_lst(rg[turn+"_discards"]), str_to_lst(rg2[turn+"_discards"]))
                    ### DUE TO CERTAIN EDGE CASES, THE FIRST ITER ADDS THE CURRENT PLAYERS DISCS AND LAST ADDS THEIR CALLS
                    if not j and not len(discs):
                        tile_was_called = True
                        discs = [rg[turn+"_discard"]]
                    if not j:
                        calls = []
                    else:
                        calls = list_difference(str_to_lst(rg[turn+"_calls"]), str_to_lst(rg2[turn+"_calls"]))
                    if j == 4:
                        discs = []
                    doras = list_difference(str_to_lst(rg["doras"]), str_to_lst(rg2["doras"]))

                    ### MORE EDGE CASES REGARDING RIICHI'D PLAYERS AND THEIR DISCARDS BEING CALLED
                    if rg[turn+"_riichi"] and not tile_was_called:
                        if not len(discs) and not len(calls):
                        
                            auto_discard_was_called_from = int(turn)
                            continue
                        elif not len(discs) and len(calls):
                            process_calls_and_discs(new_game, hands, calls, discs, turn, doras, riichi=rg[turn + "_riichi"] != rg2[turn + "_riichi"]) 
                            auto_discard_was_called_from = int(turn)
                            new_game.add(calls[0], turn=int(turn), ndx=-2)
                            continue
                        else:
                            new_game.add(discs[0], turn=int(turn))
                    if auto_discard_was_called_from is not None and len(calls): #SOMEBODY CALLED RIICHI'D DISCARD
                        call_tile = list_difference( list_difference(hands[int(rg2["player"])],str_to_lst( rg2[turn + "_hand"] )), calls)
                        #print(call_tile)
                    process_calls_and_discs(new_game, hands, calls, discs, turn, doras, riichi=rg[turn + "_riichi"] != rg2[turn + "_riichi"], call_tile=call_tile, auto_discard_was_called_from=auto_discard_was_called_from) 
                
            else:
                ### END OF THE GAME. TENHOU DATASET DOESN'T ACTUALLY TELL HOW THE GAME ENDS. FIRST WE CHECK IF THE LAST DISCARD BELONGS TO ANY OF THE WAITS MENTIONED BEFOREHAND WHICH WOULD COUNT AS RON
                new_game.add("P"+str(rg["player"])+"_DISCARD_"+disc) #rg[str(turn)+"_discard"])
                for n, i in tenpais.items():
                    if len(i["wait"]) and disc in i["wait"]:
                        #new_game.add("P"+str(rg["player"])+"_DISCARD_"+disc) #rg[str(turn)+"_discard"])
                        new_game.add("P"+str(n)+"_RON_"+disc)
                        break
                else:
                    ### IF NO RON, WE CHECK FOR TSUMO. IF THERE ARE STILL TILES LEFT IN THE WALL, IT MEANS THAT THERE WAS A TSUMO. DATASET DOESNT TELL US WHAT TILE WAS USED TO WIN SO WE JUST TAKE A RANDOM TILE FROM THE WAITS. WE ARE LAZY FOR NOW AND WE PICK ONE OF THE WAIT TILES. THIS MEANS THAT PLAYER COULD WIN BY IMPOSSIBLE 5TH TILE. IN FUTURE WE SHOULD KEEP TRACK OF ALL THE TILES AND MAKE SURE THIS DOESNT HAPPEN.
                    rg2 = readable(tmp_games[game_no+1][0])
                    if rg["wall"]:
                        winner = (int(rg["player"]) + 1) % 4
                        score = rg2["scores"][winner]-rg["scores"][winner]
                        new_game.add(tenpais[str(winner)]["wait"][0], turn=winner)
                        new_game.add("P"+str(winner)+"_TSUMO_"+tenpais[str(winner)]["wait"][0])
                    else:
                        ### HAITEI TSUMO IS ALSO POSSIBLE HERE BUT DATASET SIMPLY DOESNT INFORM ABOUT THIS. ITS JUST A DRAW.
                        new_game.add("DRAW")
                        
        #print(new_game)
        #exit()
        result.append(new_game.list())
    ### LAST CHECK THAT ALL ACTIONS ARE LEGAL
    for i in result:
        for j in i:
            for k in j:                
                assert k in words, "{} not in words".format(k)
    return result

if __name__ == "__main__":
    read_tenhou_game("discard_datasets/2009/2009073120gm-00a9-0000-08b06765.npz")
