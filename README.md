Mahj-former

A simple hobby project about a transformer based riichi mahjong bot. For the uninitiated you can check the rules and resources of Mahjong at https://riichi.wiki/Rules_overview

1st Phase - Supervision

Using dataset found here: https://www.kaggle.com/datasets/trongdt/japanese-mahjong-board-states (35GB), we translate the dataset from sparse numpy states of the board - discard pairs into action tokens which are then used to train the model.
