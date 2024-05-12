Mahj-former

A simple hobby project about a transformer based riichi mahjong bot. For the uninitiated you can check the rules and resources of Mahjong at https://riichi.wiki/Rules_overview

1st Phase - Supervision

Using dataset found here: https://www.kaggle.com/datasets/trongdt/japanese-mahjong-board-states (35GB), we translate the dataset from sparse numpy states of the board - discard pairs into action tokens which are then used to train the model e.g. m6 P0_DISCARD_m6 P1_CHI_m567 P1_DISCARD_d1 ...

Transformer model used as initial starting point: https://github.com/hyunwoongko/transformer

After initial tests model is able to receive ~60% accuracy using around 70k training samples. Evidence indicates that more samples would increase this accuracy, however further inspection shows that actual learning doesn't take place but the model simply tries to make correct looking moves. When getting predictions wrong moves are often illegal and impossible. Therefore, better training procedure is required.

### Next steps:
- [ ] Complete dataset conversion
- [ ] Player hand states as the Transformer decoder input
- [ ] DQN training procedure 
