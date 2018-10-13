# Q-Learning with Pacman
Grégoire Roussel
8/10/2018
This TP develops the Markov decision approach with Q-Learning and approximate Q-Learning


## Results with basic QLearning
- mediumGrid, with 15k training iterations (10 min)
```
Average Score: 271.6
Scores:        529.0, 527.0, 529.0, -504.0, -510.0, 529.0, 529.0, 529.0, 527.0, 529.0, 529.0, 529.0, 529.0, -490.0, -497.0, 527.0, 529.0, 529.0, 529.0, -496.0
Win Rate:      15/20 (0.75)
Record:        Win, Win, Win, Loss, Loss, Win, Win, Win, Win, Win, Win, Win, Win, Loss, Loss, Win, Win, Win, Win, Loss
```

- classicGrid, with 50k iterations (around 1h30)
didn't finish, after taking 30Gb+ of RAM...

## Results with approximate QAgent

- mediumClassic, 50 training iterations & SimpleExtractor
Training time was around 10sec
```
Average Score: 1088.1
Scores:        1308.0, 1337.0, 1344.0, 1308.0, 1331.0, 1289.0, 1333.0, 241.0, 56.0, 1334.0
Win Rate:      8/10 (0.80)
Record:        Win, Win, Win, Win, Win, Win, Win, Loss, Loss,Win
```
So this technique is *immensely* more efficient !

- originalClassic, 200 training iterations & SimpleExtractor
Training time around 7min
That is quite impressive to see :)
```
Average Score: 2265.4
Scores:        2464.0, 2480.0, 2436.0, 2404.0, 1543.0
Win Rate:      4/5 (0.80)
Record:        Win, Win, Win, Win, Loss
```
Yes, he lost once, being trapped between two ghosts. That is a mistake rather hard to avoid, as the bad decision happens 10+ moves before the negative reward (aka death).

#### Remark:
An analysis of the feature extractors confirms that, for now, the feature vector doesn't take into account the `power-mode` of Pacman (during which he could eat the ghosts, win more points and loose less time). 

That could be implemented as an additionnal feature `eatGhost`, that is 1 if Pac-Man is in `power-mode` and there is a ghost in the next cell.