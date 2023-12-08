Here is a possible README file for your repository in markdown format:

# IPL-Simulator

This repository contains the code and data for the paper "DynaSim: A Ball-by-Ball Simulation of the Dynamics of an IPL T20 Game" by Aravind Subramanya Mysore, Vishal Bharadwaj, Rithik R. Mali and Bhaskarjyoti Das.

The paper presents a novel approach to simulate the ball-by-ball progression of a cricket match, specifically the Indian Premier League (IPL) T20 format. The approach uses deep neural networks to predict the outcome of each ball, considering the current state of the game and the historical and contextual features of the players and teams involved. The paper also evaluates the realism and accuracy of the simulations by comparing them with actual IPL matches.

## Requirements

To run the code, you will need the following:

- Python 3.11
- TensorFlow 2.15
- NumPy 1.26.2
- Pandas 1.5.3
- Matplotlib 3.8.2


## CustomPlay

The iplsim-dense folder contains another Jupyter notebook: CustomPlay.ipynb. This notebook allows you to simulate custom matches using the trained model of the dense simulator. You can specify the teams, the venue, the toss, and the batting order of each team, and the notebook will simulate the match ball-by-ball and display the scorecard and the summary of the match.

## Citation

If you use this code or data for your research, please cite our paper as follows:

```
Mysore, A.S., Bharadwaj, V., Mali, R.R., Das, B. (2023). DynaSim: A Ball-by-Ball Simulation of the Dynamics of an IPL T20 Game. In: Bansal, J.C., Sharma, H., Chakravorty, A. (eds) Congress on Smart Computing Technologies. CSCT 2022. Smart Innovation, Systems and Technologies, vol 351. Springer, Singapore. https://doi.org/10.1007/978-981-99-2468-4_29
```

## Result Encoding

The following table shows the result encoding used by the neural network to represent the 57 possible outcomes of each ball:

| Index | Outcome |
| ----- | ------- |
| 0 | Retired hurt |
| 1 | Dot ball |
| 2 | 1 run |
| 3 | 2 runs |
| 4 | 3 runs |
| 5 | 4 runs |
| 6 | 5 runs |
| 7 | 6 runs |
| 8 | Wkt bowled |
| 9 | Wkt caught |
| 10 | Wkt LBW |
| 11 | Wkt stump |
| 12 | Wkt hit wicket |
| 13 | Wkt obstructing the field |
| 14 | Wkt runout non striker 0 runs |
| 15 | Wkt run out striker 0 runs |
| 16 | Wkt run out non striker 1 run |
| 17 | Wkt run out striker 1 run |
| 18 | Wkt run out non striker 2 runs |
| 19 | Wkt run out striker 2 runs |
| 20 | Wkt run out non striker 3 runs |
| 21 | Wkt run out striker 3 runs |
| 22 | Wkt run out (no ball) non striker total 2 runs |
| 23 | Wkt run out (no ball) striker total 2 runs |
| 24 | Wkt run out non striker 1 run (LB/byes) |
| 25 | Wkt run out striker 1 run (LB/byes) |
| 26 | Wkt run out non striker 2 runs (LB/byes) |
| 27 | Wkt run out striker 2 runs (LB/byes) |
| 28 | Wkt run out non striker 3 runs (LB/byes) |
| 29 | Wkt run out striker 3 runs (LB/byes) |
| 30 | Wkt run out (wide) non striker total 1 run (1 run for wide) |
| 31 | Wkt run out (wide) striker total 1 run (1 run for wide) |
| 32 | Wkt run out (wide) non striker total 2 runs (1 for wide and 1 by running) |
| 33 | Wkt run out (wide) striker total 2 runs (1 for wide and 1 by running) |
| 34 | Wkt stump (wide) total 1 run |
| 35 | Wkt stump (no ball) total 1 run |
| 36 | LB/byes 1 run |
| 37 | LB/byes 2 runs |
| 38 | LB/byes 3 runs |
| 39 | LB/byes 4 runs |
| 40 | No ball + 0 runs |
| 41 | No ball + 1 run |
| 42 | No ball + 2 runs |
| 43 | No ball + 3 runs |
| 44 | No ball + 4 runs |
| 45 | No ball + 6 runs |
| 46 | No ball + 1 run (LB/byes) |
| 47 | No ball + 2 runs (LB/byes) |
| 48 | No ball + 3 runs (LB/byes) |
| 49 | No ball + 4 runs (LB/byes) |
| 50 | Wide + 0 runs |
| 51 | Wide + 1 run |
| 52 | Wide + 2 runs |
| 53 | Wide + 3 runs |
| 54 | Wide + 4 runs |
| 55 | Wkt runout (no ball) non striker total 1 run |
| 56 | Wkt runout (no ball) striker total 1 run |
