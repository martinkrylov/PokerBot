# PokerBot
Final project for CS394R: Applying Reinforcement Lerning to Poker. This project utilizes the [neuron_poker](https://github.com/dickreuter/neuron_poker) environment by [dickreuter](https://github.com/dickreuter).
## Code Explanation
- [poker](https://github.com/martinkrylov/PokerBot/tree/main/poker) contains the all of our code.
  - [agents](https://github.com/martinkrylov/PokerBot/tree/main/poker/agents) contains implementation of all agents.
    - [agent_rl](https://github.com/martinkrylov/PokerBot/blob/main/poker/agents/agent_rl.py): deep q-learning agent.
    - [agent_consider_equity](https://github.com/martinkrylov/PokerBot/blob/main/poker/agents/agent_consider_equity.py): equity-based agent.
    - [agent_random](https://github.com/martinkrylov/PokerBot/blob/main/poker/agents/agent_random.py): random player.
  - [gym_env](https://github.com/martinkrylov/PokerBot/tree/main/poker/gym_env) contains the Texas Hold’em unlimited openai gym environment
  - [main.py](https://github.com/martinkrylov/PokerBot/blob/main/poker/main.py) is the main run file of the project. It contains code that sets up the poker games.
 
## Run the Code Yourself
### Setup the Environment
- Navigate to ```PokerBot/poker``` directory
- Install poetry ```curl -sSL https://install.python-poetry.org | python3 -``` and add it to path.
- ```poetry env use python3.11```
- ```poetry shell```
- ```poetry install --no-root```
### Run the code
```
Usage:
  main.py selfplay random [options]
  main.py selfplay keypress [options]
  main.py selfplay consider_equity [options]
  main.py selfplay equity_improvement --improvement_rounds=<> [options]
  main.py selfplay dqn_train [options]
  main.py selfplay dqn_play [options]
  main.py selfplay neo_train [options]
  main.py selfplay neo_play [options]
  main.py learn_table_scraping [options]

options:
  -h --help                 Show this screen.
  -r --render               render screen
  -c --use_cpp_montecarlo   use cpp implementation of equity calculator. Requires cpp compiler but is 500x faster
  -f --funds_plot           Plot funds at end of episode
  --log                     log file
  --name=<>                 Name of the saved model
  --screenloglevel=<>       log level on screen
  --episodes=<>             number of episodes to play
  --stack=<>                starting stack for each player [default: 500].

```

  
