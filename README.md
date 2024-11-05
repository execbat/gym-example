# Training RL policy to take actions in simulated Stock Market environment

This repository provides the scenario of how to solve the problem of automated trading. Stock market environment built on engine of OpenAI's Gymnasium environment which emulates behavior of real stock marker environment. Integration with Sample Factory implemented for efficient training of reinforcement learning (RL) policy based on asynchronoius proxomal policy optimisation (APPO). Trained RL policy takes actions to increase balance of personal account.

## Overview

This project demonstrates:
- Creation and configuring a Gym environment which simulates Stock Market
- Integration of custom Gym environment with Sample Factory framework for training and evaluation
- Monitoring and analyzing agent performance over time

## Environments

This project includes two custom Gym environments tailored for different reinforcement learning policies:

- **Trader-v0**: An environment configured for use with an MLP (Multi-Layer Perceptron) policy. MLP allows to take actions based on the last state of the Stock Market. (Initial version)
- **Trader-v1**: An environment configured for use with a CNN (Convolutional Neural Network) policy. CNN allows to take actions based on the frame stack of previous states of the Stock Market (Advanced version)

These environments simulate trading scenarios where different neural network architectures can be trained and evaluated.

### Trader-v1 Environment Description (CNN-based)

The `Trader-v1` environment in `gym_examples/envs/trader_cnn.py` is a custom environment designed for reinforcement learning agents to learn trading strategies using a Convolutional Neural Network (CNN) head of policy. Key aspects of this environment include:

- **Observation Space**: Provides the agent with information on market conditions. This typically includes price histories and technical indicators formatted as a multi-dimensional array, suitable for CNN input.
- **Action Space**: The agent can choose from a set of discrete actions such as "buy," "sell," or "hold."
- **Reward Structure**: The reward is based on the agent’s profit or loss after certain interval of steps (CNN-base environment). Rewards are calculated based on the agent’s balance changes due to trading actions, scaled by the `REWARD_SCALING` parameter.

### Key Environment Options for Trader-v1

Below are the main options and settings within `gym_examples/envs/trader_cnn.py` that configure the behavior of the `Trader-v1` environment:

- **`MAX_STEPS`**: Specifies the maximum number of steps per episode. This parameter limits the duration of each trading episode, forcing the agent to maximize profits within a constrained timeframe.
- **`OBSERVATION_SPACE`**: Defines the shape and type of observations that the agent will receive. Typically, this includes historical price data and other financial indicators, formatted to work effectively with CNN architectures.
- **`ACTION_SPACE`**: Determines the set of possible actions the agent can take, such as:
  - **Buy**: Purchase assets based on available balance.
  - **Sell**: Sell owned assets to realize profit or minimize loss.
  - **Hold**: Maintain the current position without action.
- **`REWARD_SCALING`**: A factor to scale the reward values, which can help stabilize training and prevent extreme reward fluctuations.
- **`INITIAL_BALANCE`**: The starting balance or capital given to the agent at the start of each episode, which serves as the initial state for trading.
- **`TRANSACTION_COST`**: A fixed cost incurred with each trade, representing transaction fees in real-world trading scenarios.
- **`RISK_FACTOR`**: A multiplier used in reward calculations to influence the agent’s risk tolerance, encouraging risk-aware decision-making.
- **`PRICE_HISTORY_LENGTH`**: Defines how many previous time steps or price points are included in the observation. This provides historical context, which is crucial for CNN-based policies relying on time-series data.
- **`FEATURE_SCALING`**: Option to normalize or scale the input features, which can improve learning stability by ensuring all input features are on a similar scale.
- **`DISCOUNT_FACTOR` (`gamma`)**: Defines the importance of future rewards. A higher discount factor encourages the agent to focus on long-term profitability rather than short-term gains.
- **`EPSILON_DECAY`**: Controls the decay rate for epsilon in epsilon-greedy exploration, adjusting the balance between exploration (trying new actions) and exploitation (choosing known profitable actions) as training progresses.
- **`RENDER_MODE`**: Defines the visualization mode of the environment. When set, it allows for real-time visual representation of the agent’s actions and market state during training or evaluation.

These options can be modified in `trader_cnn.py` to optimize training according to specific requirements.

## Integration with Sample Factory

This project integrates with [Sample Factory](https://github.com/alex-petrenko/sample-factory), a high-performance reinforcement learning framework. Sample Factory enables efficient training of RL models through optimized parallelism.

### Running Training with Sample Factory

To run training using Sample Factory, use the scripts located in `/run_scripts`:

1. **Set up Sample Factory**:
   - Install Sample Factory using pip:
     ```bash
     pip install sample-factory
     ```

2. **Running Trader-v0 with MLP Policy**:
   - To start training in the `Trader-v0` environment with an MLP policy, execute:
     ```bash
     python run_scripts/train_custom_env_custom_model.py --env Trader-v0 --policy mlp_policy
     ```

3. **Running Trader-v1 with CNN Policy**:
   - To start training in the `Trader-v1` environment with a CNN policy, execute:
     ```bash
     python run_scripts/train_custom_env_custom_model.py --env Trader-v1 --policy cnn_policy
     ```

These commands will initialize Sample Factory to run the specified environment and begin the training process.

## Prerequisites

- Python 3.7+
- OpenAI Gym library
- Sample Factory
- Additional dependencies (optional)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/execbat/gym-example.git
   cd gym-example


## Structure
```main.py```: Main file to run the Gym environment example.

```run_scripts/train_custom_env_custom_model.py```: Script for training custom environments with specified models using Sample Factory.

```agents/```: Contains RL algorithms implemented for this project.

```utils/```: Helper functions for data processing and analysis.

```config.py```: Settings and hyperparameters for the environment and training process.

```gym_examples/envs/trader.py```: Contains the Trader-v0 MLP-based trading environment and its configurations.

```gym_examples/envs/trader_cnn.py```: Contains the Trader-v1 CNN-based trading environment and its configurations.

```gym_examples/envs/trader_cnn_env_cfg.py```: Environment configuration for Trader-v1, specifying parameters for the CNN-based trading environment.

## Contributing
Feel free to open issues or submit pull requests to enhance this project. Contributions are welcome!

## License
This project is licensed under the MIT License.
