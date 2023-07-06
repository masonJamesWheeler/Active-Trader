# Emergent Trading Strategies with DQN in Stock Market Trading

This repository contains the implementation of a Deep Reinforcement Learning (DRL) model, specifically a Deep Q-Network (DQN), applied to the realm of stock market trading. The project is based on the research paper "Emergent Trading Strategies from Deep Reinforcement Learning Models in Stock Market Trading" by Mason James Wheeler from the University of Washington.

## Overview

The project consists of a custom trading environment where a DRL agent learns to make trading decisions based on observed market data, including various indicators along with price and volume action. The agent's actions are guided by an epsilon-greedy strategy, which provides a balance between exploration of new actions and exploitation of known information. 

To facilitate the learning process, a replay memory mechanism is implemented, which stores and samples transitions, enabling the agent to learn from past experiences and improve its future decisions. One unique aspect of this project is the use of locally trained, smaller-sized models that can generate emergent successful trading strategies when trained on a large dataset.

## Directory Structure

Here's the directory structure:

```
.
├── Data
│   ├── Get_Fast_Data.py
│   └── Data.py
├── Documentation
│   └── DQN_WHEELER.pdf
├── Environment
│   ├── LiveStockEnvironment.py
│   └── StockEnvironment.py
├── Models
│   ├── ModelOptimization.py
│   └── .pth files
├── Results
│   ├── Media
│   │   ├── Agent_Results_AAPL.png
│   │   ├── Agent_Results_GOOGL.png
│   │   ├── Agent_Results_MSFT.png
│   │   └── Agent_Results_UBER.png
│   ├── Data
│       ├── Conv_Training.csv
│       ├── Dense_TrainingL.csv
│   
├── Tests
│   ├── Unit_Tests
│   
├── Utilities
│   ├── Figure_Creator.py
|   ├── Csv_Parser.py
│   ├── Visualize_Results.py
│   └── IB_API.py
├── Active_Trader.py
├──Init_Trading.py
├── Training_Data
├── venv
└── .gitignore
```

## Getting Started

To start with this project:

1. Clone the repository.
2. Navigate into the project directory.
3. Create a `.env` file at the root of the project and add your API keys:

    ```
    ALPACA_KEY="<Your Alpaca Key>"
    ALPACA_SECRET_KEY="<Your Alpaca Secret Key>"
    ALPHA_VANTAGE_API_KEY="<Your Alpha Vantage API Key>"
    PAPER_ALPACA_KEY="<Your Paper Alpaca Key>"
    PAPER_ALPACA_SECRET_KEY="<Your Paper Alpaca Secret Key>"
    ```

4. Install the necessary dependencies using pip:

    ```
    pip install -r requirements.txt
    ```

    Note: It is recommended to use a virtual environment to avoid conflicts with other packages.

5. To start the training process, navigate to the `Trading` directory and run:

    ```
    python Init_Trading.py
    ```

## Contributing

Please read `CONTRIBUTING.md` for details on our code of conduct, and the process for submitting pull requests.

## License

This project is licensed under the MIT License. See the `LICENSE.md` file for details.