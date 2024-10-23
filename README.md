# Binary Data Classification and Randomness Testing Server

## Overview
This project implements a server-client architecture for classifying binary data using various statistical randomness tests and machine learning models (XGBoost). The server receives binary data from clients, performs feature extraction, and predicts classes based on pre-trained models. The project supports retraining the model with new data.

## Project Structure

<ul>
  <h1>Server:</h1>
  <li>
    Receives binary data from clients.
  </li>
  <li>
    Runs various statistical randomness tests using features like Frequency, Matrix, and Spectral tests.
  </li>
  <li>
    Predicts the class of the binary data using an XGBoost model.
  </li>
  <li>
    Allows retraining of the model with newly received data.
  </li>
</ul>

<ul>
  <h1>Client:</h1>
  <li>
    Reads `.bin` files from a specified directory.
  </li>
  <li>
    Sends the files to the server in either "test" or "train" mode.
  </li>
  <li>
    Supports multithreading for sending multiple files concurrently.
  </li>
</ul>

## Requirements

- Python 3.x
- Libraries:
  - `socket`
  - `json`
  - `xgboost`
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `concurrent.futures`

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Hardik-2311/DRDO_project.git
    cd projectname
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Place the pre-trained XGBoost model (`xgboost_model.model`) in the appropriate folder.

## How to Use

### Starting the Server

The server listens for connections and processes binary data sent by the clients. To start the server:

```bash

python socket_server.py

```

### Starting the Client

```bash
python socket_client.py [mode: test/train] [dir: for the bin files]

```


