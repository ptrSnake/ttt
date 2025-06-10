# Tic-Tac-Toe Neural Network (Rust)

---

This project implements a neural network to play **Tic-Tac-Toe (Noughts and Crosses)**, trained using a **reinforcement learning** algorithm based on backpropagation in Rust.

The core idea and architecture are derived from the work of [Antirez](https://github.com/antirez) and his `ttt-rl` project written in C. This is a rewrite in Rust that aims to explore neural networks and reinforcement learning in a simple game environment.

## Table of Contents

* [Features](#features)

* [Neural Network Architecture](#neural-network-architecture)

* [Training](#training)

* [How to Compile and Run](#how-to-compile-and-run)

  * [Prerequisites](#prerequisites)

  * [Compilation](#compilation)

  * [Running the Application](#running-the-application)

## Features

* **Feed-Forward Neural Network:** A simple neural network with an input layer, a hidden layer, and an output layer.

* **Reinforcement Learning:** The network learns by playing games against a random opponent, updating its weights and biases based on the game's outcome (win, loss, tie).

* **Backpropagation:** The core algorithm used for efficient updating of the network's weights and biases.

* **Activation Functions:** It implements the **ReLU** activation function for the hidden layer and **Softmax** for the output layer, converting logits into move probabilities.

* **Human vs. AI Games:** Play interactive games against the trained AI.

* **Training Simulation:** A function to simulate a large number of games against a random player to train the network.

---

## Neural Network Architecture

The neural network has the following configuration:

* **Input Layer (`NN_INPUT_SIZE` = 18):** Represents the Tic-Tac-Toe board state. Each cell (9 in total) has two features: one for the presence of 'X' and one for the presence of 'O'. This allows the network to clearly distinguish who occupies each cell.

* **Hidden Layer (`NN_HIDDEN_SIZE` = 100):** Contains 100 hidden neurons. This layer enables the network to learn complex, non-linear relationships from the inputs. The activation function used is **ReLU**.

* **Output Layer (`NN_OUTPUT_SIZE` = 9):** Corresponds to the 9 possible moves on the board. The output of this layer, after applying the **Softmax** function, represents the probability that the network considers a given move to be the best.

* **Learning Rate (`NN_LEARNING_RATE` = 0.1):** Controls the magnitude of adjustments to weights and biases during training.

---

## Training

Training is carried out by simulating games against a random player. For each game, the network records the sequence of moves. At the end of the game, based on the result (win, loss, or tie) and a "move importance" factor (more recent moves have a greater impact), the network uses backpropagation to update its weights and biases.

* **Reward:**

  * AI Win: +1.0

  * Tie: +0.3

  * AI Loss: -2.0 (higher penalty for losing)

* **Move Importance:** Later moves in the game (closer to the final outcome) are weighted more heavily when updating the network's parameters. This helps the network learn from the immediate consequences of its actions.

The `train_against_random` function allows you to train the network for a specified number of games, displaying win, loss, and tie statistics every 1000 games.

---

## How to Compile and Run

### Prerequisites

Ensure you have [Rust and Cargo](https://www.rust-lang.org/tools/install) installed on your system.

### Compilation

To compile the project in **release mode** (optimized for performance), navigate to the project's root directory (where `Cargo.toml` is located) and use the following command:

```bash
cargo build --release