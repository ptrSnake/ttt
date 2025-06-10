use rand::Rng;
use std::io;

use crate::game::{Cell, GameState, Player};

// Neural network parameters
pub const NN_INPUT_SIZE: usize = 18; // 9 cells * 2 (X and O) -> there are 18 input features 
pub const NN_HIDDEN_SIZE: usize = 100; // 100 hidden neurons
pub const NN_OUTPUT_SIZE: usize = 9; // 9 possible moves
pub const NN_LEARNING_RATE: f64 = 0.1; // Learning rate for backpropagation

pub struct NeuralNetwork {
    pub weights_ih: [f64; NN_INPUT_SIZE * NN_HIDDEN_SIZE],
    pub weights_ho: [f64; NN_HIDDEN_SIZE * NN_OUTPUT_SIZE],
    pub biases_h: [f64; NN_HIDDEN_SIZE],
    pub biases_o: [f64; NN_OUTPUT_SIZE],

    pub inputs: [f64; NN_INPUT_SIZE],
    pub hidden: [f64; NN_HIDDEN_SIZE],
    pub raw_logits: [f64; NN_OUTPUT_SIZE],
    pub outputs: [f64; NN_OUTPUT_SIZE],
}

impl NeuralNetwork {
    /// Initializes a new neural network with random weights and biases
    /// using a uniform distribution in the range [-0.5, 0.5]
    /// from_fn -> is a Rust function that generates an array by applying a closure to each index
    pub fn new() -> Self {
        let mut rng = rand::rng();

        let weights_ih = std::array::from_fn(|_| rng.random_range(-0.5..0.5));
        let weights_ho = std::array::from_fn(|_| rng.random_range(-0.5..0.5));
        let biases_h = std::array::from_fn(|_| rng.random_range(-0.5..0.5));
        let biases_o = std::array::from_fn(|_| rng.random_range(-0.5..0.5));

        NeuralNetwork {
            weights_ih,
            weights_ho,
            biases_h,
            biases_o,
            inputs: [0.0; NN_INPUT_SIZE],
            hidden: [0.0; NN_HIDDEN_SIZE],
            raw_logits: [0.0; NN_OUTPUT_SIZE],
            outputs: [0.0; NN_OUTPUT_SIZE],
        }
    }

    /// Performs a forward pass through the neural network.
    /// This function computes the outputs of the neural network given the inputs.
    /// It calculates the hidden layer activations using ReLU,
    /// then computes the output layer logits, and finally applies softmax to get the output probabilities.
    pub fn forward_pass(&mut self, inputs: &[f64]) {
        assert_eq!(inputs.len(), NN_INPUT_SIZE);
        self.inputs.copy_from_slice(inputs);

        // Input -> Hidden
        // hᵢ = ReLU(Σⱼ (xⱼ * wⱼᵢ) + bᵢ)
        for i in 0..NN_HIDDEN_SIZE {
            let mut sum = self.biases_h[i];

            // hidden layer neurons are connected to all input features
            // hidden layers give us a non-linear transformation of the inputs
            for j in 0..NN_INPUT_SIZE {
                // weights_ih is a 2D matrix flattened to 1D
                // weights_ih[j * NN_HIDDEN_SIZE + i] gives us the weight from input j to hidden i
                // inputs[j] is the j-th input feature
                sum += self.inputs[j] * self.weights_ih[j * NN_HIDDEN_SIZE + i];
            }

            self.hidden[i] = relu(sum);
        }

        // Hidden -> Output
        // oᵢ = softmax(Σⱼ (hⱼ * wⱼᵢ) + bᵢ)
        for i in 0..NN_OUTPUT_SIZE {
            let mut sum = self.biases_o[i];

            // output layer neurons are connected to all hidden features
            // output layer gives us the final probabilities for each move
            // weights_ho is a 2D matrix flattened to 1D
            // weights_ho[j * NN_OUTPUT_SIZE + i] gives us the weight from hidden j to output i
            // self.hidden[j] is the j-th hidden feature
            for j in 0..NN_HIDDEN_SIZE {
                sum += self.hidden[j] * self.weights_ho[j * NN_OUTPUT_SIZE + i];
            }

            self.raw_logits[i] = sum;
        }

        // Apply softmax to raw logits to get output probabilities
        // softmax function converts raw logits into probabilities
        softmax(&self.raw_logits, &mut self.outputs, NN_OUTPUT_SIZE);
    }

    /// Backpropagation algorithm to update weights and biases based on the target probabilities
    /// This function calculates the deltas for the output and hidden layers,
    /// then updates the weights and biases accordingly.
    pub fn backprop(&mut self, target_probs: &[f64], learning_rate: f64, reward_scaling: f64) {
        let mut output_deltas = [0.0; NN_OUTPUT_SIZE];
        let mut hidden_deltas = [0.0; NN_HIDDEN_SIZE];

        // Calculate output layer deltas
        for i in 0..NN_OUTPUT_SIZE {
            output_deltas[i] = (self.outputs[i] - target_probs[i]) * reward_scaling.abs();
        }

        // Calculate hidden layer deltas
        for i in 0..NN_HIDDEN_SIZE {
            let mut error = 0.0;
            for j in 0..NN_OUTPUT_SIZE {
                error += output_deltas[j] * self.weights_ho[i * NN_OUTPUT_SIZE + j];
            }
            hidden_deltas[i] = error * relu_derivative(self.hidden[i]);
        }

        // Update hidden -> output weights
        for i in 0..NN_HIDDEN_SIZE {
            for j in 0..NN_OUTPUT_SIZE {
                self.weights_ho[i * NN_OUTPUT_SIZE + j] -=
                    learning_rate * output_deltas[j] * self.hidden[i];
            }
        }

        // Update output biases
        for i in 0..NN_OUTPUT_SIZE {
            self.biases_o[i] -= learning_rate * output_deltas[i];
        }

        // Update input -> hidden weights
        for i in 0..NN_INPUT_SIZE {
            for j in 0..NN_HIDDEN_SIZE {
                self.weights_ih[i * NN_HIDDEN_SIZE + j] -=
                    learning_rate * hidden_deltas[j] * self.inputs[i];
            }
        }

        // Update hidden biases
        for i in 0..NN_HIDDEN_SIZE {
            self.biases_h[i] -= learning_rate * hidden_deltas[i];
        }
    }

    /// Learns from a completed game by updating the neural network weights
    /// based on the move history and the game outcome.
    /// This function processes the move history,
    /// determines if the move was made by the neural network,
    /// reconstructs the game state up to that move,
    /// and updates the neural network weights based on the outcome.
    pub fn learn_from_game(&mut self, move_history: &[usize], nn_plays_as_o: bool, winner: Cell) {
        assert!(!move_history.is_empty(), "Move history cannot be empty");

        // Determine the reward based on the game outcome
        let reward = match winner {
            Cell::Empty => 0.3,               // Tie
            Cell::O if nn_plays_as_o => 1.0,  // NN wins as O
            Cell::X if !nn_plays_as_o => 1.0, // NN wins as X
            _ => -2.0,                        // NN loses
        };
        let num_moves = move_history.len();

        for move_idx in 0..num_moves {
            // Determine if this move was made by the NN
            // Game starts with Human (X), so:
            // - Even indices (0, 2, 4, 6, 8): Human plays X
            // - Odd indices (1, 3, 5, 7): AI plays O
            let is_nn_move = if nn_plays_as_o {
                move_idx % 2 == 1 // NN plays on odd moves (as O)
            } else {
                move_idx % 2 == 0 // NN plays on even moves (as X)
            };

            if !is_nn_move {
                continue;
            }

            // Reconstruct game state up to this move
            let mut state = GameState::new();

            for i in 0..move_idx {
                let _ = state.make_move(move_history[i]); // Ignore errors in reconstruction
            }

            // Get NN input for this state
            let mut inputs = [0.0; NN_INPUT_SIZE];
            state.board_to_input(&mut inputs);
            self.forward_pass(&inputs);

            // Create target probabilities
            let mut target_probs = [0.0; NN_OUTPUT_SIZE];
            let chosen_move = move_history[move_idx];

            let move_importance = 0.25 + 0.75 * (move_idx as f64 / num_moves as f64);
            let scaled_reward = reward * move_importance;

            if scaled_reward > 0.0 {
                // Good move - reinforce it
                target_probs[chosen_move] = 1.0;
            } else {
                // Bad move - reinforce other valid moves
                let valid_moves = state.get_valid_moves();
                let valid_alternatives: Vec<usize> = valid_moves
                    .into_iter()
                    .filter(|&m| m != chosen_move)
                    .collect();

                if !valid_alternatives.is_empty() {
                    let prob_per_move = 1.0 / valid_alternatives.len() as f64;
                    for &mv in &valid_alternatives {
                        target_probs[mv] = prob_per_move;
                    }
                }
            }

            self.backprop(&target_probs, NN_LEARNING_RATE, scaled_reward);
        }
    }

    /// Plays a game of Tic-Tac-Toe against the computer.
    /// This function allows a human player to play against the AI,
    /// tracking the moves made during the game.
    pub fn play_game(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mut state = GameState::new();
        let mut move_history: Vec<usize> = Vec::with_capacity(9);

        println!("Welcome to Tic-Tac-Toe! You are X, computer is O.");

        loop {
            state.display_board();

            // Check if game is over
            if let Some(winner) = state.check_game_over() {
                println!(
                    "{}",
                    match winner {
                        Cell::X => "You win!",
                        Cell::O => "Computer wins!",
                        Cell::Empty => "It's a tie!",
                    }
                );

                self.learn_from_game(&move_history, true, winner);
                break;
            }

            match state.current_player {
                Player::Human => {
                    println!("Your move (0-8): ");
                    let mut input = String::new();
                    io::stdin().read_line(&mut input)?;

                    let mv: usize = match input.trim().parse() {
                        Ok(n) if state.is_valid_move(n) => n,
                        _ => {
                            println!("Invalid move. Try again.");
                            continue;
                        }
                    };

                    state.make_move(mv)?;
                    move_history.push(mv);
                }
                Player::AI => {
                    if let Some(mv) = state.get_computer_move(self, true) {
                        println!("Computer chose position {}", mv);
                        state.make_move(mv)?;
                        move_history.push(mv);
                    } else {
                        println!("No valid moves available!");
                        break;
                    }
                }
            }
        }

        Ok(())
    }

    /// Plays a random game where the AI plays against a random player.
    /// This function simulates a game where the AI plays against a random player,
    /// learning from the game outcome.
    pub fn play_random_game(&mut self) -> Cell {
        let mut state = GameState::new();
        let mut move_history = Vec::with_capacity(9);

        loop {
            if let Some(winner) = state.check_game_over() {
                self.learn_from_game(&move_history, true, winner);
                return winner;
            }

            let mv = match state.current_player {
                Player::Human => {
                    // Random player
                    state.get_random_move()
                }
                Player::AI => {
                    // Neural network
                    state.get_computer_move(self, false)
                }
            };

            if let Some(mv) = mv {
                let _ = state.make_move(mv); // Ignore errors in training
                move_history.push(mv);
            } else {
                break; // No valid moves
            }
        }

        Cell::Empty // Shouldn't reach here, but return tie as fallback
    }

    /// Trains the neural network by playing a specified number of games against a random player.
    /// This function simulates multiple games where the neural network plays against a random player,
    /// learning from each game outcome. It tracks wins, losses, and ties,
    pub fn train_against_random(&mut self, num_games: u32) {
        let mut wins = 0;
        let mut losses = 0;
        let mut ties = 0;

        let mut total_games = 0;
        let mut total_wins = 0;
        let mut total_losses = 0;
        let mut total_ties = 0;

        println!("Training against random player...");

        for i in 0..num_games {
            let winner = self.play_random_game();
            match winner {
                Cell::O => wins += 1,   // Assuming NN is O
                Cell::X => losses += 1, // Random player is X
                Cell::Empty => ties += 1,
            }

            if (i + 1) % 1000 == 0 {
                let win_rate = wins as f64 / 1000.0 * 100.0;

                total_games += 1000;
                total_wins += wins;
                total_losses += losses;
                total_ties += ties;

                println!(
                    "Games: {}, Wins: {} ({:.1}%), Losses: {}, Ties: {}",
                    i + 1,
                    wins,
                    win_rate,
                    losses,
                    ties
                );
                wins = 0;
                losses = 0;
                ties = 0;
            }
        }

        println!("Training complete! {} games played.", num_games);
        println!(
            "Total Games: {}, Wins: {}, Losses: {}, Ties: {}",
            total_games, total_wins, total_losses, total_ties
        );
        println!(
            "Final Win Rate: {:.1}%",
            total_wins as f64 / total_games as f64 * 100.0
        );
    }
}

impl Default for NeuralNetwork {
    fn default() -> Self {
        Self::new()
    }
}

// Activation functions
pub fn relu(x: f64) -> f64 {
    x.max(0.0)
}

pub fn relu_derivative(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

pub fn softmax(inputs: &[f64], outputs: &mut [f64], size: usize) {
    let max_val = inputs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let mut sum = 0.0;
    for i in 0..size {
        outputs[i] = (inputs[i] - max_val).exp();
        sum += outputs[i];
    }

    if sum > 0.0 {
        for i in 0..size {
            outputs[i] /= sum;
        }
    } else {
        // Fallback to uniform distribution
        for i in 0..size {
            outputs[i] = 1.0 / size as f64;
        }
    }
}
