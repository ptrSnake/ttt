use rand::Rng;

use crate::game::GameState;

// Network parameters
pub const NN_INPUT_SIZE: usize = 18;
pub const NN_HIDDEN_SIZE: usize = 100;
pub const NN_OUTPUT_SIZE: usize = 9;
pub const NN_LEARNING_RATE: f64 = 0.1;

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
    pub fn new() -> Self {
        let mut rng = rand::rng();
        let weights_ih = [0.0; NN_INPUT_SIZE * NN_HIDDEN_SIZE].map(|_| rng.random_range(-0.5..0.5));
        let weights_ho =
            [0.0; NN_HIDDEN_SIZE * NN_OUTPUT_SIZE].map(|_| rng.random_range(-0.5..0.5));
        let biases_h = [0.0; NN_HIDDEN_SIZE].map(|_| rng.random_range(-0.5..0.5));
        let biases_o = [0.0; NN_OUTPUT_SIZE].map(|_| rng.random_range(-0.5..0.5));

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

    pub fn forward_pass(&mut self, inputs: &[f64]) {
        assert_eq!(inputs.len(), NN_HIDDEN_SIZE);

        // copy inputs to the internal state
        self.inputs.copy_from_slice(inputs);

        // input to hidden layer
        for i in 0..NN_HIDDEN_SIZE {
            let mut sum = self.biases_h[i];

            for j in 0..NN_INPUT_SIZE {
                sum += self.inputs[j] * self.weights_ih[j * NN_HIDDEN_SIZE + i];
            }

            self.hidden[i] = relu(sum);
        }

        // hidden to output layer
        for i in 0..NN_OUTPUT_SIZE {
            self.raw_logits[i] = self.biases_o[i];

            for j in 0..NN_HIDDEN_SIZE {
                self.raw_logits[i] += self.hidden[j] * self.weights_ho[j * NN_OUTPUT_SIZE + i];
            }
        }

        softmax(&self.raw_logits, &mut self.outputs, NN_OUTPUT_SIZE);
    }

    pub fn backdrop(&mut self, target_probs: &[f64], learning_rate: f64, reward_scailing: f64) {
        let mut output_deltas = [0.0; NN_OUTPUT_SIZE];
        let mut hidden_deltas = [0.0; NN_HIDDEN_SIZE];

        for i in 0..NN_OUTPUT_SIZE {
            output_deltas[i] = (self.outputs[i] - target_probs[i]) * reward_scailing.abs();
        }

        for i in 0..NN_HIDDEN_SIZE {
            let mut error = 0.0;

            for j in 0..NN_OUTPUT_SIZE {
                error += output_deltas[j] * self.weights_ho[i * NN_OUTPUT_SIZE + j];
            }

            hidden_deltas[i] = error * relu_derivative(self.hidden[i]);
        }

        for i in 0..NN_HIDDEN_SIZE {
            for j in 0..NN_OUTPUT_SIZE {
                self.weights_ho[i * NN_OUTPUT_SIZE + j] -=
                    learning_rate * output_deltas[j] * self.hidden[i];
            }
        }

        for i in 0..NN_OUTPUT_SIZE {
            self.biases_o[i] -= learning_rate * output_deltas[i];
        }

        for i in 0..NN_INPUT_SIZE {
            for j in 0..NN_HIDDEN_SIZE {
                self.weights_ih[i * NN_HIDDEN_SIZE + j] -=
                    learning_rate * hidden_deltas[j] * self.inputs[i];
            }
        }

        for i in 0..NN_HIDDEN_SIZE {
            self.biases_h[i] -= learning_rate * hidden_deltas[i];
        }
    }

    pub fn learn_from_game(
        &mut self,
        move_history: &[f64],
        num_moves: u32,
        nn_moves_even: bool,
        winner: char,
    ) {
        let mut reward = 0.0;

        let nn_symbol = if nn_moves_even { 'O' } else { 'X' };

        if winner == 'T' {
            reward = 0.3;
        } else if winner == nn_symbol {
            reward = 1.0;
        } else {
            reward = -2.0;
        }

        let mut state: GameState;

        for move_idx in 0..num_moves {
            if (nn_moves_even && move_idx % 2 != 1) || (!nn_moves_even && move_idx % 2 != 0) {
                continue; // Skip moves not made by the neural network
            }

            state = GameState::new();
            let mut target_probs = [0.0; NN_OUTPUT_SIZE];

            for i in 0..move_idx as usize {
                let symbol = if i % 2 == 0 { 'X' } else { 'O' };
                state.board[move_history[i] as usize] = symbol;
            }

            let mut inputs = [0.0; NN_INPUT_SIZE];
            state.board_to_inputd(inputs.as_mut_slice());
            self.forward_pass(&inputs);

            let m = move_history[move_idx as usize] as usize;

            let move_importance = 0.5 * 0.5 * move_idx as f64 / num_moves as f64;
            let scaled_reward = reward * move_importance;

            for i in 0..NN_OUTPUT_SIZE {
                target_probs[i] = 0.0;
            }

            if scaled_reward > 0.0 {
                target_probs[m] = 1.0; // Target probability for the move made by the NN
            } else {
                let valid_moves_left = 9 - move_idx as usize - 1;
                let other_prob = 1.0 / valid_moves_left as f64;

                for i in 0..9 {
                    if state.board[i] == '.' && i != m {
                        target_probs[i] = other_prob; // Uniform distribution for other moves
                    }
                }
            }

            self.backdrop(&target_probs, NN_LEARNING_RATE, scaled_reward);
        }
    }

    pub fn play_game(&mut self) {
        let mut state = GameState::new();
        let mut move_history = [0.0; 9];
        let mut num_moves = 0;

        while !state.check_game_over().is_none() {
            state.display_board();

            if state.current_player == 0 {
                println!("Player X's turn. Enter your move (0-8):");
                let mut input = String::new();
                std::io::stdin()
                    .read_line(&mut input)
                    .expect("Failed to read line");

                let move_idx: usize = input.trim().parse().expect("Please enter a number");

                if move_idx >= 9 || state.board[move_idx] != '.' {
                    println!("Invalid move. Try again.");
                    continue;
                }

                state.board[move_idx] = 'X';
                num_moves += 1;
                move_history[num_moves as usize] = move_idx as f64;
            } else {
                println!("Computer's move:");
                let m = state.get_computer_move(self, true);
                state.board[m as usize] = 'O';
                println!("Computer placed 'O' at position {}", m);
                num_moves += 1;
                move_history[num_moves as usize] = m as f64;
            }

            state.current_player = 1 - state.current_player; // Switch players

            let winner = state.check_game_over();

            if let Some(winner_char) = winner {
                if winner_char == 'X' {
                    println!("Player X wins!");
                } else if winner_char == 'O' {
                    println!("Computer wins!");
                } else {
                    println!("It's a tie!");
                }
            }

            if let Some(winner_char) = winner {
                self.learn_from_game(&move_history, num_moves, true, winner_char);
            }
        }

        state.display_board();
    }
}

pub fn relu(x: f64) -> f64 {
    if x < 0.0 { 0.0 } else { x }
}

pub fn relu_derivative(x: f64) -> f64 {
    if x < 0.0 { 0.0 } else { 1.0 }
}

pub fn softmax(inputs: &[f64], outputs: &mut [f64], size: usize) {
    let mut max_val = inputs[0];

    for i in 1..size {
        if inputs[i] > max_val {
            max_val = inputs[i];
        }
    }

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
        for i in 0..size {
            outputs[i] = 1.0 / size as f64; // Uniform distribution if sum is zero
        }
    }
}
