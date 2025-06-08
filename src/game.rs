use crate::nn::*;

pub struct GameState {
    pub board: [char; 9],   // 0 for empty, 1 for player 1, 2 for player 2
    pub current_player: u8, // 1 or 2
}

impl GameState {
    pub fn new() -> Self {
        GameState {
            board: ['.'; 9],
            current_player: 0,
        }
    }

    pub fn display_board(&self) {
        for i in 0..3 {
            println!(
                "{} | {} | {}",
                self.board[i * 3],
                self.board[i * 3 + 1],
                self.board[i * 3 + 2]
            );

            if i < 2 {
                println!("---------");
            }
        }
    }

    pub fn board_to_inputd(&self, inputs: &mut [f64]) {
        for i in 0..9 {
            if self.board[i] == '.' {
                inputs[i * 2] = 0.0;
                inputs[i * 2 + 1] = 0.0;
            } else if self.board[i] == 'X' {
                inputs[i * 2] = 1.0;
                inputs[i * 2 + 1] = 0.0;
            } else {
                inputs[i * 2] = 0.0;
                inputs[i * 2 + 1] = 1.0;
            }
        }
    }

    pub fn check_game_over(&self) -> Option<char> {
        // Check rows, columns, and diagonals for a win

        // Check rows
        for i in 0..3 {
            if self.board[i * 3] != '.'
                && self.board[i * 3] == self.board[i * 3 + 1]
                && self.board[i * 3 + 1] == self.board[i * 3 + 2]
            {
                return Some(self.board[i * 3]);
            }
        }

        // Check columns
        for i in 0..3 {
            if self.board[i] != '.'
                && self.board[i] == self.board[i + 3]
                && self.board[i + 3] == self.board[i + 6]
            {
                return Some(self.board[i]);
            }
        }

        // Check diagonals
        if self.board[0] != '.' && self.board[0] == self.board[4] && self.board[4] == self.board[8]
        {
            return Some(self.board[0]);
        }

        // Check for tie
        let mut empty_tiles = 0;

        for i in 0..9 {
            if self.board[i] == '.' {
                empty_tiles += 1;
            }
        }

        if empty_tiles == 0 {
            return Some('T'); // Tie
        }

        None // Game is still ongoing
    }

    pub fn get_computer_move(&mut self, nn: &mut NeuralNetwork, display_probs: bool) -> i32 {
        let mut inputs = [0.0; NN_INPUT_SIZE];

        self.board_to_inputd(&mut inputs);
        nn.forward_pass(&inputs);

        let mut highest_prob = -1.0;
        let mut hightest_prob_idx: i32 = -1;
        let mut best_move: i32 = -1;
        let mut best_legal_prob = -1.0;

        for i in 0..9 {
            let i32_idx = i as i32;

            if nn.outputs[i] > highest_prob {
                highest_prob = nn.outputs[i];
                hightest_prob_idx = i32_idx;
            }

            if self.board[i] == '.' && (best_move == -1 || nn.outputs[i] > best_legal_prob) {
                best_move = i32_idx;
                best_legal_prob = nn.outputs[i];
            }
        }

        if display_probs {
            println!("Neural Network move probabilities:");

            for row in 0..3 {
                for col in 0..3 {
                    let pos = row * 3 + col;

                    println!("{:.2}", nn.outputs[pos] * 100.0);

                    if pos == hightest_prob_idx as usize {
                        print!("*");
                    }

                    if pos == best_move as usize {
                        print!("#");
                    }

                    print!(" ");
                }

                println!();
            }

            let mut total_prob = 0.0;

            for i in 0..9 {
                total_prob += nn.outputs[i];
                println!("Sum of all probabilities: {:.2}", total_prob);
            }
        }

        best_move
    }
}
