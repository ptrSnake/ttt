use crate::nn::*;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Player {
    Human = 0,
    AI = 1,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Cell {
    Empty,
    X,
    O,
}

impl fmt::Display for Cell {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Cell::Empty => write!(f, "."),
            Cell::X => write!(f, "X"),
            Cell::O => write!(f, "O"),
        }
    }
}

pub struct GameState {
    pub board: [Cell; 9],
    pub current_player: Player,
}

impl GameState {
    /// Creates a new game state
    pub fn new() -> Self {
        GameState {
            board: [Cell::Empty; 9],
            current_player: Player::Human,
        }
    }

    /// Displays the game board
    pub fn display_board(&self) {
        for i in 0..3 {
            println!(
                " {} | {} | {} ",
                self.board[i * 3],
                self.board[i * 3 + 1],
                self.board[i * 3 + 2]
            );
            if i < 2 {
                println!("-----------");
            }
        }
    }

    /// Makes a move at the specified position
    pub fn make_move(&mut self, position: usize) -> Result<(), &'static str> {
        if position >= 9 {
            return Err("Invalid position: must be 0-8");
        }

        if self.board[position] != Cell::Empty {
            return Err("Position already occupied");
        }

        self.board[position] = match self.current_player {
            Player::Human => Cell::X,
            Player::AI => Cell::O,
        };

        // Switch player
        self.current_player = match self.current_player {
            Player::Human => Player::AI,
            Player::AI => Player::Human,
        };

        Ok(())
    }

    /// Converts the board to neural network input
    pub fn board_to_input(&self, inputs: &mut [f64]) {
        for i in 0..9 {
            match self.board[i] {
                Cell::Empty => {
                    inputs[i * 2] = 0.0;
                    inputs[i * 2 + 1] = 0.0;
                }
                Cell::X => {
                    inputs[i * 2] = 1.0;
                    inputs[i * 2 + 1] = 0.0;
                }
                Cell::O => {
                    inputs[i * 2] = 0.0;
                    inputs[i * 2 + 1] = 1.0;
                }
            }
        }
    }

    /// Checks if the game is over and returns the winner
    pub fn check_game_over(&self) -> Option<Cell> {
        // Check rows
        for i in 0..3 {
            if self.board[i * 3] != Cell::Empty
                && self.board[i * 3] == self.board[i * 3 + 1]
                && self.board[i * 3 + 1] == self.board[i * 3 + 2]
            {
                return Some(self.board[i * 3]);
            }
        }

        // Check columns
        for i in 0..3 {
            if self.board[i] != Cell::Empty
                && self.board[i] == self.board[i + 3]
                && self.board[i + 3] == self.board[i + 6]
            {
                return Some(self.board[i]);
            }
        }

        // Check diagonals
        if self.board[0] != Cell::Empty
            && self.board[0] == self.board[4]
            && self.board[4] == self.board[8]
        {
            return Some(self.board[0]);
        }

        if self.board[2] != Cell::Empty
            && self.board[2] == self.board[4]
            && self.board[4] == self.board[6]
        {
            return Some(self.board[2]);
        }

        // Check for draw
        if !self.board.contains(&Cell::Empty) {
            return Some(Cell::Empty); // Use Empty to represent draw
        }

        None // Game continues
    }

    /// Gets valid moves
    pub fn get_valid_moves(&self) -> Vec<usize> {
        self.board
            .iter()
            .enumerate()
            .filter(|&(_, &cell)| cell == Cell::Empty)
            .map(|(i, _)| i)
            .collect()
    }

    /// Gets the computer's move using neural network
    pub fn get_computer_move(
        &mut self,
        nn: &mut NeuralNetwork,
        display_probs: bool,
    ) -> Option<usize> {
        let mut inputs = [0.0; NN_INPUT_SIZE];
        self.board_to_input(&mut inputs);
        nn.forward_pass(&inputs);

        let valid_moves = self.get_valid_moves();
        if valid_moves.is_empty() {
            return None;
        }

        let mut best_move = valid_moves[0];
        let mut best_prob = nn.outputs[best_move];

        // Find the best valid move
        for &mv in &valid_moves {
            if nn.outputs[mv] > best_prob {
                best_prob = nn.outputs[mv];
                best_move = mv;
            }
        }

        if display_probs {
            self.display_move_probabilities(nn, best_move);
        }

        Some(best_move)
    }

    /// Displays move probabilities in a grid format
    fn display_move_probabilities(&self, nn: &NeuralNetwork, best_move: usize) {
        println!("Move probabilities (NN):");

        for row in 0..3 {
            for col in 0..3 {
                let pos = row * 3 + col;
                let prob = nn.outputs[pos] * 100.0;

                print!("{:>5.1}", prob);

                if pos == best_move {
                    print!("*");
                } else {
                    print!(" ");
                }

                if col < 2 {
                    print!(" |");
                }
            }
            println!();
            if row < 2 {
                println!("      |      |      ");
            }
        }

        let total_prob: f64 = nn.outputs.iter().sum();
        println!("Total probability: {:.2}", total_prob);
    }

    /// Gets a random valid move
    pub fn get_random_move(&self) -> Option<usize> {
        let valid_moves = self.get_valid_moves();
        if valid_moves.is_empty() {
            return None;
        }

        let idx = (rand::random::<u32>() as usize) % valid_moves.len();
        Some(valid_moves[idx])
    }

    /// Resets the game state
    pub fn reset(&mut self) {
        self.board = [Cell::Empty; 9];
        self.current_player = Player::Human;
    }

    /// Checks if a position is valid and empty
    pub fn is_valid_move(&self, position: usize) -> bool {
        position < 9 && self.board[position] == Cell::Empty
    }
}

impl Default for GameState {
    fn default() -> Self {
        Self::new()
    }
}
