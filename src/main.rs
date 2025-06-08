pub mod game;
pub mod nn;

fn main() {
    let game = game::GameState::new();
    game.display_board();
}
