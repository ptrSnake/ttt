use std::env;

pub mod game;
pub mod nn;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    let random_games = if args.len() > 1 {
        args[1].parse::<u32>().unwrap_or(2_000_000)
    } else {
        2_000_000
    };

    println!("Creating neural network...");
    let mut nn = nn::NeuralNetwork::new();

    if random_games > 0 {
        println!(
            "Training against random player with {} games...",
            random_games
        );
        nn.train_against_random(random_games);
        println!("Training completed!");
    }

    loop {
        // Play a game
        if let Err(e) = nn.play_game() {
            println!("Error during game: {}", e);
            continue;
        }

        // Ask if player wants to play again
        println!("Play again? (y/n)");
        let mut input = String::new();
        std::io::stdin()
            .read_line(&mut input)
            .expect("Failed to read line");

        let play_again = input.trim().to_lowercase().chars().next().unwrap_or('n');

        if play_again != 'y' {
            break;
        }
    }

    println!("Thanks for playing!");
    Ok(())
}
