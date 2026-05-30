use std::io::{self, BufRead};
use std::process::{Child, Command};

/// Starts the `arecord` background process to record 16kHz mono audio.
pub fn start_recording(path: &str) -> io::Result<Child> {
    Command::new("arecord")
        .args(["-r", "16000", "-f", "S16_LE", "-c", "1", path])
        .spawn()
}

/// Prompts the user to press Enter to start recording, then spawns the process, and stops it when they press Enter again.
pub fn record_audio_to_file(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n>>> Press [Enter] to START recording...");
    let mut line = String::new();
    let stdin = io::stdin();
    stdin.lock().read_line(&mut line)?;

    let mut child = start_recording(path)?;
    println!(">>> Recording... Speak into your microphone.");
    println!(">>> Press [Enter] to STOP recording.");

    line.clear();
    stdin.lock().read_line(&mut line)?;

    let _ = child.kill();
    let _ = child.wait();
    println!(">>> Recording stopped. Saved to {}.\n", path);
    Ok(())
}

/// Plays a WAV audio file using the host `aplay` command.
pub fn play_audio_file(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!(">>> Playing audio file: {} ...", path);
    let status = Command::new("aplay")
        .arg(path)
        .status()?;
    if !status.success() {
        return Err("aplay command failed".into());
    }
    Ok(())
}
