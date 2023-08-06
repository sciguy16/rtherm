use color_eyre::{eyre::eyre, Result};
use opencv::{
    highgui,
    prelude::*,
    videoio::{self, VideoCapture},
};

mod argparse;

const WIN: &str = "rtherm";

fn main() -> Result<()> {
    color_eyre::install()?;

    let args = argparse::Args::parse();
    dbg!(&args);

    let cap = VideoCapture::from_file(
        args.device.to_str().unwrap(),
        videoio::CAP_ANY,
    )?;
    if !videoio::VideoCapture::is_opened(&cap)? {
        return Err(eyre!(
            "Unable to open camera at {}",
            args.device.display()
        ));
    }

    highgui::named_window(WIN, highgui::WINDOW_AUTOSIZE)?;

    capture_loop(cap)
}

fn capture_loop(mut cap: VideoCapture) -> Result<()> {
    let mut frame = Mat::default();

    loop {
        VideoCapture::read(&mut cap, &mut frame)?;

        if !frame.empty() {
            highgui::imshow(WIN, &frame)?;
        }
        let key = highgui::wait_key(10)?;
        if key > 0 && key != 255 {
            break Ok(());
        }
    }
}
