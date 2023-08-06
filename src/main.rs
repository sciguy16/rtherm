use color_eyre::{eyre::eyre, Result};
use opencv::{
    core::{Range, VecN},
    highgui,
    imgproc::{self, COLOR_YUV2BGR_YUYV},
    prelude::*,
    videoio::{self, VideoCapture},
};

mod argparse;

const WIN: &str = "rtherm";

fn main() -> Result<()> {
    color_eyre::install()?;

    let args = argparse::Args::parse();
    dbg!(&args);

    let mut cap = VideoCapture::from_file(
        args.device.to_str().unwrap(),
        videoio::CAP_ANY,
    )?;
    if !videoio::VideoCapture::is_opened(&cap)? {
        return Err(eyre!(
            "Unable to open camera at {}",
            args.device.display()
        ));
    }

    cap.set(videoio::CAP_PROP_CONVERT_RGB, 0.0)?;

    highgui::named_window(WIN, highgui::WINDOW_AUTOSIZE)?;

    capture_loop(cap)
}

fn capture_loop(mut cap: VideoCapture) -> Result<()> {
    let mut frame = Mat::default();

    loop {
        VideoCapture::read(&mut cap, &mut frame)?;

        if !frame.empty() {
            process_frame(&mut frame)?;
            highgui::imshow(WIN, &frame)?;
        }
        let key = highgui::wait_key(10)?;
        if key > 0 && key != 255 {
            break Ok(());
        }
    }
}

fn process_frame(frame: &mut Mat) -> Result<()> {
    let width = frame.cols();
    let height = frame.rows();

    dbg!((width, height));

    let imdata = Mat::rowscols(
        frame,
        &Range::new(0, height / 2)?,
        &Range::new(0, width)?,
    )?;
    dbg!(());
    let thermdata = Mat::rowscols(
        frame,
        &Range::new(height / 2, height)?,
        &Range::new(0, width)?,
    )?;

    dbg!(());
    let centre_pixel = {
        dbg!(thermdata.channels());
        let px: &VecN<u8, 2> = thermdata.at_2d(96, 128)?;
        let temp = u32::from_be_bytes([0, 0, px[1], px[0]]);

        let temp = temp as f32;

        temp / 64.0 - 273.15
    };

    dbg!(centre_pixel);

    let mut disp = Mat::default();
    imgproc::cvt_color(&imdata, &mut disp, COLOR_YUV2BGR_YUYV, 0)?;

    *frame = disp.clone();
    Ok(())
}
