use color_eyre::{eyre::eyre, Result};
use cv_convert::{FromCv, IntoCv, TryFromCv, TryIntoCv};
use ndarray::{Array, Array2, Array3, Axis};
use opencv::{
    core::{self, Range, Size, VecN, CV_8UC1},
    highgui,
    imgproc::{self, COLOR_YUV2BGR_YUYV, INTER_CUBIC},
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

    let imdata = Mat::rowscols(
        frame,
        &Range::new(0, height / 2)?,
        &Range::new(0, width)?,
    )?;
    let thermdata = Mat::rowscols(
        frame,
        &Range::new(height / 2, height)?,
        &Range::new(0, width)?,
    )?;

    let centre_pixel = {
        let px: &VecN<u8, 2> = thermdata.at_2d(96, 128)?;
        let temp = u32::from_be_bytes([0, 0, px[1], px[0]]);

        let temp = temp as f32;

        temp / 64.0 - 273.15
    };

    let temp_map = therm_map_to_array(thermdata)?;
    if let Some(max) =
        temp_map
            .indexed_iter()
            .max_by(|(_point, temp_a), (_, temp_b)| {
                temp_a.partial_cmp(temp_b).unwrap()
            })
    {
        println!("Max temperature: {max:?}");
    }

    let mut coloured = Mat::default();
    imgproc::cvt_color(&imdata, &mut coloured, COLOR_YUV2BGR_YUYV, 0)?;

    let mut scaled = Mat::default();
    core::convert_scale_abs(&coloured, &mut scaled, 1.0, 0.0)?;

    let mut resized = Mat::default();
    imgproc::resize(
        &scaled,
        &mut resized,
        Default::default(),
        3.0,
        3.0,
        INTER_CUBIC,
    )?;

    let mut heatmap = Mat::default();
    imgproc::apply_color_map(&resized, &mut heatmap, imgproc::COLORMAP_HOT)?;

    *frame = heatmap.clone();
    Ok(())
}

fn therm_map_to_array(thermdata: Mat) -> Result<Array2<f32>> {
    let thermdata = Array3::<u8>::try_from_cv(thermdata).unwrap();

    let mapped = thermdata.fold_axis(Axis(2), Vec::new(), |acc, ele| {
        let mut acc = acc.clone();
        acc.push(*ele);
        acc
    });

    let mut mapped =
        mapped.mapv(|ele| u32::from_be_bytes([0, 0, ele[1], ele[0]]) as f32);

    mapped
        .iter_mut()
        .for_each(|ele| *ele = *ele / 64.0 - 273.15);

    Ok(mapped)
}
