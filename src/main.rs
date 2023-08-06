use color_eyre::{eyre::eyre, Result};
use cv_convert::TryFromCv;
use ndarray::{Array2, Array3, Axis};
use opencv::{
    core::{self, Point, Range, VecN},
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
            break;
        }
    }

    VideoCapture::release(&mut cap)?;

    Ok(())
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

    // process the temperature map and draw a peak crosshair on image
    let temp_map = therm_map_to_array(thermdata)?;
    if let Some(max) =
        temp_map
            .indexed_iter()
            .max_by(|(_point, temp_a), (_, temp_b)| {
                temp_a.partial_cmp(temp_b).unwrap()
            })
    {
        println!("Max temperature: {max:?}");
        const LEN: i32 = 5;
        let ((x, y), _temp) = max;
        let x: i32 = x.try_into().unwrap();
        let y: i32 = y.try_into().unwrap();
        let width = 256i32;
        let height = 192i32;

        imgproc::line(
            &mut heatmap,
            Point::new(i32::max(0, x - LEN), y) * 3,
            Point::new(i32::min(x + LEN, width), y) * 3,
            VecN::new(255.0, 0.0, 0.0, 0.0),
            3,
            1,
            0,
        )?;
        imgproc::line(
            &mut heatmap,
            Point::new(x, i32::max(0, y - LEN)) * 3,
            Point::new(x, i32::min(y + LEN, height)) * 3,
            VecN::new(255.0, 0.0, 0.0, 0.0),
            3,
            1,
            0,
        )?;
    }

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
