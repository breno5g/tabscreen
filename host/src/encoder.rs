//! H.264 encoding with GStreamer: raw frames in through `appsrc`, Annex B
//! access units out through `appsink` onto a broadcast channel.

use std::sync::atomic::{AtomicU32, Ordering::Relaxed};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_allocators as gst_allocators;
use gstreamer_app as gst_app;
use gstreamer_video as gst_video;
use tokio::sync::broadcast;

#[derive(Clone, Copy, Debug, PartialEq, clap::ValueEnum)]
pub enum Codec {
    /// NVENC if available, else x264
    Auto,
    /// CPU (libx264)
    X264,
    /// NVIDIA NVENC
    Nvenc,
}

impl Codec {
    /// Pick a concrete encoder. The nvcodec plugin only registers `nvh264enc`
    /// when a usable NVIDIA GPU is present.
    pub fn resolve(self) -> Codec {
        match self {
            Codec::Auto if gst::ElementFactory::find("nvh264enc").is_some() => Codec::Nvenc,
            Codec::Auto => Codec::X264,
            other => other,
        }
    }
}

/// How frames get from the compositor to the encoder.
#[derive(Clone, Copy, Debug, PartialEq, clap::ValueEnum)]
pub enum Capture {
    /// Copy each frame through system memory: up to 120 fps, more CPU
    Cpu,
    /// Keep frames on the GPU (DMA-BUF to NVENC): less CPU, power and latency,
    /// but ~60 fps in fast motion because the GPU stays in its power-saving state
    Gpu,
}

/// A DRM format the GPU pipeline can import: fourcc and modifier.
pub type DrmFormat = (u32, u64);

#[derive(Debug)]
pub struct EncodedFrame {
    pub keyframe: bool,
    pub pts_us: u64,
    pub data: Vec<u8>,
}

pub struct Encoder {
    pipeline: gst::Pipeline,
    appsrc: gst_app::AppSrc,
    appsink: gst_app::AppSink,
    /// Last frame, re-encoded as a keyframe when a client needs one while
    /// the screen is static (Mutter only sends frames on damage).
    last: Mutex<Option<gst::Buffer>>,
    last_keyframe_request: Mutex<Option<Instant>>,
    /// Buffer timestamps are time since `epoch`.
    epoch: Instant,
    pushed: Arc<AtomicU32>,
    /// Most frames per second capture asks for, and the rate the encoder
    /// budgets bits for (bitrate / frame rate per frame).
    pub max_fps: u32,
}

/// Encode latency and throughput, logged every few seconds while frames flow.
struct Stats {
    since: Instant,
    pushed: Arc<AtomicU32>,
    frames: u32,
    bytes: usize,
    total: Duration,
    max: Duration,
}

impl Stats {
    const INTERVAL: Duration = Duration::from_secs(5);

    fn new(pushed: Arc<AtomicU32>) -> Self {
        Self { since: Instant::now(), pushed, frames: 0, bytes: 0, total: Duration::ZERO, max: Duration::ZERO }
    }

    fn add(&mut self, latency: Duration, bytes: usize) {
        self.frames += 1;
        self.bytes += bytes;
        self.total += latency;
        self.max = self.max.max(latency);
        let elapsed = self.since.elapsed();
        if elapsed >= Self::INTERVAL {
            let secs = elapsed.as_secs_f64();
            eprintln!(
                "encoder: {:.0} fps in, {:.0} fps out, {:.1} Mbit/s, encode latency avg {:.1} ms, max {:.1} ms",
                self.pushed.swap(0, Relaxed) as f64 / secs,
                self.frames as f64 / secs,
                self.bytes as f64 * 8.0 / secs / 1e6,
                self.total.as_secs_f64() * 1000.0 / self.frames as f64,
                self.max.as_secs_f64() * 1000.0
            );
            *self = Self::new(self.pushed.clone());
        }
    }
}

/// Keyframes are ~10× the size of other frames; don't let clients ask for them back to back.
const MIN_KEYFRAME_INTERVAL: Duration = Duration::from_millis(250);

impl Encoder {
    pub fn new(
        codec: Codec,
        capture: Capture,
        bitrate_kbps: u32,
        max_fps: u32,
        frames: broadcast::Sender<Arc<EncodedFrame>>,
    ) -> Result<Arc<Self>> {
        let nvenc = format!(
            "nvh264enc preset=p1 tune=ultra-low-latency zerolatency=true bframes=0 bitrate={bitrate_kbps} gop-size=600"
        );
        let encode = match (codec.resolve(), capture) {
            // NVENC takes BGRx/RGBx directly and converts on the GPU.
            (Codec::Nvenc, Capture::Cpu) => nvenc,
            // DMA-BUF → GL texture → NV12 → CUDA. Measured on the tablet: handing nvh264enc
            // the GL texture itself took 18 ms a frame; converting to NV12 in GL and copying
            // it to CUDA with cudaupload takes ~6 ms for the whole path.
            (Codec::Nvenc, Capture::Gpu) => format!(
                "glupload ! glcolorconvert ! video/x-raw(memory:GLMemory),format=NV12 \
                 ! cudaupload ! video/x-raw(memory:CUDAMemory) ! {nvenc}"
            ),
            (_, Capture::Gpu) => bail!("GPU capture needs NVENC"),
            _ => format!(
                "videoconvert n-threads=4 ! video/x-raw,format=I420 ! x264enc tune=zerolatency \
                 speed-preset=ultrafast bitrate={bitrate_kbps} key-int-max=600 bframes=0 sliced-threads=true"
            ),
        };
        let desc = format!(
            "appsrc name=src is-live=true format=time max-buffers=2 leaky-type=downstream \
             ! {encode} ! video/x-h264,profile=constrained-baseline \
             ! h264parse config-interval=-1 ! video/x-h264,stream-format=byte-stream,alignment=au \
             ! appsink name=sink sync=false max-buffers=4"
        );
        let pipeline = gst::parse::launch(&desc)
            .with_context(|| format!("building pipeline: {desc}"))?
            .downcast::<gst::Pipeline>()
            .unwrap();
        let appsrc = pipeline.by_name("src").unwrap().downcast::<gst_app::AppSrc>().unwrap();
        let appsink = pipeline.by_name("sink").unwrap().downcast::<gst_app::AppSink>().unwrap();

        let epoch = Instant::now();
        // When each frame left appsrc's queue. Encoders rewrite timestamps,
        // but they're 1:1 here, so frames come out in the same order.
        let in_flight: Arc<Mutex<VecDeque<Instant>>> = Arc::default();
        let entered = in_flight.clone();
        appsrc.static_pad("src").unwrap().add_probe(
            gst::PadProbeType::BUFFER,
            move |_, _| {
                entered.lock().unwrap().push_back(Instant::now());
                gst::PadProbeReturn::Ok
            },
        );
        let pushed = Arc::new(AtomicU32::new(0));
        let mut stats = Stats::new(pushed.clone());
        appsink.set_callbacks(
            gst_app::AppSinkCallbacks::builder()
                .new_sample(move |sink| {
                    let sample = sink.pull_sample().map_err(|_| gst::FlowError::Eos)?;
                    let buffer = sample.buffer().ok_or(gst::FlowError::Error)?;
                    let map = buffer.map_readable().map_err(|_| gst::FlowError::Error)?;
                    static FIRST: std::sync::Once = std::sync::Once::new();
                    FIRST.call_once(|| eprintln!("encoder: first encoded frame ({} bytes)", map.len()));
                    if let Some(entered) = in_flight.lock().unwrap().pop_front() {
                        stats.add(entered.elapsed(), map.len());
                    }
                    let _ = frames.send(Arc::new(EncodedFrame {
                        keyframe: !buffer.flags().contains(gst::BufferFlags::DELTA_UNIT),
                        pts_us: buffer.pts().map_or(0, |t| t.useconds()),
                        data: map.to_vec(),
                    }));
                    Ok(gst::FlowSuccess::Ok)
                })
                .build(),
        );

        let bus = pipeline.bus().unwrap();
        std::thread::Builder::new().name("gst-bus".into()).spawn(move || {
            for msg in bus.iter_timed(gst::ClockTime::NONE) {
                match msg.view() {
                    gst::MessageView::Error(e) => {
                        eprintln!("encoder: error from {:?}: {} ({:?})", e.src().map(|s| s.path_string()), e.error(), e.debug())
                    }
                    gst::MessageView::Warning(w) => eprintln!("encoder: warning: {}", w.error()),
                    gst::MessageView::Eos(_) => break,
                    _ => {}
                }
            }
        })?;

        pipeline.set_state(gst::State::Playing).context("starting encoder pipeline")?;
        Ok(Arc::new(Self { pipeline, appsrc, appsink, last: Mutex::new(None),
            last_keyframe_request: Mutex::new(None),
            epoch,
            pushed,
            max_fps,
        }))
    }

    pub fn set_format(&self, format: &str, width: u32, height: u32) {
        let caps = gst::Caps::builder("video/x-raw")
            .field("format", format)
            .field("width", width as i32)
            .field("height", height as i32)
            // Rate control spends bitrate / framerate per frame, so at the top
            // frame rate the stream matches --bitrate. A variable rate (0/1)
            // makes x264 hold back one frame, so the last change before the
            // screen goes static would never be sent.
            .field("framerate", gst::Fraction::new(self.max_fps as i32, 1))
            .build();
        self.appsrc.set_caps(Some(&caps));
    }

    /// DRM formats the GPU pipeline can import on this machine, as glupload
    /// reports them. Empty unless the encoder was built for [`Capture::Gpu`].
    pub fn dmabuf_formats(&self) -> Vec<DrmFormat> {
        let caps = self.appsrc.static_pad("src").unwrap().peer_query_caps(None);
        let mut names: Vec<String> = Vec::new();
        for s in caps.iter() {
            let Ok(value) = s.value("drm-format") else { continue };
            match value.get::<gst::List>() {
                Ok(list) => names.extend(list.iter().filter_map(|v| v.get::<String>().ok())),
                Err(_) => names.extend(value.get::<String>().ok()),
            }
        }
        let mut formats: Vec<DrmFormat> =
            names.iter().filter_map(|name| gst_video::dma_drm_fourcc_from_str(name).ok()).collect();
        formats.sort_unstable();
        formats.dedup();
        formats
    }

    /// Caps for DMA-BUF frames in DRM format `fourcc` with `modifier`.
    pub fn set_dmabuf_format(&self, (fourcc, modifier): DrmFormat, width: u32, height: u32) {
        let caps = gst::Caps::builder("video/x-raw")
            .features([gst_allocators::CAPS_FEATURE_MEMORY_DMABUF])
            .field("format", "DMA_DRM")
            .field("drm-format", gst_video::dma_drm_fourcc_to_string(fourcc, modifier).as_str())
            .field("width", width as i32)
            .field("height", height as i32)
            .field("framerate", gst::Fraction::new(self.max_fps as i32, 1))
            .build();
        self.appsrc.set_caps(Some(&caps));
    }

    pub fn push(&self, frame: Vec<u8>) {
        static FIRST: std::sync::Once = std::sync::Once::new();
        FIRST.call_once(|| eprintln!("encoder: first raw frame ({} bytes)", frame.len()));
        self.push_buffer(gst::Buffer::from_mut_slice(frame));
    }

    /// Encode `buffer`, and keep it to re-send when a keyframe is requested on a static screen.
    pub fn push_buffer(&self, mut buffer: gst::Buffer) {
        buffer.make_mut().set_pts(self.now());
        *self.last.lock().unwrap() = Some(buffer.clone());
        self.pushed.fetch_add(1, Relaxed);
        let _ = self.appsrc.push_buffer(buffer);
    }

    fn now(&self) -> gst::ClockTime {
        gst::ClockTime::from_nseconds(self.epoch.elapsed().as_nanos() as u64)
    }

    /// Make the next encoded frame an IDR with SPS/PPS, re-sending the last
    /// frame so one is produced even if the screen doesn't change.
    pub fn request_keyframe(&self) {
        {
            let mut last = self.last_keyframe_request.lock().unwrap();
            if last.is_some_and(|t| t.elapsed() < MIN_KEYFRAME_INTERVAL) {
                return; // the pending keyframe serves this request too
            }
            *last = Some(Instant::now());
        }
        let event = gst_video::UpstreamForceKeyUnitEvent::builder().all_headers(true).build();
        self.appsink.send_event(event);
        if let Some(last) = self.last.lock().unwrap().as_ref() {
            let mut buffer = last.copy();
            buffer.get_mut().unwrap().set_pts(self.now());
            let _ = self.appsrc.push_buffer(buffer);
        }
    }

    pub fn stop(&self) {
        let _ = self.pipeline.set_state(gst::State::Null);
    }
}
