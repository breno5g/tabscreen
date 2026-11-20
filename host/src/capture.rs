//! PipeWire consumer for the virtual monitor's stream.
//!
//! Frames with pixel data go to the encoder; `SPA_META_Cursor` updates go to
//! the cursor channels. Cursor-only buffers are why this isn't `pipewiresrc`:
//! it drops them, see PLAN.md (M0 follow-up).
//!
//! Frames arrive in shared memory and are copied out ([`Transport::Shm`]), or as
//! DMA-BUF and are lent to the encoder without a copy ([`Transport::DmaBuf`]).

use std::cell::RefCell;
use std::collections::HashMap;
use std::mem::size_of;
use std::os::fd::BorrowedFd;
use std::rc::Rc;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use gstreamer as gst;
use gstreamer_allocators as gst_allocators;
use gstreamer_allocators::prelude::*;
use gstreamer_video as gst_video;
use pipewire as pw;
use pw::spa;
use pw::spa::param::video::VideoFormat;
use pw::spa::pod::deserialize::PodDeserializer;
use pw::spa::pod::{ChoiceValue, Object, Pod, PodPropFlags, Property, PropertyFlags, Value};
use pw::spa::sys as spa_sys;
use pw::spa::utils::{Choice, ChoiceEnum, ChoiceFlags, Fraction, Id, Rectangle};
use tokio::sync::watch;

use crate::encoder::{DrmFormat, Encoder};

/// Cursor image in RGBA. `width == 0` means the cursor is hidden.
#[derive(Debug, Default)]
pub struct CursorImage {
    pub width: u16,
    pub height: u16,
    pub hot_x: u16,
    pub hot_y: u16,
    pub rgba: Vec<u8>,
}

/// Latest cursor state, shared by every session's capture thread.
#[derive(Clone)]
pub struct CursorChannels {
    /// Cursor position in monitor pixels.
    pub position: Arc<watch::Sender<(i32, i32)>>,
    pub image: Arc<watch::Sender<Arc<CursorImage>>>,
}

impl Default for CursorChannels {
    fn default() -> Self {
        Self {
            position: Arc::new(watch::channel((0, 0)).0),
            image: Arc::new(watch::channel(Arc::new(CursorImage::default())).0),
        }
    }
}

/// How frames arrive from PipeWire.
#[derive(Clone, Debug)]
pub enum Transport {
    /// Shared memory, copied out for the encoder.
    Shm,
    /// DMA-BUF in one of these DRM formats (what the encoder can import).
    DmaBuf(Vec<DrmFormat>),
}

/// Packed RGB formats we accept: PipeWire's, GStreamer's name, and the DRM
/// fourcc with the same bytes in memory.
const FORMATS: [(VideoFormat, &str, &[u8; 4]); 4] = [
    (VideoFormat::BGRx, "BGRx", b"XR24"),
    (VideoFormat::BGRA, "BGRA", b"AR24"),
    (VideoFormat::RGBx, "RGBx", b"XB24"),
    (VideoFormat::RGBA, "RGBA", b"AB24"),
];

fn fourcc(code: &[u8; 4]) -> u32 {
    u32::from_le_bytes(*code)
}

const fn cursor_meta_size(w: usize, h: usize) -> i32 {
    (size_of::<spa_sys::spa_meta_cursor>() + size_of::<spa_sys::spa_meta_bitmap>() + w * h * 4) as i32
}

fn serialize(obj: Object) -> Vec<u8> {
    pw::spa::pod::serialize::PodSerializer::serialize(std::io::Cursor::new(Vec::new()), &Value::Object(obj))
        .expect("serialize pod")
        .0
        .into_inner()
}

/// A video format offer: one of `formats`, at up to 120 fps. `size`: the
/// virtual monitor size we ask for (a fixed size makes Mutter size the
/// monitor), or `None` to accept whatever the source has (a mirror).
fn format_object(size: Option<(u32, u32)>, formats: &[VideoFormat]) -> Object {
    use spa::param::format::{FormatProperties, MediaSubtype, MediaType};
    let mut format = pw::spa::pod::object!(
        spa::utils::SpaTypes::ObjectParamFormat,
        spa::param::ParamType::EnumFormat,
        pw::spa::pod::property!(FormatProperties::MediaType, Id, MediaType::Video),
        pw::spa::pod::property!(FormatProperties::MediaSubtype, Id, MediaSubtype::Raw),
        pw::spa::pod::property!(FormatProperties::VideoFramerate, Fraction, Fraction { num: 0, denom: 1 }),
        pw::spa::pod::property!(
            FormatProperties::VideoMaxFramerate,
            Choice,
            Range,
            Fraction,
            Fraction { num: 120, denom: 1 },
            Fraction { num: 1, denom: 1 },
            Fraction { num: 120, denom: 1 }
        ),
    );
    let ids: Vec<Id> = formats.iter().map(|f| Id(f.as_raw())).collect();
    format.properties.push(Property::new(
        spa_sys::SPA_FORMAT_VIDEO_format,
        Value::Choice(ChoiceValue::Id(Choice(ChoiceFlags::empty(), ChoiceEnum::Enum { default: ids[0], alternatives: ids }))),
    ));
    format.properties.push(match size {
        Some((width, height)) => {
            pw::spa::pod::property!(FormatProperties::VideoSize, Rectangle, Rectangle { width, height })
        }
        None => pw::spa::pod::property!(
            FormatProperties::VideoSize,
            Choice,
            Range,
            Rectangle,
            Rectangle { width: 1920, height: 1080 },
            Rectangle { width: 1, height: 1 },
            Rectangle { width: 8192, height: 8192 }
        ),
    });
    format
}

/// DMA-BUF modifier property: Mutter picks from `modifiers` (and we then fix
/// its choice), or the one already picked.
fn modifier_property(modifiers: &[u64], fixed: bool) -> Property {
    let as_long = |m: &u64| *m as i64;
    let (flags, value) = if fixed {
        (PropertyFlags::MANDATORY, Value::Long(as_long(&modifiers[0])))
    } else {
        let alternatives: Vec<i64> = modifiers.iter().map(as_long).collect();
        let choice = ChoiceEnum::Enum { default: alternatives[0], alternatives };
        (PropertyFlags::MANDATORY | PropertyFlags::DONT_FIXATE, Value::Choice(ChoiceValue::Long(Choice(ChoiceFlags::empty(), choice))))
    };
    Property { key: spa_sys::SPA_FORMAT_VIDEO_modifier, flags, value }
}

/// What we offer Mutter to connect: one param per format.
fn format_params(size: Option<(u32, u32)>, transport: &Transport) -> Vec<Vec<u8>> {
    match transport {
        Transport::Shm => vec![serialize(format_object(size, &FORMATS.map(|f| f.0)))],
        Transport::DmaBuf(drm) => FORMATS
            .iter()
            .filter_map(|&(format, _, code)| {
                let modifiers: Vec<u64> = drm.iter().filter(|(f, _)| *f == fourcc(code)).map(|&(_, m)| m).collect();
                if modifiers.is_empty() {
                    return None;
                }
                let mut object = format_object(size, &[format]);
                object.properties.push(modifier_property(&modifiers, false));
                Some(serialize(object))
            })
            .collect(),
    }
}

/// The modifiers Mutter can allocate, from a format it still wants us to fix
/// (`DONT_FIXATE`). `None` when the modifier is already fixed or missing.
fn modifiers_to_fixate(param: &Pod) -> Option<Vec<u64>> {
    let prop = param.as_object().ok()?.find_prop(Id(spa_sys::SPA_FORMAT_VIDEO_modifier))?;
    if !prop.flags().contains(PodPropFlags::DONT_FIXATE) {
        return None;
    }
    let (_, value) = PodDeserializer::deserialize_any_from(prop.value().as_bytes()).ok()?;
    Some(match value {
        Value::Choice(ChoiceValue::Long(Choice(_, ChoiceEnum::Enum { default, alternatives }))) => {
            std::iter::once(default).chain(alternatives).map(|m| m as u64).collect()
        }
        Value::Long(m) => vec![m as u64],
        _ => return None,
    })
}

fn has_modifier(param: &Pod) -> bool {
    param.as_object().is_ok_and(|o| o.find_prop(Id(spa_sys::SPA_FORMAT_VIDEO_modifier)).is_some())
}

fn cursor_meta_param() -> Vec<u8> {
    serialize(Object {
        type_: spa_sys::SPA_TYPE_OBJECT_ParamMeta,
        id: spa_sys::SPA_PARAM_Meta,
        properties: vec![
            Property::new(spa_sys::SPA_PARAM_META_type, Value::Id(Id(spa_sys::SPA_META_Cursor))),
            Property::new(
                spa_sys::SPA_PARAM_META_size,
                Value::Choice(ChoiceValue::Int(Choice(
                    ChoiceFlags::empty(),
                    ChoiceEnum::Range {
                        default: cursor_meta_size(64, 64),
                        min: cursor_meta_size(1, 1),
                        max: cursor_meta_size(384, 384),
                    },
                ))),
            ),
        ],
    })
}

fn dmabuf_buffers_param() -> Vec<u8> {
    let dmabuf = 1 << spa_sys::SPA_DATA_DmaBuf;
    serialize(Object {
        type_: spa_sys::SPA_TYPE_OBJECT_ParamBuffers,
        id: spa_sys::SPA_PARAM_Buffers,
        properties: vec![Property::new(
            spa_sys::SPA_PARAM_BUFFERS_dataType,
            Value::Choice(ChoiceValue::Int(Choice(ChoiceFlags::empty(), ChoiceEnum::Flags { default: dmabuf, flags: vec![dmabuf] }))),
        )],
    })
}

/// Convert a cursor bitmap to RGBA, or `None` for formats we don't handle.
fn bitmap_to_rgba(format: u32, width: usize, height: usize, stride: usize, src: *const u8) -> Option<Vec<u8>> {
    // Byte positions of R, G, B, A within a pixel.
    let order: [usize; 4] = match format {
        spa_sys::SPA_VIDEO_FORMAT_RGBA => [0, 1, 2, 3],
        spa_sys::SPA_VIDEO_FORMAT_BGRA => [2, 1, 0, 3],
        spa_sys::SPA_VIDEO_FORMAT_ARGB => [1, 2, 3, 0],
        spa_sys::SPA_VIDEO_FORMAT_ABGR => [3, 2, 1, 0],
        _ => return None,
    };
    let mut rgba = Vec::with_capacity(width * height * 4);
    for row in 0..height {
        let line = unsafe { std::slice::from_raw_parts(src.add(row * stride), width * 4) };
        for px in line.chunks_exact(4) {
            rgba.extend(order.iter().map(|&i| px[i]));
        }
    }
    Some(rgba)
}

fn read_cursor(buf: &spa_sys::spa_buffer, cursor: &CursorChannels) {
    for i in 0..buf.n_metas as usize {
        let meta = unsafe { &*buf.metas.add(i) };
        if meta.type_ != spa_sys::SPA_META_Cursor || meta.size < size_of::<spa_sys::spa_meta_cursor>() as u32 {
            continue;
        }
        let c = unsafe { &*(meta.data as *const spa_sys::spa_meta_cursor) };
        if c.id == 0 {
            continue; // no new cursor data
        }
        cursor.position.send_if_modified(|pos| {
            let new = (c.position.x, c.position.y);
            std::mem::replace(pos, new) != new
        });
        if (c.bitmap_offset as usize) < size_of::<spa_sys::spa_meta_cursor>() {
            continue; // position-only update
        }
        let base = unsafe { (meta.data as *const u8).add(c.bitmap_offset as usize) };
        let bitmap = unsafe { &*(base as *const spa_sys::spa_meta_bitmap) };
        let image = if bitmap.format == 0 || (bitmap.offset as usize) < size_of::<spa_sys::spa_meta_bitmap>() {
            CursorImage::default()
        } else {
            let (w, h) = (bitmap.size.width as usize, bitmap.size.height as usize);
            match bitmap_to_rgba(bitmap.format, w, h, bitmap.stride as usize, unsafe { base.add(bitmap.offset as usize) }) {
                Some(rgba) => CursorImage {
                    width: w as u16,
                    height: h as u16,
                    hot_x: c.hotspot.x.max(0) as u16,
                    hot_y: c.hotspot.y.max(0) as u16,
                    rgba,
                },
                None => {
                    eprintln!("capture: unsupported cursor bitmap format {}", bitmap.format);
                    continue;
                }
            }
        };
        cursor.image.send_replace(Arc::new(image));
    }
}

/// The buffer's planes, or `None` for a buffer without pixels (cursor only).
fn frame_planes(buf: &spa_sys::spa_buffer) -> Option<&[spa_sys::spa_data]> {
    if buf.n_datas == 0 || buf.datas.is_null() {
        return None;
    }
    let datas = unsafe { std::slice::from_raw_parts(buf.datas, buf.n_datas as usize) };
    let chunk = unsafe { &*datas.first()?.chunk };
    (chunk.size > 0 && chunk.flags & spa_sys::SPA_CHUNK_FLAG_CORRUPTED as i32 == 0).then_some(datas)
}

/// Copy the frame's pixels into a tightly packed buffer, if the buffer has any.
fn read_frame(buf: &spa_sys::spa_buffer, width: usize, height: usize) -> Option<Vec<u8>> {
    let data = frame_planes(buf)?.first()?;
    if data.data.is_null() {
        return None;
    }
    let chunk = unsafe { &*data.chunk };
    let row = width * 4;
    let stride = if chunk.stride > 0 { chunk.stride as usize } else { row };
    if (chunk.offset as usize) + stride * (height - 1) + row > data.maxsize as usize {
        eprintln!("capture: frame smaller than {width}x{height}, skipping");
        return None;
    }
    let src = unsafe { (data.data as *const u8).add(chunk.offset as usize) };
    let mut out = Vec::with_capacity(row * height);
    for y in 0..height {
        out.extend_from_slice(unsafe { std::slice::from_raw_parts(src.add(y * stride), row) });
    }
    Some(out)
}

/// Hands a PipeWire buffer back to the stream once GStreamer drops the last
/// frame that uses it. Rides along as the frame's `ParentBufferMeta`.
struct Release {
    sender: pw::channel::Sender<usize>,
    buffer: usize,
    bytes: [u8; 1],
}

impl AsRef<[u8]> for Release {
    fn as_ref(&self) -> &[u8] {
        &self.bytes
    }
}

impl Drop for Release {
    fn drop(&mut self) {
        let _ = self.sender.send(self.buffer);
    }
}

/// DMA-BUF frames lent to the encoder: GStreamer memories for each PipeWire
/// buffer's planes, made once per buffer, so GL can keep its imported image.
/// Only buffers listed here are live; others were removed by PipeWire.
#[derive(Default)]
struct Lent {
    memories: HashMap<usize, Vec<gst::Memory>>,
}

impl Lent {
    /// The frame in `pw_buf` as a GStreamer buffer that shares its DMA-BUF
    /// planes, or `None` when it has no pixels. `pw_buf` goes back to PipeWire
    /// through `release` once GStreamer is done with it.
    fn frame(
        &mut self,
        pw_buf: *mut pw::sys::pw_buffer,
        (width, height): (usize, usize),
        allocator: &gst_allocators::DmaBufAllocator,
        release: &pw::channel::Sender<usize>,
    ) -> Result<Option<gst::Buffer>> {
        let buf = unsafe { &*(*pw_buf).buffer };
        let Some(planes) = frame_planes(buf) else { return Ok(None) };
        if planes[0].type_ != spa_sys::SPA_DATA_DmaBuf {
            bail!("expected a DMA-BUF, got data type {}", planes[0].type_);
        }
        let memories = match self.memories.entry(pw_buf as usize) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                let mut memories = Vec::with_capacity(planes.len());
                for plane in planes {
                    if plane.fd < 0 {
                        bail!("DMA-BUF plane without a file descriptor");
                    }
                    // Our own fd: the memory may outlive PipeWire's buffer.
                    let fd = unsafe { BorrowedFd::borrow_raw(plane.fd as i32) }.try_clone_to_owned()?;
                    memories.push(unsafe { allocator.alloc_dmabuf(fd, plane.maxsize as usize) }?);
                }
                e.insert(memories)
            }
        };
        let mut frame = gst::Buffer::new();
        {
            let frame = frame.get_mut().unwrap();
            let (mut offsets, mut strides, mut base) = (Vec::new(), Vec::new(), 0);
            for (plane, memory) in planes.iter().zip(memories.iter()) {
                let chunk = unsafe { &*plane.chunk };
                offsets.push(base + chunk.offset as usize);
                strides.push(chunk.stride);
                base += memory.size();
                frame.append_memory(memory.clone());
            }
            gst_video::VideoMeta::add_full(
                frame,
                gst_video::VideoFrameFlags::empty(),
                gst_video::VideoFormat::DmaDrm,
                width as u32,
                height as u32,
                &offsets,
                &strides,
            )?;
            let guard = Release { sender: release.clone(), buffer: pw_buf as usize, bytes: [0] };
            gst::ParentBufferMeta::add(frame, &gst::Buffer::from_slice(guard));
        }
        Ok(Some(frame))
    }
}

/// Frame rate from PipeWire and time spent copying (or wrapping) each frame,
/// logged every few seconds.
struct Stats {
    since: Instant,
    frames: u32,
    work: Duration,
    what: &'static str,
}

impl Stats {
    fn new(what: &'static str) -> Self {
        Self { since: Instant::now(), frames: 0, work: Duration::ZERO, what }
    }

    fn add(&mut self, work: Duration) {
        self.frames += 1;
        self.work += work;
        let elapsed = self.since.elapsed();
        if elapsed >= Duration::from_secs(5) {
            eprintln!(
                "capture: {:.0} fps from PipeWire, {} avg {:.1} ms",
                self.frames as f64 / elapsed.as_secs_f64(),
                self.what,
                self.work.as_secs_f64() * 1000.0 / self.frames as f64
            );
            *self = Self::new(self.what);
        }
    }
}

/// Run the PipeWire consumer on its own thread until the stream goes away.
/// Every negotiated frame size is sent on `negotiated`.
pub fn spawn(
    node_id: u32,
    size: Option<(u32, u32)>,
    transport: Transport,
    encoder: Arc<Encoder>,
    cursor: CursorChannels,
    negotiated: std::sync::mpsc::Sender<(u32, u32)>,
) -> JoinHandle<Result<()>> {
    std::thread::Builder::new()
        .name("pipewire".into())
        .spawn(move || run(node_id, size, transport, encoder, cursor, negotiated))
        .expect("spawn pipewire thread")
}

fn run(
    node_id: u32,
    size: Option<(u32, u32)>,
    transport: Transport,
    encoder: Arc<Encoder>,
    cursor: CursorChannels,
    negotiated: std::sync::mpsc::Sender<(u32, u32)>,
) -> Result<()> {
    pw::init();
    let mainloop = pw::main_loop::MainLoopRc::new(None)?;
    let context = pw::context::ContextRc::new(&mainloop, None)?;
    let core = context.connect_rc(None)?;
    let stream = pw::stream::StreamBox::new(
        &core,
        "tabscreen",
        pw::properties::properties! {
            *pw::keys::MEDIA_TYPE => "Video",
            *pw::keys::MEDIA_CATEGORY => "Capture",
            *pw::keys::MEDIA_ROLE => "Screen",
        },
    )?;

    let dmabuf = matches!(transport, Transport::DmaBuf(_));
    let lent = Rc::new(RefCell::new(Lent::default()));
    let allocator = gst_allocators::DmaBufAllocator::new();
    // Frames come back from GStreamer threads; requeue them on this loop.
    let (release, released) = pw::channel::channel::<usize>();
    let stream_ptr = stream.as_raw_ptr();
    let _released = released.attach(mainloop.loop_(), {
        let lent = lent.clone();
        move |buffer| {
            // A buffer PipeWire removed meanwhile (renegotiation) is gone.
            if lent.borrow().memories.contains_key(&buffer) {
                unsafe { pw::sys::pw_stream_queue_buffer(stream_ptr, buffer as *mut pw::sys::pw_buffer) };
            }
        }
    });

    let meta = cursor_meta_param();
    let buffers = dmabuf_buffers_param();
    let format_encoder = encoder.clone();
    let format_transport = transport.clone();
    let removed = lent.clone();
    let weak = mainloop.downgrade();
    let _listener = stream
        .add_local_listener_with_user_data(((0usize, 0usize), Stats::new(if dmabuf { "wrap" } else { "copy" })))
        .state_changed(move |_, _, old, new| {
            eprintln!("capture: stream {old:?} -> {new:?}");
            if matches!(new, pw::stream::StreamState::Unconnected | pw::stream::StreamState::Error(_))
                && let Some(ml) = weak.upgrade() {
                    ml.quit();
                }
        })
        .param_changed(move |stream, size, id, param| {
            let Some(param) = param else { return };
            if id != spa::param::ParamType::Format.as_raw() {
                return;
            }
            let mut info = spa::param::video::VideoInfoRaw::default();
            if info.parse(param).is_err() {
                return;
            }
            let (w, h) = (info.size().width, info.size().height);
            let Some(&(format, name, code)) = FORMATS.iter().find(|f| f.0 == info.format()) else {
                eprintln!("capture: unexpected format {:?}", info.format());
                return;
            };
            let mut params: Vec<Vec<u8>> = vec![meta.clone()];
            match &format_transport {
                Transport::Shm => {
                    eprintln!("capture: negotiated {name} {w}x{h}");
                    format_encoder.set_format(name, w, h);
                }
                Transport::DmaBuf(drm) => {
                    if !has_modifier(param) {
                        eprintln!("capture: Mutter sent {name} {w}x{h} without DMA-BUF; try --capture cpu");
                        return;
                    }
                    if let Some(offered) = modifiers_to_fixate(param) {
                        // Mutter's first choice that we can import.
                        let Some(&modifier) = offered.iter().find(|m| drm.contains(&(fourcc(code), **m))) else {
                            eprintln!("capture: no DMA-BUF modifier in common for {name}; try --capture cpu");
                            return;
                        };
                        let mut fixed = format_object(Some((w, h)), &[format]);
                        fixed.properties.push(modifier_property(&[modifier], true));
                        let fixed = serialize(fixed);
                        if let Err(e) = stream.update_params(&mut [Pod::from_bytes(&fixed).unwrap()]) {
                            eprintln!("capture: update_params: {e}");
                        }
                        return; // Mutter sends the fixed format next
                    }
                    let drm_format = (fourcc(code), info.modifier());
                    eprintln!(
                        "capture: negotiated {w}x{h} as DMA-BUF {}",
                        gst_video::dma_drm_fourcc_to_string(drm_format.0, drm_format.1)
                    );
                    format_encoder.set_dmabuf_format(drm_format, w, h);
                    params.push(buffers.clone());
                }
            }
            size.0 = (w as usize, h as usize);
            let _ = negotiated.send((w, h));
            let mut pods: Vec<&Pod> = params.iter().map(|p| Pod::from_bytes(p).unwrap()).collect();
            if let Err(e) = stream.update_params(&mut pods) {
                eprintln!("capture: update_params: {e}");
            }
        })
        .remove_buffer(move |_, _, buffer| {
            removed.borrow_mut().memories.remove(&(buffer as usize));
        })
        .process(move |stream, (size, stats)| unsafe {
            let pw_buf = stream.dequeue_raw_buffer();
            if pw_buf.is_null() {
                return;
            }
            let buf = &*(*pw_buf).buffer;
            read_cursor(buf, &cursor);
            let start = Instant::now();
            if size.0 > 0 && dmabuf {
                let frame = lent.borrow_mut().frame(pw_buf, *size, &allocator, &release);
                match frame {
                    Ok(Some(frame)) => {
                        stats.add(start.elapsed());
                        encoder.push_buffer(frame);
                        return; // requeued by `Release` once the encoder is done with it
                    }
                    Ok(None) => {}
                    Err(e) => eprintln!("capture: {e:#}"),
                }
            } else if size.0 > 0
                && let Some(frame) = read_frame(buf, size.0, size.1)
            {
                stats.add(start.elapsed());
                encoder.push(frame);
            }
            stream.queue_raw_buffer(pw_buf);
        })
        .register()?;

    let formats = format_params(size, &transport);
    if formats.is_empty() {
        bail!("the GPU can't import a DMA-BUF in any format we use; try --capture cpu");
    }
    let mut pods: Vec<&Pod> = formats.iter().map(|f| Pod::from_bytes(f).unwrap()).collect();
    stream.connect(
        spa::utils::Direction::Input,
        Some(node_id),
        pw::stream::StreamFlags::AUTOCONNECT | pw::stream::StreamFlags::MAP_BUFFERS,
        &mut pods,
    )?;
    mainloop.run();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drm_fourccs_match_gstreamer() {
        gst::init().unwrap();
        for (_, name, code) in FORMATS {
            let gst_format = gst_video::VideoFormat::from_string(name);
            assert_eq!(gst_video::dma_drm_fourcc_from_format(gst_format).unwrap(), fourcc(code), "{name}");
        }
    }

    #[test]
    fn offers_only_formats_with_modifiers() {
        let drm = vec![(fourcc(b"AR24"), 0x0300_0000_0060_6012), (fourcc(b"AR24"), 0), (fourcc(b"NV12"), 0)];
        assert_eq!(format_params(None, &Transport::DmaBuf(drm)).len(), 1, "only BGRA (AR24) has modifiers");
        assert_eq!(format_params(None, &Transport::Shm).len(), 1);
        assert!(format_params(None, &Transport::DmaBuf(vec![])).is_empty());
    }

    #[test]
    fn reads_back_modifiers_to_fixate() {
        let mut object = format_object(Some((64, 64)), &[VideoFormat::BGRA]);
        object.properties.push(modifier_property(&[7, 0], false));
        let bytes = serialize(object);
        assert_eq!(modifiers_to_fixate(Pod::from_bytes(&bytes).unwrap()), Some(vec![7, 7, 0]));

        let mut object = format_object(Some((64, 64)), &[VideoFormat::BGRA]);
        object.properties.push(modifier_property(&[7], true));
        let bytes = serialize(object);
        let pod = Pod::from_bytes(&bytes).unwrap();
        assert_eq!(modifiers_to_fixate(pod), None, "already fixed");
        assert!(has_modifier(pod));
    }
}
