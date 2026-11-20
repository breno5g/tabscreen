//! HTTP + WebSocket server: serves the web client and speaks the protocol in PLAN.md.

use std::collections::{HashMap, VecDeque};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use axum::Router;
use axum::serve::ListenerExt;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{ConnectInfo, Path, Query, State};
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Redirect, Response};
use axum::routing::get;
use tokio::sync::broadcast;
use tokio::time::MissedTickBehavior;

use crate::capture::CursorImage;
use crate::encoder::EncodedFrame;
use crate::input::{BUTTON_LEFT, BUTTON_RIGHT, InputCmd, TouchAction};
use crate::session::{Manager, Session};

const MSG_HELLO: u8 = 0x01;
const MSG_CONFIG: u8 = 0x02;
const MSG_FRAME: u8 = 0x03;
const MSG_TOUCH: u8 = 0x04;
const MSG_KEYFRAME_REQUEST: u8 = 0x05;
const MSG_PING: u8 = 0x06;
const MSG_CURSOR_MOVE: u8 = 0x07;
const MSG_CURSOR_IMAGE: u8 = 0x08;
const MSG_FRAME_ACK: u8 = 0x09;
const MSG_POINTER: u8 = 0x0A;

/// Largest monitor we create; H.264 encoders top out around 4096 per side.
const MAX_SIDE: u32 = 4096;

/// How long a frame may go unacked before we stop sending that client more.
/// Past this its link is slower than the video: frames would pile up in TCP
/// buffers (seconds of delay), so we skip frames for it instead.
const MAX_FRAME_DELAY: Duration = Duration::from_millis(100);

/// Frames sent to one client that it hasn't acked yet, oldest first.
#[derive(Default)]
struct InFlight(VecDeque<SentFrame>);

struct SentFrame {
    /// When it was sent, or when the keyframe in front of it was acked.
    since: Instant,
    keyframe: bool,
}

impl InFlight {
    fn sent(&mut self, keyframe: bool, now: Instant) {
        self.0.push_back(SentFrame { since: now, keyframe });
    }

    /// Clients ack every frame, in order.
    fn acked(&mut self, now: Instant) {
        if self.0.pop_front().is_some_and(|f| f.keyframe) {
            // The frames behind a keyframe waited for it; don't count that against them.
            for f in &mut self.0 {
                f.since = now;
            }
        }
    }

    /// The oldest frame has waited too long. Keyframes are ~10× the size of
    /// other frames, so they get 10× as long.
    fn congested(&self, now: Instant) -> bool {
        self.0.front().is_some_and(|f| {
            let limit = if f.keyframe { 10 * MAX_FRAME_DELAY } else { MAX_FRAME_DELAY };
            now.duration_since(f.since) > limit
        })
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// One stream as the server sees it.
pub struct Stream {
    pub manager: Arc<Manager>,
    /// Required from non-local clients (Wi‑Fi). Local ones (USB via
    /// `adb reverse`, which arrive on loopback) don't need it. Each stream has
    /// its own token, so a link to one stream doesn't open another.
    pub token: Option<Arc<str>>,
}

/// All streams; the first one is also served at `/`.
type Streams = Arc<Vec<Stream>>;

async fn page() -> Html<&'static str> {
    Html(include_str!("../../web/index.html"))
}

fn router(streams: Streams) -> Router {
    Router::new()
        .route("/", get(page))
        .route(
            "/client.js",
            get(|| async { ([("content-type", "text/javascript")], include_str!("../../web/client.js")) }),
        )
        .route("/ws", get(default_ws))
        .route("/s/{name}", get(|Path(name): Path<String>| async move { Redirect::permanent(&format!("/s/{name}/")) }))
        .route("/s/{name}/", get(named_page))
        .route("/s/{name}/ws", get(named_ws))
        .with_state(streams)
}

/// Plain HTTP, for USB (`adb reverse` makes the tablet's localhost reach us).
pub async fn serve(addr: SocketAddr, streams: Streams) -> Result<()> {
    let app = router(streams).into_make_service_with_connect_info::<SocketAddr>();
    // Small messages (pongs, cursor moves) mustn't wait up to 40 ms for the ACK
    // of the previous frame (Nagle). axum-server already does this for Wi‑Fi.
    let listener = tokio::net::TcpListener::bind(addr).await?.tap_io(|tcp| {
        if let Err(e) = tcp.set_nodelay(true) {
            eprintln!("server: TCP_NODELAY: {e}");
        }
    });
    axum::serve(listener, app).await?;
    Ok(())
}

/// HTTPS, for Wi‑Fi.
pub async fn serve_tls(addr: SocketAddr, streams: Streams, access: &crate::wifi::Access) -> Result<()> {
    let config = axum_server::tls_rustls::RustlsConfig::from_pem(
        access.cert_pem.clone().into_bytes(),
        access.key_pem.clone().into_bytes(),
    )
    .await
    .context("loading TLS certificate")?;
    let app = router(streams).into_make_service_with_connect_info::<SocketAddr>();
    axum_server::bind_rustls(addr, config).serve(app).await.with_context(|| format!("HTTPS on {addr}"))?;
    Ok(())
}

async fn named_page(State(streams): State<Streams>, Path(name): Path<String>) -> Response {
    if streams.iter().any(|s| s.manager.name() == name) {
        page().await.into_response()
    } else {
        (StatusCode::NOT_FOUND, format!("no stream named {name:?}")).into_response()
    }
}

async fn default_ws(
    ws: WebSocketUpgrade,
    State(streams): State<Streams>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    Query(query): Query<HashMap<String, String>>,
) -> Response {
    let name = streams[0].manager.name().to_string();
    open(ws, streams, &name, peer, query)
}

async fn named_ws(
    ws: WebSocketUpgrade,
    State(streams): State<Streams>,
    Path(name): Path<String>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    Query(query): Query<HashMap<String, String>>,
) -> Response {
    open(ws, streams, &name, peer, query)
}

fn open(
    ws: WebSocketUpgrade,
    streams: Streams,
    name: &str,
    peer: SocketAddr,
    query: HashMap<String, String>,
) -> Response {
    let Some(stream) = streams.iter().find(|s| s.manager.name() == name) else {
        return (StatusCode::NOT_FOUND, format!("no stream named {name:?}")).into_response();
    };
    if let Some(token) = &stream.token
        && !peer.ip().to_canonical().is_loopback()
        && query.get("t").map(String::as_str) != Some(&**token)
    {
        eprintln!("{name}: rejected {peer}: missing or wrong token");
        return (StatusCode::UNAUTHORIZED, "missing or wrong token").into_response();
    }
    let manager = stream.manager.clone();
    ws.on_upgrade(move |socket| async move {
        let clients = manager.client_connected();
        eprintln!("{}: client {peer} connected ({clients} connected)", manager.name());
        if let Err(e) = client(socket, &manager, peer).await {
            // A tab closed or a cable pulled resets the connection; not worth an error.
            let reset = e.chain().any(|c| {
                let c = c.to_string();
                c.contains("without closing handshake") || c.contains("close_notify")
            });
            if !reset {
                eprintln!("{}: client error: {e:#}", manager.name());
            }
        }
        if let Some(session) = manager.subscribe().borrow().as_ref() {
            let _ = session.input.send(InputCmd::ReleaseAll);
        }
        manager.client_disconnected();
    })
    .into_response()
}

fn frame_message(frame: &EncodedFrame) -> Message {
    let mut msg = Vec::with_capacity(10 + frame.data.len());
    msg.push(MSG_FRAME);
    msg.push(frame.keyframe as u8);
    msg.extend_from_slice(&frame.pts_us.to_le_bytes());
    msg.extend_from_slice(&frame.data);
    Message::Binary(msg.into())
}

fn cursor_move_message((x, y): (i32, i32)) -> Message {
    let mut msg = vec![MSG_CURSOR_MOVE];
    msg.extend_from_slice(&x.to_le_bytes());
    msg.extend_from_slice(&y.to_le_bytes());
    Message::Binary(msg.into())
}

fn cursor_image_message(image: &CursorImage) -> Message {
    let mut msg = vec![MSG_CURSOR_IMAGE];
    for v in [image.width, image.height, image.hot_x, image.hot_y] {
        msg.extend_from_slice(&v.to_le_bytes());
    }
    msg.extend_from_slice(&image.rgba);
    Message::Binary(msg.into())
}

fn parse_touch(payload: &[u8]) -> Option<InputCmd> {
    let [action, slot, rest @ ..] = payload else { return None };
    let x = f32::from_le_bytes(rest.get(0..4)?.try_into().ok()?);
    let y = f32::from_le_bytes(rest.get(4..8)?.try_into().ok()?);
    let action = match action {
        0 => TouchAction::Down,
        1 => TouchAction::Move,
        2 => TouchAction::Up,
        3 => TouchAction::Cancel,
        _ => return None,
    };
    Some(InputCmd::Touch { action, slot: *slot as u32, x, y })
}

fn parse_pointer(payload: &[u8]) -> Option<InputCmd> {
    let [buttons, rest @ ..] = payload else { return None };
    let x = f32::from_le_bytes(rest.get(0..4)?.try_into().ok()?);
    let y = f32::from_le_bytes(rest.get(4..8)?.try_into().ok()?);
    // NaN would reach Mutter as a position; unknown button bits are dropped.
    (x.is_finite() && y.is_finite()).then_some(InputCmd::Pointer { buttons: buttons & (BUTTON_LEFT | BUTTON_RIGHT), x, y })
}

/// Screen size from a hello message, rounded down to even numbers (4:2:0 video).
fn parse_hello(name: &str, payload: &[u8]) -> Result<(u32, u32)> {
    let hello: serde_json::Value = serde_json::from_slice(payload).context("hello is not JSON")?;
    eprintln!("{name}: hello {hello}");
    let side = |key| {
        let v = hello[key].as_u64().with_context(|| format!("hello has no {key}"))?;
        anyhow::Ok((v as u32).clamp(64, MAX_SIDE) & !1)
    };
    Ok((side("w")?, side("h")?))
}

async fn acquire(manager: &Arc<Manager>, size: (u32, u32)) -> Result<Arc<Session>> {
    let manager = manager.clone();
    tokio::task::spawn_blocking(move || manager.acquire(size)).await?
}

/// Tell the client about a (new) session and get it a keyframe.
async fn start_session(socket: &mut WebSocket, session: &Session, touch: bool) -> Result<()> {
    let config = serde_json::json!({ "w": session.width, "h": session.height, "touch": touch });
    let mut msg = vec![MSG_CONFIG];
    msg.extend_from_slice(config.to_string().as_bytes());
    socket.send(Message::Binary(msg.into())).await?;
    session.encoder.request_keyframe();
    Ok(())
}

async fn client(mut socket: WebSocket, manager: &Arc<Manager>, peer: SocketAddr) -> Result<()> {
    // The client says how big its screen is before we create a monitor for it.
    let wanted = tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            match socket.recv().await {
                Some(Ok(Message::Binary(data))) if data.first() == Some(&MSG_HELLO) => break parse_hello(manager.name(), &data[1..]),
                Some(Ok(_)) => continue,
                Some(Err(e)) => break Err(e.into()),
                None => bail!("closed before hello"),
            }
        }
    })
    .await
    .context("no hello within 10s")??;
    let mut wanted = wanted;

    let mut sessions = manager.subscribe();
    let mut session = acquire(manager, wanted).await?;
    sessions.borrow_and_update();
    let mut frames = manager.frames.subscribe();
    let mut position = manager.cursor.position.subscribe();
    let mut image = manager.cursor.image.subscribe();

    start_session(&mut socket, &session, manager.touch_enabled()).await?;
    let (current_image, current_position) = (image.borrow_and_update().clone(), *position.borrow_and_update());
    socket.send(cursor_image_message(&current_image)).await?;
    socket.send(cursor_move_message(current_position)).await?;
    // Decoding can only start at a keyframe.
    let mut need_keyframe = true;
    let mut in_flight = InFlight::default();
    // The encoder drops a request made within 250 ms of another, and a keyframe
    // that came while this client was congested was skipped: keep asking.
    let mut keyframe_retry = tokio::time::interval(Duration::from_millis(250));
    keyframe_retry.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let mut skipped = 0u32;
    let mut skip_report: Option<Instant> = None;

    loop {
        tokio::select! {
            incoming = socket.recv() => {
                let Some(incoming) = incoming else { return Ok(()) };
                let Message::Binary(data) = incoming? else { continue };
                match data.first() {
                    Some(&MSG_TOUCH) if manager.touch_enabled() => {
                        if let Some(cmd) = parse_touch(&data[1..]) {
                            let _ = session.input.send(cmd);
                        }
                    }
                    Some(&MSG_POINTER) if manager.touch_enabled() => {
                        if let Some(cmd) = parse_pointer(&data[1..]) {
                            let _ = session.input.send(cmd);
                        }
                    }
                    Some(&MSG_KEYFRAME_REQUEST) => {
                        need_keyframe = true;
                        session.encoder.request_keyframe();
                    }
                    Some(&MSG_PING) => socket.send(Message::Binary(data)).await?,
                    Some(&MSG_FRAME_ACK) => in_flight.acked(Instant::now()),
                    // Sent again when the tablet rotates.
                    Some(&MSG_HELLO) => {
                        wanted = parse_hello(manager.name(), &data[1..])?;
                        let next = acquire(manager, wanted).await?;
                        sessions.borrow_and_update();
                        if !Arc::ptr_eq(&next, &session) {
                            session = next;
                            need_keyframe = true;
                            start_session(&mut socket, &session, manager.touch_enabled()).await?;
                        }
                    }
                    _ => {}
                }
            }
            Ok(()) = sessions.changed() => {
                // Another client resized the monitor, or Mutter ended the stream.
                let current = sessions.borrow_and_update().clone();
                let next = match current {
                    Some(next) => next,
                    None => acquire(manager, wanted).await?,
                };
                if !Arc::ptr_eq(&next, &session) {
                    session = next;
                    need_keyframe = true;
                    start_session(&mut socket, &session, manager.touch_enabled()).await?;
                }
            }
            frame = frames.recv() => match frame {
                Ok(frame) => {
                    let now = Instant::now();
                    if in_flight.congested(now) {
                        // Let it catch up, then resync with a keyframe: a lower
                        // frame rate for this client instead of a growing delay.
                        need_keyframe = true;
                        skipped += 1;
                        if skip_report.is_none_or(|t| t.elapsed() >= Duration::from_secs(5)) {
                            eprintln!("{}: client {peer} can't keep up, skipped {skipped} frames", manager.name());
                            (skipped, skip_report) = (0, Some(now));
                        }
                        continue;
                    }
                    if need_keyframe && !frame.keyframe {
                        continue;
                    }
                    need_keyframe = false;
                    in_flight.sent(frame.keyframe, now);
                    socket.send(frame_message(&frame)).await?;
                }
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    eprintln!("{}: client fell behind by {n} frames, waiting for a keyframe", manager.name());
                    need_keyframe = true;
                    session.encoder.request_keyframe();
                }
                Err(broadcast::error::RecvError::Closed) => return Ok(()),
            },
            _ = keyframe_retry.tick(), if need_keyframe && in_flight.is_empty() => {
                session.encoder.request_keyframe();
            }
            Ok(()) = position.changed() => {
                let pos = *position.borrow_and_update();
                socket.send(cursor_move_message(pos)).await?;
            }
            Ok(()) = image.changed() => {
                let current = image.borrow_and_update().clone();
                socket.send(cursor_image_message(&current)).await?;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pointer(buttons: u8, x: f32, y: f32) -> Vec<u8> {
        [vec![buttons], x.to_le_bytes().to_vec(), y.to_le_bytes().to_vec()].concat()
    }

    #[test]
    fn parses_pointer_and_rejects_bad_input() {
        let cmd = parse_pointer(&pointer(BUTTON_RIGHT | 0x80, 0.25, 0.5));
        assert!(matches!(cmd, Some(InputCmd::Pointer { buttons: BUTTON_RIGHT, x: 0.25, y: 0.5 })), "{cmd:?}");
        assert!(parse_pointer(&pointer(BUTTON_LEFT, f32::NAN, 0.5)).is_none());
        assert!(parse_pointer(&pointer(BUTTON_LEFT, 0.5, 0.5)[..8]).is_none(), "too short");
    }

    #[test]
    fn congested_when_a_frame_waits_too_long() {
        let t0 = Instant::now();
        let ms = |n| t0 + Duration::from_millis(n);
        let mut f = InFlight::default();
        f.acked(t0); // a stray ack is ignored
        assert!(f.is_empty() && !f.congested(t0));

        f.sent(false, t0);
        assert!(!f.congested(ms(100)));
        assert!(f.congested(ms(101)));
        f.acked(ms(101));
        assert!(f.is_empty() && !f.congested(ms(500)));

        // A keyframe gets 10× as long, and the frame behind it doesn't pay for that wait.
        f.sent(true, t0);
        f.sent(false, ms(10));
        assert!(!f.congested(ms(1000)));
        assert!(f.congested(ms(1001)));
        f.acked(ms(900));
        assert!(!f.congested(ms(1000)));
        assert!(f.congested(ms(1001)));
    }
}
