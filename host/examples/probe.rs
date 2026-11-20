//! Connect to one stream like a client would and report what arrives.
//!
//!   cargo run --example probe -- ws://localhost:8765/s/audience/ws 1280x800 [SECONDS] [KBPS]
//!
//! KBPS reads video no faster than that, like a viewer on a slow link. Pings
//! share the connection with video, so their round trip shows any queueing.

use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::tungstenite::Message;

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let url = args.next().context("usage: probe WS_URL WxH [SECONDS] [KBPS]")?;
    let size = args.next().unwrap_or_else(|| "1280x800".into());
    let seconds: u64 = args.next().map_or(Ok(4), |s| s.parse())?;
    let kbps: Option<u64> = args.next().map(|s| s.parse()).transpose()?;
    let (w, h) = size.split_once('x').context("size must be WxH")?;

    let (mut ws, _) = tokio_tungstenite::connect_async(&url).await.with_context(|| format!("connecting to {url}"))?;
    let hello = format!(r#"{{"w":{w},"h":{h},"codecs":["h264"],"version":1}}"#);
    ws.send(Message::Binary([&[0x01], hello.as_bytes()].concat().into())).await?;

    let (mut frames, mut keyframes, mut bytes, mut cursor) = (0, 0, 0, 0);
    let mut rtts = Vec::new();
    let start = Instant::now();
    let mut ping = tokio::time::interval(Duration::from_millis(250));
    let end = tokio::time::Instant::now() + Duration::from_secs(seconds);
    loop {
        let msg = tokio::select! {
            _ = ping.tick() => {
                let t_us = start.elapsed().as_micros() as u64;
                ws.send(Message::Binary([&[0x06], &t_us.to_le_bytes()[..]].concat().into())).await?;
                continue;
            }
            msg = tokio::time::timeout_at(end, ws.next()) => match msg {
                Ok(Some(msg)) => msg,
                _ => break,
            },
        };
        let Message::Binary(data) = msg? else { continue };
        match data[0] {
            0x02 => println!("config {}", String::from_utf8_lossy(&data[1..])),
            0x03 => {
                frames += 1;
                keyframes += (data[1] & 1) as u32;
                bytes += data.len();
                if let Some(kbps) = kbps {
                    tokio::time::sleep(Duration::from_micros(data.len() as u64 * 8000 / kbps)).await;
                }
                ws.send(Message::Binary([&[0x09], &data[2..10]].concat().into())).await?;
            }
            0x06 => {
                let sent = u64::from_le_bytes(data[1..9].try_into()?);
                rtts.push(start.elapsed().as_micros() as u64 - sent);
            }
            0x07 | 0x08 => cursor += 1,
            _ => {}
        }
    }
    println!("{frames} frames ({keyframes} keyframes, {} KiB), {cursor} cursor messages", bytes / 1024);
    if !rtts.is_empty() {
        rtts.sort_unstable();
        let ms = |us: u64| us as f64 / 1000.0;
        println!(
            "rtt over {} pings: median {:.1} ms, p90 {:.1} ms, max {:.1} ms",
            rtts.len(),
            ms(rtts[rtts.len() / 2]),
            ms(rtts[rtts.len() * 9 / 10]),
            ms(rtts[rtts.len() - 1])
        );
    }
    Ok(())
}
