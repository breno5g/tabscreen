//! Keeps `adb reverse` set up for every connected Android device, so the
//! tablet's `localhost:PORT` reaches us even after the cable is replugged
//! (adb forgets reverse rules when a device disconnects). When a device
//! appears and no page is connected, opens the page in Chrome on it.

use std::collections::{HashMap, HashSet};
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};

use crate::session::Manager;

const POLL: Duration = Duration::from_secs(2);
/// Time for a page that is already open to reconnect (it retries every second)
/// before we open one: reopening it would reload it and leave fullscreen.
const OPEN_GRACE: Duration = Duration::from_secs(4);

/// Serials of devices that are connected and authorized.
fn devices() -> Result<HashSet<String>> {
    let out = Command::new("adb").arg("devices").output().context("running adb (is it installed and on PATH?)")?;
    if !out.status.success() {
        bail!("adb devices: {}", String::from_utf8_lossy(&out.stderr).trim());
    }
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .skip(1)
        .filter_map(|line| line.split_once('\t'))
        .filter(|(_, state)| *state == "device")
        .map(|(serial, _)| serial.to_string())
        .collect())
}

fn reverse(serial: &str, port: u16) -> Result<()> {
    let spec = format!("tcp:{port}");
    let out = Command::new("adb").args(["-s", serial, "reverse", &spec, &spec]).output()?;
    if !out.status.success() {
        bail!("{}", String::from_utf8_lossy(&out.stderr).trim());
    }
    Ok(())
}

/// Opens the page in Chrome on the device. The application id makes Chrome
/// reuse the tab it opened last time instead of adding one per replug.
fn open_page(serial: &str, port: u16) -> Result<()> {
    let url = format!("http://localhost:{port}/");
    let out = Command::new("adb")
        .args(["-s", serial, "shell", "am", "start", "-a", "android.intent.action.VIEW", "-d", &url])
        .args(["-e", "com.android.browser.application_id", "tabscreen", "com.android.chrome"])
        .output()?;
    // `am start` reports some failures, such as Chrome not being installed, only in its output.
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    if !out.status.success() || text.contains("Error") {
        bail!("{}", text.trim());
    }
    Ok(())
}

/// `page` is the stream served at `/`, the page opened on new devices.
pub fn spawn(port: u16, page: Arc<Manager>) {
    std::thread::Builder::new()
        .name("adb".into())
        .spawn(move || {
            let mut forwarded: HashSet<String> = HashSet::new();
            let mut failed: HashSet<String> = HashSet::new();
            // Devices just forwarded, giving an open page time to reconnect.
            let mut opening: HashMap<String, Instant> = HashMap::new();
            loop {
                match devices() {
                    Ok(present) => {
                        for serial in forwarded.iter().filter(|s| !present.contains(*s)) {
                            eprintln!("usb: {serial} disconnected");
                        }
                        forwarded.retain(|s| present.contains(s));
                        failed.retain(|s| present.contains(s));
                        opening.retain(|s, _| present.contains(s));
                        for serial in present.difference(&forwarded.clone()) {
                            match reverse(serial, port) {
                                Ok(()) => {
                                    eprintln!("usb: {serial}: localhost:{port} on the device now reaches this host");
                                    forwarded.insert(serial.clone());
                                    opening.insert(serial.clone(), Instant::now());
                                }
                                // Log once per device, keep retrying quietly.
                                Err(e) if failed.insert(serial.clone()) => eprintln!("usb: {serial}: adb reverse failed: {e}"),
                                Err(_) => {}
                            }
                        }
                        // shortcut: USB clients all arrive from loopback, so a page open on one
                        // device keeps a second device from getting its own; fine for one tablet.
                        opening.retain(|serial, since| {
                            if since.elapsed() < OPEN_GRACE {
                                return true;
                            }
                            if page.clients() == 0 {
                                match open_page(serial, port) {
                                    Ok(()) => eprintln!("usb: {serial}: opened http://localhost:{port}/ in Chrome"),
                                    Err(e) => eprintln!("usb: {serial}: could not open the page in Chrome: {e:#}"),
                                }
                            }
                            false
                        });
                    }
                    Err(e) => {
                        eprintln!("usb: {e:#}; giving up on automatic port forwarding");
                        return;
                    }
                }
                std::thread::sleep(POLL);
            }
        })
        .expect("spawn adb thread");
}
