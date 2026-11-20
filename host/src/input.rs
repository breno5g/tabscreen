//! Touch and pointer injection on its own thread, so slow D-Bus calls never block the server.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};

use crate::display::Scale;
use crate::mutter::ScreenStream;

#[derive(Debug, Clone, Copy)]
pub enum TouchAction {
    Down,
    Move,
    Up,
    Cancel,
}

#[derive(Debug)]
pub enum InputCmd {
    /// `x, y` normalized to `[0, 1]` across the monitor.
    Touch { action: TouchAction, slot: u32, x: f32, y: f32 },
    /// Pen or mouse: move the pointer to `x, y` (normalized), then press or
    /// release buttons to match `buttons` ([`BUTTON_LEFT`] | [`BUTTON_RIGHT`]).
    Pointer { buttons: u8, x: f32, y: f32 },
    /// Lift every active finger and release the pointer's buttons (client disconnected).
    ReleaseAll,
}

pub const BUTTON_LEFT: u8 = 1;
pub const BUTTON_RIGHT: u8 = 2;
/// Our button bits and their evdev codes (`BTN_LEFT`, `BTN_RIGHT`).
const BUTTONS: [(u8, i32); 2] = [(BUTTON_LEFT, 0x110), (BUTTON_RIGHT, 0x111)];

pub fn spawn(vm: Arc<ScreenStream>, width: u32, height: u32, scale: Arc<Scale>) -> Sender<InputCmd> {
    let (tx, rx) = channel();
    std::thread::Builder::new()
        .name("input".into())
        .spawn(move || run(vm, width, height, &scale, rx))
        .expect("spawn input thread");
    tx
}

fn run(vm: Arc<ScreenStream>, width: u32, height: u32, scale: &Scale, rx: Receiver<InputCmd>) {
    let mut active = HashSet::new();
    let mut pressed = 0u8;
    let set_buttons = |changed: u8, buttons: u8| {
        BUTTONS
            .iter()
            .filter(|(bit, _)| changed & bit != 0)
            .try_for_each(|&(bit, code)| vm.pointer_button(code, buttons & bit != 0))
    };
    // Mutter takes logical pixels: device pixels divided by the monitor's scale.
    let to_px = |v: f32, size: u32| v.clamp(0.0, 1.0) as f64 * (size - 1) as f64 / scale.get();
    for cmd in rx {
        let result = match cmd {
            InputCmd::Touch { action, slot, x, y } => {
                let (x, y) = (to_px(x, width), to_px(y, height));
                match action {
                    TouchAction::Down => {
                        active.insert(slot);
                        vm.touch_down(slot, x, y)
                    }
                    TouchAction::Move if active.contains(&slot) => vm.touch_motion(slot, x, y),
                    TouchAction::Move => Ok(()),
                    TouchAction::Up | TouchAction::Cancel => {
                        if active.remove(&slot) { vm.touch_up(slot) } else { Ok(()) }
                    }
                }
            }
            InputCmd::Pointer { buttons, x, y } => {
                let changed = buttons ^ pressed;
                pressed = buttons;
                // Move first, so a click lands where the pen is.
                vm.pointer_motion_absolute(to_px(x, width), to_px(y, height)).and_then(|()| set_buttons(changed, buttons))
            }
            InputCmd::ReleaseAll => {
                let released = std::mem::take(&mut pressed);
                active.drain().try_for_each(|slot| vm.touch_up(slot)).and_then(|()| set_buttons(released, 0))
            }
        };
        if let Err(e) = result {
            eprintln!("input: {e:#}");
        }
    }
}
