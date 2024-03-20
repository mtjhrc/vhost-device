use rutabaga_gfx::{
    RutabagaBuilder, RutabagaChannel, RutabagaFenceHandler, RUTABAGA_CHANNEL_TYPE_WAYLAND,
};
use std::env;
use std::path::PathBuf;

// This is a weird test to have, but it is temporary, this code is copied from VirtioGpu::new
// but we cannot call that here because we don't have a vring
#[test]
fn initialize_rutabaga() {
    let xdg_runtime_dir = match env::var("XDG_RUNTIME_DIR") {
        Ok(dir) => dir,
        Err(_) => "/run/user/1000".to_string(),
    };
    let wayland_display = match env::var("WAYLAND_DISPLAY") {
        Ok(display) => display,
        Err(_) => "wayland-0".to_string(),
    };
    let path = PathBuf::from(format!("{}/{}", xdg_runtime_dir, wayland_display));

    let rutabaga_channels: Vec<RutabagaChannel> = vec![RutabagaChannel {
        base_channel: path,
        channel_type: RUTABAGA_CHANNEL_TYPE_WAYLAND,
    }];
    let rutabaga_channels_opt = Some(rutabaga_channels);

    let builder = RutabagaBuilder::new(rutabaga_gfx::RutabagaComponentType::VirglRenderer, 0)
        .set_rutabaga_channels(rutabaga_channels_opt)
        .set_use_egl(true)
        .set_use_gles(true)
        .set_use_glx(true)
        .set_use_surfaceless(true);
    // TODO: figure out if we need this:
    // this was part of libkrun modification and not upstream crossvm rutabaga
    //.set_use_drm(true);

    let fence_handler = RutabagaFenceHandler::new(|fence| {});
    let rutabaga = builder
        .build(fence_handler, None)
        .expect("Rutabaga initialization failed!");
}
