// VIRTIO GPU Emulation via vhost-user
//
//
// SPDX-License-Identifier: Apache-2.0 or BSD-3-Clause

pub mod vhu_gpu;
pub mod virtio_gpu;
pub mod protocol;
pub mod virt_gpu;

use log::{error, info};
use std::path::PathBuf;
use std::process::exit;
use std::sync::{Arc, RwLock};
use std::thread::{spawn, JoinHandle};

use clap::Parser;
use thiserror::Error as ThisError;
use vhost_user_backend::VhostUserDaemon;
use vm_memory::{GuestMemoryAtomic, GuestMemoryMmap};

use crate::vhu_gpu::VhostUserGpuBackend;
//use vhu_gpu::VhostUserGpuBackend;
use vhost_device_gpu::GpuConfig;

type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, ThisError)]
pub(crate) enum Error {
    #[error("Could not create backend: {0}")]
    CouldNotCreateBackend(vhu_gpu::Error),
    #[error("Could not create daemon: {0}")]
    CouldNotCreateDaemon(vhost_user_backend::Error),
    #[error("Fatal error: {0}")]
    ServeFailed(vhost_user_backend::Error),
}

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct GpuArgs {
    /// vhost-user Unix domain socket.
    #[clap(short, long, value_name = "SOCKET")]
    socket_path: PathBuf,
}


impl TryFrom<GpuArgs> for GpuConfig {
    type Error = Error;

    fn try_from(args: GpuArgs) -> Result<Self> {
        let socket_path = args.socket_path;

        Ok(GpuConfig::new(socket_path))
    }
}

fn start_backend(config: GpuConfig) -> Result<()> {

    let handle: JoinHandle<Result<()>> = spawn(move || loop {
        info!("Starting backend");
        // There isn't much value in complicating code here to return an error from the threads,
        // and so the code uses unwrap() instead. The panic on a thread won't cause trouble to the
        // main() function and should be safe for the daemon.
        let backend = Arc::new(RwLock::new(
            VhostUserGpuBackend::new(config.clone()).map_err(Error::CouldNotCreateBackend)?,
        ));

        let socket = config.get_socket_path();

        let mut daemon = VhostUserDaemon::new(
            String::from("vhost-device-gpu-backend"),
            backend,
            GuestMemoryAtomic::new(GuestMemoryMmap::new()),
        )
        .map_err(Error::CouldNotCreateDaemon)?;

        daemon.serve(socket).map_err(Error::ServeFailed)?;
    });

    handle.join().map_err(std::panic::resume_unwind).unwrap()
}

fn main() {
    env_logger::init();

    if let Err(e) = start_backend(GpuConfig::try_from(GpuArgs::parse()).unwrap()) {
        error!("{e}");
        exit(1);
    }
}

#[cfg(test)]
mod tests {
    use std::env;
    use assert_matches::assert_matches;
    use std::path::Path;
    use rutabaga_gfx::{RUTABAGA_CHANNEL_TYPE_WAYLAND, RutabagaChannel, RutabagaFenceHandler};

    use super::*;

    impl GpuArgs {
        pub(crate) fn from_args(path: &Path) -> GpuArgs {
            GpuArgs {
                socket_path: path.to_path_buf(),
            }
        }
    }

    #[test]
    fn test_parse_successful() {
        let socket_name = Path::new("vgpu.sock");

        let cmd_args = GpuArgs::from_args(socket_name);
        let config = GpuConfig::try_from(cmd_args).unwrap();

        assert_eq!(config.get_socket_path(), socket_name);
    }

    #[test]
    fn test_fail_listener() {
        // This will fail the listeners and thread will panic.
        let socket_name = Path::new("~/path/not/present/gpu");
        let cmd_args = GpuArgs::from_args(socket_name);
        let config = GpuConfig::try_from(cmd_args).unwrap();

        assert_matches!(start_backend(config).unwrap_err(), Error::ServeFailed(_));
    }
}
