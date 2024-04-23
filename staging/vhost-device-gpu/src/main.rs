// VIRTIO GPU Emulation via vhost-user
//
//
// SPDX-License-Identifier: Apache-2.0 or BSD-3-Clause

use log::{error, info};
use std::path::PathBuf;
use std::process::exit;
use std::sync::{Arc, RwLock};

use clap::Parser;
use thiserror::Error as ThisError;
use vhost_user_backend::VhostUserDaemon;
use vm_memory::{GuestMemoryAtomic, GuestMemoryMmap};

use vhost_device_gpu::vhu_gpu;
use vhost_device_gpu::vhu_gpu::VhostUserGpuBackend;
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
    info!("Starting backend");
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
    Ok(())
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
    use assert_matches::assert_matches;
    use rutabaga_gfx::{RutabagaChannel, RutabagaFenceHandler, RUTABAGA_CHANNEL_TYPE_WAYLAND};
    use std::env;
    use std::path::Path;

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
