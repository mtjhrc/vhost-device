pub mod virtio_gpu;
pub mod vhu_gpu;

use std::path::PathBuf;

use virtio_gpu::VirtioGpuCtrlHdr;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum GPUstate {
    GpuCmdStateNew,
    GpuCmdStatePending,
    GpuCmdStateFinished,
}
pub struct VirtioGpuCtrlCommand {
    pub cmd_hdr: VirtioGpuCtrlHdr,
    pub state: GPUstate,

}
#[derive(Debug, Clone)]
/// This structure is the public API through which an external program
/// is allowed to configure the backend.
pub struct GpuConfig {
    /// vhost-user Unix domain socket
    socket_path: PathBuf,
}

impl GpuConfig {
    /// Create a new instance of the GpuConfig struct, containing the
    /// parameters to be fed into the gpu-backend server.
    pub const fn new(socket_path: PathBuf) -> Self {
        Self {
            socket_path,
            //params,
        }
    }

    /// Return the path of the unix domain socket which is listening to
    /// requests from the guest.
    pub fn get_socket_path(&self) -> PathBuf {
        PathBuf::from(&self.socket_path)
    }

    // pub const fn get_audio_backend(&self) -> BackendType {
    //     self.audio_backend
    // }
}