pub mod virtio_gpu;
pub mod virt_gpu;
pub mod vhu_gpu;
pub mod protocol;

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

/// Interrupt flags (re: interrupt status & acknowledge registers).
/// See linux/virtio_mmio.h.
pub const VIRTIO_MMIO_INT_VRING: u32 = 0x01;
pub const VIRTIO_MMIO_INT_CONFIG: u32 = 0x02;

#[derive(Debug)]
pub enum GpuError {
    /// Failed to create event fd.
    EventFd(std::io::Error),
    /// Failed to decode incoming command.
    DecodeCommand(std::io::Error),
    /// Error creating Reader for Queue.
    // QueueReader(DescriptorError),
    /// Error creating Writer for Queue.
    // QueueWriter(DescriptorError),
    /// Error writting to the Queue.
    WriteDescriptor(std::io::Error),
    /// Error reading Guest Memory,
    GuestMemory,
}

//type Result<T> = std::result::Result<T, GpuError>;




#[cfg(target_os = "linux")]
pub struct Gic {}

#[cfg(target_os = "linux")]
impl Gic {
    pub fn set_irq(&mut self, _irq: u32) {}
}