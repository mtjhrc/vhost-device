// VirtIO GPIO definitions
//
// SPDX-License-Identifier: Apache-2.0 or BSD-3-Clause
use vm_memory::{ByteValued, Le32};

/// Virtio Gpu Feature bits
pub const VIRTIO_GPU_F_VIRGL: u32 = 0;
pub const VIRTIO_GPU_F_EDID: u32 = 1;
pub const _VIRTIO_GPU_F_RESOURCE_UUID: u32 = 2;
pub const _VIRTIO_GPU_F_RESOURCE_BLOB: u32 = 3;
pub const _VIRTIO_GPU_F_CONTEXT_INIT: u32 = 4;

pub const QUEUE_SIZE: usize = 1024;
pub const NUM_QUEUES: usize = 2;

pub const _VIRTIO_GPU_EVENT_DISPLAY: u32 = 1 << 0;

/// Virtio Gpu Configuration
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct VirtioGpuConfig {
    /// Signals pending events to the driver
    pub events_read: Le32,
    /// Clears pending events in the device
    pub events_clear: Le32,
    /// Maximum number of scanouts supported by the device
    pub num_scanouts: Le32,
    /// Maximum number of capability sets supported by the device
    pub num_capsets: Le32,
}

// SAFETY: The layout of the structure is fixed and can be initialized by
// reading its content from byte array.
unsafe impl ByteValued for VirtioGpuConfig {}

/* VIRTIO_GPU_RESP_OK_DISPLAY_INFO */
pub const VIRTIO_GPU_MAX_SCANOUTS: usize = 16;