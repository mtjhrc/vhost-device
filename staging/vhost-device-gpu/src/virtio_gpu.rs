// VirtIO GPIO definitions
//
// SPDX-License-Identifier: Apache-2.0 or BSD-3-Clause
use vm_memory::{ByteValued, Le32, Le64};

/// Virtio Gpu Feature bits
pub const VIRTIO_GPU_F_VIRGL: u32 = 0;
pub const VIRTIO_GPU_F_EDID: u32 = 1;
pub const _VIRTIO_GPU_F_RESOURCE_UUID: u32 = 2;
pub const _VIRTIO_GPU_F_RESOURCE_BLOB: u32 = 3;
pub const _VIRTIO_GPU_F_CONTEXT_INIT: u32 = 4;

pub const QUEUE_SIZE: usize = 1024;
pub const NUM_QUEUES: usize = 2;

pub const CONTROL_QUEUE: u16 = 0;
pub const CURSOR_QUEUE: u16 = 1;

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

/// Virtio GPU Request / Response common header
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct VirtioGpuCtrlHdr {
    /// request type / device response
    pub gpu_type: Le32,
    /// request / response flags
    pub flags: Le32,
    /// request / response fence_id field
    pub fence_id: Le64,
    /// Rendering context (used in 3D mode only)
    pub ctx_id: Le32,
    /// request / response ring index
    pub ring_idx: u8,
    pub padding: [u8; 3],
}
// SAFETY: The layout of the structure is fixed and can be initialized by
// reading its content from byte array.
unsafe impl ByteValued for VirtioGpuCtrlHdr {}

/* VIRTIO_GPU_CMD_RESOURCE_CREATE_2D: create a 2d resource with a format */
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct virtioGpuResourceCreate2d {
    pub hdr: VirtioGpuCtrlHdr,
    pub resource_id: Le32,
    pub format: Le32,
    pub width: Le32,
    pub height: Le32,
}

/* VIRTIO_GPU_RESP_OK_DISPLAY_INFO */
pub const VIRTIO_GPU_MAX_SCANOUTS: usize = 16;

/* 2d commands */
pub const VIRTIO_GPU_CMD_GET_DISPLAY_INFO: u32 = 0x100;
pub const VIRTIO_GPU_CMD_RESOURCE_CREATE_2D: u32 = 0x101;
pub const VIRTIO_GPU_CMD_RESOURCE_UNREF: u32 = 0x102;
pub const VIRTIO_GPU_CMD_SET_SCANOUT: u32 = 0x103;
pub const VIRTIO_GPU_CMD_RESOURCE_FLUSH: u32 = 0x104;
pub const VIRTIO_GPU_CMD_TRANSFER_TO_HOST_2D: u32 = 0x105;
pub const VIRTIO_GPU_CMD_RESOURCE_ATTACH_BACKING: u32 = 0x106;
pub const VIRTIO_GPU_CMD_RESOURCE_DETACH_BACKING: u32 = 0x107;
pub const VIRTIO_GPU_CMD_GET_CAPSET_INFO: u32 = 0x108;
pub const VIRTIO_GPU_CMD_GET_CAPSET: u32 = 0x109;
pub const VIRTIO_GPU_CMD_GET_EDID: u32 = 0x10a;
pub const VIRTIO_GPU_CMD_RESOURCE_ASSIGN_UUID: u32 = 0x10b;
pub const VIRTIO_GPU_CMD_RESOURCE_CREATE_BLOB: u32 = 0x10c;
pub const VIRTIO_GPU_CMD_SET_SCANOUT_BLOB: u32 = 0x10d;

/* 3d commands */
pub const VIRTIO_GPU_CMD_CTX_CREATE: u32 = 0x200;
pub const VIRTIO_GPU_CMD_CTX_DESTROY: u32 = 0x201;
pub const VIRTIO_GPU_CMD_CTX_ATTACH_RESOURCE: u32 = 0x202;
pub const VIRTIO_GPU_CMD_CTX_DETACH_RESOURCE: u32 = 0x203;
pub const VIRTIO_GPU_CMD_RESOURCE_CREATE_3D: u32 = 0x204;
pub const VIRTIO_GPU_CMD_TRANSFER_TO_HOST_3D: u32 = 0x205;
pub const VIRTIO_GPU_CMD_TRANSFER_FROM_HOST_3D: u32 = 0x206;
pub const VIRTIO_GPU_CMD_SUBMIT_3D: u32 = 0x207;
pub const VIRTIO_GPU_CMD_RESOURCE_MAP_BLOB: u32 = 0x208;
pub const VIRTIO_GPU_CMD_RESOURCE_UNMAP_BLOB: u32 = 0x209;

/* cursor commands */
pub const VIRTIO_GPU_CMD_UPDATE_CURSOR: u32 = 0x300;
pub const VIRTIO_GPU_CMD_MOVE_CURSOR: u32 = 0x301;

#[derive(Debug, PartialEq, Eq)]
pub struct InvalidCommandType(u32);

impl std::fmt::Display for InvalidCommandType {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(fmt, "Invalid command type {}", self.0)
    }
}

impl From<InvalidCommandType> for crate::vhu_gpu::Error {
    fn from(val: InvalidCommandType) -> Self {
        Self::InvalidCommandType(val.0)
    }
}

impl std::error::Error for InvalidCommandType {}

#[derive(Copy, Clone)]
pub enum GpuCommandType {
    GetDisplayInfo = 0x100,
    ResourceCreate2d = 0x101,
    ResourceUnref = 0x102,
    SetScanout = 0x103,
    SetScanoutBlob = 0x10d,
    ResourceFlush = 0x104,
    TransferToHost2d = 0x105,
    ResourceAttachBacking = 0x106,
    ResourceDetachBacking = 0x107,
    GetCapsetInfo = 0x108,
    GetCapset = 0x109,
    GetEdid = 0x10a,
    CtxCreate = 0x200,
    CtxDestroy = 0x201,
    CtxAttachResource = 0x202,
    CtxDetachResource = 0x203,
    ResourceCreate3d = 0x204,
    TransferToHost3d = 0x205,
    TransferFromHost3d = 0x206,
    CmdSubmit3d = 0x207,
    ResourceCreateBlob = 0x10c,
    ResourceMapBlob = 0x208,
    ResourceUnmapBlob = 0x209,
    UpdateCursor = 0x300,
    MoveCursor = 0x301,
    ResourceAssignUuid = 0x10b,
}

impl TryFrom<Le32> for GpuCommandType {
    type Error = InvalidCommandType;

    fn try_from(value: Le32) -> Result<Self, Self::Error> {
        Ok(match u32::from(value) {
            VIRTIO_GPU_CMD_GET_DISPLAY_INFO => Self::GetDisplayInfo,
            VIRTIO_GPU_CMD_RESOURCE_CREATE_2D => Self::ResourceCreate2d,
            VIRTIO_GPU_CMD_RESOURCE_UNREF => Self::ResourceUnref,
            VIRTIO_GPU_CMD_SET_SCANOUT => Self::SetScanout,
            VIRTIO_GPU_CMD_SET_SCANOUT_BLOB => Self::SetScanoutBlob,
            VIRTIO_GPU_CMD_RESOURCE_FLUSH => Self::ResourceFlush,
            VIRTIO_GPU_CMD_TRANSFER_TO_HOST_2D => Self::TransferToHost2d,
            VIRTIO_GPU_CMD_RESOURCE_ATTACH_BACKING => Self::ResourceAttachBacking,
            VIRTIO_GPU_CMD_RESOURCE_DETACH_BACKING => Self::ResourceDetachBacking,
            VIRTIO_GPU_CMD_GET_CAPSET => Self::GetCapset,
            VIRTIO_GPU_CMD_GET_CAPSET_INFO => Self::GetCapsetInfo,
            VIRTIO_GPU_CMD_GET_EDID => Self::GetEdid,
            VIRTIO_GPU_CMD_CTX_CREATE => Self::CtxCreate,
            VIRTIO_GPU_CMD_CTX_DESTROY => Self::CtxDestroy,
            VIRTIO_GPU_CMD_CTX_ATTACH_RESOURCE => Self::CtxAttachResource,
            VIRTIO_GPU_CMD_CTX_DETACH_RESOURCE => Self::CtxDetachResource,
            VIRTIO_GPU_CMD_RESOURCE_CREATE_3D => Self::ResourceCreate3d,
            VIRTIO_GPU_CMD_TRANSFER_TO_HOST_3D => Self::TransferToHost3d,
            VIRTIO_GPU_CMD_TRANSFER_FROM_HOST_3D => Self::TransferFromHost3d,
            VIRTIO_GPU_CMD_SUBMIT_3D => Self::CmdSubmit3d,
            VIRTIO_GPU_CMD_RESOURCE_CREATE_BLOB => Self::ResourceCreateBlob,
            VIRTIO_GPU_CMD_RESOURCE_MAP_BLOB => Self::ResourceMapBlob,
            VIRTIO_GPU_CMD_RESOURCE_UNMAP_BLOB => Self::ResourceUnmapBlob,
            VIRTIO_GPU_CMD_UPDATE_CURSOR => Self::UpdateCursor,
            VIRTIO_GPU_CMD_MOVE_CURSOR => Self::MoveCursor,
            VIRTIO_GPU_CMD_RESOURCE_ASSIGN_UUID => Self::ResourceAssignUuid,
            other => return Err(InvalidCommandType(other)),
        })
    }
}
