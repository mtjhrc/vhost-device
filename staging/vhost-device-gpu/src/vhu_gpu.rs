// vhost device Gpu
//
// Copyright 2024 RedHat
//
// SPDX-License-Identifier: Apache-2.0 or BSD-3-Clause

use log::{debug, info, warn, error};
use std::{
    convert,
    io::{self, Result as IoResult, Read, Write},
};

use thiserror::Error as ThisError;
use vhost::vhost_user::message::{VhostUserProtocolFeatures, VhostUserVirtioFeatures};
use vhost_user_backend::{VhostUserBackendMut, VringRwLock, VringT};
use virtio_bindings::{bindings::virtio_config::{VIRTIO_F_NOTIFY_ON_EMPTY, VIRTIO_F_VERSION_1}, virtio_gpu::{VIRTIO_GPU_F_CONTEXT_INIT, VIRTIO_GPU_F_RESOURCE_BLOB}};
use virtio_bindings::bindings::virtio_ring::{
    VIRTIO_RING_F_EVENT_IDX, VIRTIO_RING_F_INDIRECT_DESC,
};
use virtio_queue::{DescriptorChain, QueueOwnedT};
use vm_memory::{ByteValued, Bytes, GuestAddress, GuestAddressSpace, GuestMemoryAtomic, GuestMemoryLoadGuard, GuestMemoryMmap, Le32};
use vmm_sys_util::epoll::EventSet;
use vmm_sys_util::eventfd::{EventFd, EFD_NONBLOCK};

use rutabaga_gfx::{
    ResourceCreate3D, ResourceCreateBlob, RutabagaFence, Transfer3D,
    RUTABAGA_PIPE_BIND_RENDER_TARGET, RUTABAGA_PIPE_TEXTURE_2D,
};
use crate::{
    GpuConfig,
    virtio_gpu::*,
    virtio_gpu::GpuCommandType,
    //VirtioShmRegion,
};
use super::protocol::{
    virtio_gpu_ctrl_hdr, virtio_gpu_mem_entry, GpuCommand, GpuResponse, VirtioGpuResult,
};
use crate::protocol::{VIRTIO_GPU_FLAG_FENCE, VIRTIO_GPU_FLAG_INFO_RING_IDX};
use super::virt_gpu::{
    VirtioGpu,
    VirtioGpuRing,
    VirtioShmRegion,
};

type Result<T> = std::result::Result<T, Error>;

#[derive(Copy, Clone, Debug, Eq, PartialEq, ThisError)]
pub(crate) enum Error {
    #[error("Failed to handle event, didn't match EPOLLIN")]
    HandleEventNotEpollIn,
    #[error("Failed to handle unknown event")]
    HandleEventUnknown,
    #[error("Descriptor not found")]
    DescriptorNotFound,
    #[error("Descriptor read failed")]
    DescriptorReadFailed,
    #[error("Descriptor write failed")]
    DescriptorWriteFailed,
    #[error("Invalid command type {0}")]
    InvalidCommandType(u32),
    #[error("Failed to send notification")]
    NotificationFailed,
    #[error("No memory configured")]
    NoMemoryConfigured,
    #[error("Failed to create new EventFd")]
    EventFdFailed,
    #[error("Received unexpected write only descriptor at index {0}")]
    UnexpectedWriteOnlyDescriptor(usize),
    #[error("Received unexpected readable descriptor at index {0}")]
    UnexpectedReadableDescriptor(usize),
    #[error("Invalid descriptor count {0}")]
    UnexpectedDescriptorCount(usize),
    #[error("Invalid descriptor size, expected: {0}, found: {1}")]
    UnexpectedDescriptorSize(usize, u32),
}

impl convert::From<Error> for io::Error {
    fn from(e: Error) -> Self {
        io::Error::new(io::ErrorKind::Other, e)
    }
}
pub(crate) struct VhostUserGpuBackend {
    virtio_cfg: VirtioGpuConfig,
    event_idx: bool,
    pub exit_event: EventFd,
    mem: Option<GuestMemoryAtomic<GuestMemoryMmap>>,
    shm_region: Option<VirtioShmRegion>,
}

type GpuDescriptorChain = DescriptorChain<GuestMemoryLoadGuard<GuestMemoryMmap<()>>>;

impl VhostUserGpuBackend {
    pub fn new(gpu_config: GpuConfig) -> Result<Self> {
        log::trace!("VhostUserGpuBackend::new(config = {:?})", &gpu_config);
        Ok(VhostUserGpuBackend {
            virtio_cfg: VirtioGpuConfig {
                events_read: 0.into(),
                events_clear: 0.into(),
                num_scanouts: Le32::from(VIRTIO_GPU_MAX_SCANOUTS as u32),
                num_capsets: 0.into(),
            },
            event_idx: false,
            exit_event: EventFd::new(EFD_NONBLOCK).map_err(|_| Error::EventFdFailed)?,
            mem: None,
            shm_region: None,
        })
    }

    fn set_shm_region(&mut self, shm_region: VirtioShmRegion) {
        debug!("virtio_gpu: set_shm_region");
        self.shm_region = Some(shm_region);
    }

    fn process_gpu_command(
        &mut self,
        virtio_gpu: &mut VirtioGpu,
        mem: &GuestMemoryMmap,
        hdr: virtio_gpu_ctrl_hdr,
        cmd: GpuCommand,
        reader: &GuestMemoryMmap,
        desc_addr: GuestAddress,
    ) -> VirtioGpuResult {
        virtio_gpu.force_ctx_0();
        match cmd {
            GpuCommand::GetDisplayInfo(_) => {
                panic!("virtio_gpu: GpuCommand::GetDisplayInfo unimplemented");
            }
            GpuCommand::ResourceCreate2d(info) => {
                let resource_id = info.resource_id;

                let resource_create_3d = ResourceCreate3D {
                    target: RUTABAGA_PIPE_TEXTURE_2D,
                    format: info.format,
                    bind: RUTABAGA_PIPE_BIND_RENDER_TARGET,
                    width: info.width,
                    height: info.height,
                    depth: 1,
                    array_size: 1,
                    last_level: 0,
                    nr_samples: 0,
                    flags: 0,
                };

                virtio_gpu.resource_create_3d(resource_id, resource_create_3d)
            }
            GpuCommand::ResourceUnref(info) => virtio_gpu.unref_resource(info.resource_id),
            GpuCommand::SetScanout(_info) => {
                panic!("virtio_gpu: GpuCommand::SetScanout unimplemented");
            }
            GpuCommand::ResourceFlush(info) => virtio_gpu.flush_resource(info.resource_id),
            GpuCommand::TransferToHost2d(info) => {
                let resource_id = info.resource_id;
                let transfer = Transfer3D::new_2d(info.r.x, info.r.y, info.r.width, info.r.height);
                virtio_gpu.transfer_write(0, resource_id, transfer)
            }
            GpuCommand::ResourceAttachBacking(info) => {todo!()}
            // GpuCommand::ResourceAttachBacking(info) => {
            //     let available_bytes = reader.available_bytes();
            //     if available_bytes != 0 {
            //         let entry_count = info.nr_entries as usize;
            //         let mut vecs = Vec::with_capacity(entry_count);
            //         for _ in 0..entry_count {
            //             match reader.read_obj::<virtio_gpu_mem_entry>(desc_addr) {
            //                 Ok(entry) => {
            //                     let addr = GuestAddress(entry.addr);
            //                     let len = entry.length as usize;
            //                     vecs.push((addr, len))
            //                 }
            //                 Err(_) => return Err(GpuResponse::ErrUnspec),
            //             }
            //         }
            //         virtio_gpu.attach_backing(info.resource_id, mem, vecs)
            //     } else {
            //         error!("missing data for command {:?}", cmd);
            //         Err(GpuResponse::ErrUnspec)
            //     }
            // }
            GpuCommand::ResourceDetachBacking(info) => virtio_gpu.detach_backing(info.resource_id),
            GpuCommand::UpdateCursor(_info) => {
                panic!("virtio_gpu: GpuCommand:UpdateCursor unimplemented");
            }
            GpuCommand::MoveCursor(_info) => {
                panic!("virtio_gpu: GpuCommand::MoveCursor unimplemented");
            }
            GpuCommand::ResourceAssignUuid(info) => {
                let resource_id = info.resource_id;
                virtio_gpu.resource_assign_uuid(resource_id)
            }
            GpuCommand::GetCapsetInfo(info) => virtio_gpu.get_capset_info(info.capset_index),
            GpuCommand::GetCapset(info) => {
                virtio_gpu.get_capset(info.capset_id, info.capset_version)
            }

            GpuCommand::CtxCreate(info) => {
                let context_name: Option<String> = String::from_utf8(info.debug_name.to_vec()).ok();
                virtio_gpu.create_context(hdr.ctx_id, info.context_init, context_name.as_deref())
            }
            GpuCommand::CtxDestroy(_info) => virtio_gpu.destroy_context(hdr.ctx_id),
            GpuCommand::CtxAttachResource(info) => {
                virtio_gpu.context_attach_resource(hdr.ctx_id, info.resource_id)
            }
            GpuCommand::CtxDetachResource(info) => {
                virtio_gpu.context_detach_resource(hdr.ctx_id, info.resource_id)
            }
            GpuCommand::ResourceCreate3d(info) => {
                let resource_id = info.resource_id;
                let resource_create_3d = ResourceCreate3D {
                    target: info.target,
                    format: info.format,
                    bind: info.bind,
                    width: info.width,
                    height: info.height,
                    depth: info.depth,
                    array_size: info.array_size,
                    last_level: info.last_level,
                    nr_samples: info.nr_samples,
                    flags: info.flags,
                };

                virtio_gpu.resource_create_3d(resource_id, resource_create_3d)
            }
            GpuCommand::TransferToHost3d(info) => {
                let ctx_id = hdr.ctx_id;
                let resource_id = info.resource_id;

                let transfer = Transfer3D {
                    x: info.box_.x,
                    y: info.box_.y,
                    z: info.box_.z,
                    w: info.box_.w,
                    h: info.box_.h,
                    d: info.box_.d,
                    level: info.level,
                    stride: info.stride,
                    layer_stride: info.layer_stride,
                    offset: info.offset,
                };

                virtio_gpu.transfer_write(ctx_id, resource_id, transfer)
            }
            GpuCommand::TransferFromHost3d(info) => {
                let ctx_id = hdr.ctx_id;
                let resource_id = info.resource_id;

                let transfer = Transfer3D {
                    x: info.box_.x,
                    y: info.box_.y,
                    z: info.box_.z,
                    w: info.box_.w,
                    h: info.box_.h,
                    d: info.box_.d,
                    level: info.level,
                    stride: info.stride,
                    layer_stride: info.layer_stride,
                    offset: info.offset,
                };

                virtio_gpu.transfer_read(ctx_id, resource_id, transfer, None)
            }
            GpuCommand::CmdSubmit3d(info) => {todo!()}
            GpuCommand::ResourceCreateBlob(info) => {todo!()}
            // GpuCommand::CmdSubmit3d(info) => {
            //     if reader.available_bytes() != 0 {
            //         let num_in_fences = info.num_in_fences as usize;
            //         let cmd_size = info.size as usize;
            //         let mut cmd_buf = vec![0; cmd_size];
            //         let mut fence_ids: Vec<u64> = Vec::with_capacity(num_in_fences);

            //         for _ in 0..num_in_fences {
            //             match reader.read_obj::<u64>(desc_addr) {
            //                 Ok(fence_id) => {
            //                     fence_ids.push(fence_id);
            //                 }
            //                 Err(_) => return Err(GpuResponse::ErrUnspec),
            //             }
            //         }

            //         if reader.read_exact(&mut cmd_buf[..]).is_ok() {
            //             virtio_gpu.submit_command(hdr.ctx_id, &mut cmd_buf[..], &fence_ids)
            //         } else {
            //             Err(GpuResponse::ErrInvalidParameter)
            //         }
            //     } else {
            //         // Silently accept empty command buffers to allow for
            //         // benchmarking.
            //         Ok(GpuResponse::OkNoData)
            //     }
            // }
            // GpuCommand::ResourceCreateBlob(info) => {
            //     let resource_id = info.resource_id;
            //     let ctx_id = hdr.ctx_id;

            //     let resource_create_blob = ResourceCreateBlob {
            //         blob_mem: info.blob_mem,
            //         blob_flags: info.blob_flags,
            //         blob_id: info.blob_id,
            //         size: info.size,
            //     };

            //     let entry_count = info.nr_entries;
            //     if reader.available_bytes() == 0 && entry_count > 0 {
            //         return Err(GpuResponse::ErrUnspec);
            //     }

            //     let mut vecs = Vec::with_capacity(entry_count as usize);
            //     for _ in 0..entry_count {
            //         match reader.read_obj::<virtio_gpu_mem_entry>(desc_addr) {
            //             Ok(entry) => {
            //                 let addr = GuestAddress(entry.addr);
            //                 let len = entry.length as usize;
            //                 vecs.push((addr, len))
            //             }
            //             Err(_) => return Err(GpuResponse::ErrUnspec),
            //         }
            //     }

            //     virtio_gpu.resource_create_blob(
            //         ctx_id,
            //         resource_id,
            //         resource_create_blob,
            //         vecs,
            //         mem,
            //     )
            // }
            GpuCommand::SetScanoutBlob(_info) => {
                panic!("virtio_gpu: GpuCommand::SetScanoutBlob unimplemented");
            }
            GpuCommand::ResourceMapBlob(info) => {
                let resource_id = info.resource_id;
                let offset = info.offset;
                let sregion = self.shm_region.as_ref().unwrap();
                virtio_gpu.resource_map_blob(resource_id, &sregion, offset)
            }
            GpuCommand::ResourceUnmapBlob(info) => {
                let resource_id = info.resource_id;
                let sregion = self.shm_region.as_ref().unwrap();
                virtio_gpu.resource_unmap_blob(resource_id, &sregion)
            }
        }
    }
    /// Process the requests in the vring and dispatch replies
    fn process_requests(
        &mut self,
        requests: Vec<GpuDescriptorChain>,
        virtio_gpu: &mut VirtioGpu,
        vring: &VringRwLock,
    ) -> Result<()> {
        if requests.is_empty() {
            info!("No pending requests");
            return Ok(());
        }
        let mut used_any = false;

        // Iterate over each gpu request.
        //
        // The layout of the various structures, to be read from and written into the descriptor
        // buffers, is defined in the Virtio specification for each protocol.
        for desc_chain in requests.clone() {
            let descriptors: Vec<_> = desc_chain.clone().collect();
            if descriptors.len() < 2 {
                return Err(Error::UnexpectedDescriptorCount(descriptors.len()).into());
            }

            info!(
                "Request contains {} descriptors",
                descriptors.len(),
            );

            for (i, desc) in descriptors.iter().enumerate() {
                let perm = if desc.is_write_only() {
                    "write only"
                } else {
                    "read only"
                };

                // We now can iterate over the set of descriptors and process the messages. There
                // will be a number of read only descriptors containing messages as defined by the
                // specification. If any replies are needed, the driver should have placed one or
                // more writable descriptors at the end for the device to use to reply.
                info!("Length of the {} descriptor@{} is: {}", perm, i, desc.len());
            }

            // Request Header descriptor
            let desc_request = descriptors[0];
            if desc_request.is_write_only() {
                return Err(Error::UnexpectedWriteOnlyDescriptor(0).into());
            }

            // let mut reader = desc_chain.memory().read_obj::<virtio_gpu_mem_entry>(desc_request.addr());
            let mut reader = desc_chain.memory();
            let request = desc_chain
                .memory()
                .read_obj::<VirtioGpuCtrlHdr>(desc_request.addr())
                .map_err(|_| Error::DescriptorReadFailed)?;

            // Keep track of bytes that will be written in the VQ.
            let mut used_len = 0;

            // Reply header descriptor.
            let desc_hdr = descriptors[1];
            if !desc_hdr.is_write_only() {
                return Err(Error::UnexpectedReadableDescriptor(1).into());
            }

            let Some(ref atomic_mem) = self.mem else {
                return Err(Error::NoMemoryConfigured.into());
            };
            let mem: GuestMemoryMmap = (*atomic_mem.memory().into_inner()).clone();

            let mut resp: std::result::Result<GpuResponse, GpuResponse> = Err(GpuResponse::ErrUnspec);
            let mut gpu_cmd: Option<GpuCommand> = None;
            let mut ctrl_hdr: Option<virtio_gpu_ctrl_hdr> = None;
            let mut len = 0;
            let mut desc_addr = desc_request.addr();
            let mut desc_hdraddr = desc_hdr.addr();

            match GpuCommand::decode(&reader, desc_addr) {
                Ok((hdr, cmd)) => {
                    resp = self.process_gpu_command(virtio_gpu, &mem, hdr, cmd, &reader, desc_addr);
                    ctrl_hdr = Some(hdr);
                    gpu_cmd = Some(cmd);
                }
                Err(e) => debug!("descriptor decode error: {:?}", e),
            }
            let mut gpu_response = match resp {
                Ok(gpu_response) => gpu_response,
                Err(gpu_response) => {
                    debug!("{:?} -> {:?}", gpu_cmd, gpu_response);
                    gpu_response
                }
            };

            let mut add_to_queue = true;

            let mut fence_id = 0;
            let mut ctx_id = 0;
            let mut flags = 0;
            let mut ring_idx = 0;
            if let Some(_cmd) = gpu_cmd {
                let ctrl_hdr = ctrl_hdr.unwrap();
                if ctrl_hdr.flags & VIRTIO_GPU_FLAG_FENCE != 0 {
                    flags = ctrl_hdr.flags;
                    fence_id = ctrl_hdr.fence_id;
                    ctx_id = ctrl_hdr.ctx_id;
                    ring_idx = ctrl_hdr.ring_idx;

                    let fence = RutabagaFence {
                        flags,
                        fence_id,
                        ctx_id,
                        ring_idx,
                    };
                    gpu_response = match virtio_gpu.create_fence(fence) {
                        Ok(_) => gpu_response,
                        Err(fence_resp) => {
                            warn!("create_fence {} -> {:?}", fence_id, fence_resp);
                            fence_resp
                        }
                    };
                }
            }

            // Prepare the response now, even if it is going to wait until
            // fence is complete.

            //edit reader definition later to something better signifying response wrte
            match gpu_response.encode(flags, fence_id, ctx_id, ring_idx, &reader, desc_hdraddr) {
                Ok(l) => len = l,
                Err(e) => debug!("ctrl queue response encode error: {:?}", e),
            }

            if flags & VIRTIO_GPU_FLAG_FENCE != 0 {
                let ring = match flags & VIRTIO_GPU_FLAG_INFO_RING_IDX {
                    0 => VirtioGpuRing::Global,
                    _ => VirtioGpuRing::ContextSpecific { ctx_id, ring_idx },
                };

                add_to_queue = virtio_gpu.process_fence(ring, fence_id, desc_chain.head_index(), len);
            }
            if add_to_queue {
                used_len += desc_hdr.len();
                if vring.add_used(desc_chain.head_index(), used_len).is_err() {
                    log::error!("Couldn't return used descriptors to the ring");
                }
                used_any = true;
            }

            // // Keep track of bytes that will be written in the VQ.
            // let mut used_len = 0;

            // // Response Header descriptor.
            // let desc_response = descriptors[1];
            // if !desc_response.is_write_only() {
            //     return Err(Error::UnexpectedReadableDescriptor(1).into());
            // }
            // let gpu_cmd_type = GpuCommandType::try_from(request.gpu_type).map_err(Error::from)?;
            // //let resp = self.process_gpu_command(gpu_cmd_type);
        }

        Ok(())
    }

    /// Process the requests in the vring and dispatch replies
    fn process_control_queue(&mut self, vring: &VringRwLock) -> Result<()> {
        // Collect all pending requests
        debug!("Processing control queue");
        let Some(ref atomic_mem) = self.mem else {
            return Err(Error::NoMemoryConfigured.into());
        };
        let requests: Vec<_> = vring
            .get_mut()
            .get_queue_mut()
            .iter(atomic_mem.memory())
            .map_err(|_| Error::DescriptorNotFound)?
            .collect();

        let mem: GuestMemoryMmap = (*atomic_mem.memory().into_inner()).clone();

        let mut virtio_gpu = VirtioGpu::new(
            vring,
        );

        if self.process_requests(requests, &mut virtio_gpu, vring).is_ok() {
            // Send notification once all the requests are processed
            debug!("Sending processed request notification");
            vring
                .signal_used_queue()
                .map_err(|_| Error::NotificationFailed)?;
            debug!("Notification sent");
        }

        debug!("Processing control queue finished");
        Ok(())
    }

    fn process_cursor_queue(&self, _vring: &VringRwLock) -> IoResult<()> {
        debug!("process_cusor_q");
        Ok(())
    }
}

/// VhostUserBackendMut trait methods
impl VhostUserBackendMut for VhostUserGpuBackend {
    type Vring = VringRwLock;
    type Bitmap = ();

    fn num_queues(&self) -> usize {
        debug!("Num queues called");
        NUM_QUEUES
    }

    fn max_queue_size(&self) -> usize {
        debug!("Max queues called");
        QUEUE_SIZE
    }

    fn features(&self) -> u64 {
        debug!("Features called");
        1 << VIRTIO_F_VERSION_1
            | 1 << VIRTIO_F_NOTIFY_ON_EMPTY
            | 1 << VIRTIO_RING_F_INDIRECT_DESC
            | 1 << VIRTIO_RING_F_EVENT_IDX
            | 1 << VIRTIO_GPU_F_VIRGL
            | 1 << VIRTIO_GPU_F_EDID
            | 1 << VIRTIO_GPU_F_RESOURCE_BLOB
            | 1 << VIRTIO_GPU_F_CONTEXT_INIT

            | VhostUserVirtioFeatures::PROTOCOL_FEATURES.bits()
    }

    fn protocol_features(&self) -> VhostUserProtocolFeatures {
        debug!("Protocol features called");
        VhostUserProtocolFeatures::CONFIG | VhostUserProtocolFeatures::MQ
    }

    fn set_event_idx(&mut self, enabled: bool) {
        self.event_idx = enabled;
        debug!("Event idx set to: {}", enabled);
    }

    fn update_memory(&mut self, mem: GuestMemoryAtomic<GuestMemoryMmap>) -> IoResult<()> {
        debug!("Update memory called");
        self.mem = Some(mem);
        Ok(())
    }

    fn handle_event(
        &mut self,
        device_event: u16,
        evset: EventSet,
        vrings: &[VringRwLock],
        _thread_id: usize,
    ) -> IoResult<()> {
        debug!("Handle event called");
        if evset != EventSet::IN {
            return Err(Error::HandleEventNotEpollIn.into());
        }

        match device_event {
            CONTROL_QUEUE => {
                let vring = &vrings
                    .get(device_event as usize)
                    .ok_or_else(|| Error::HandleEventUnknown)?;

                if self.event_idx {
                    // vm-virtio's Queue implementation only checks avail_index
                    // once, so to properly support EVENT_IDX we need to keep
                    // calling process_queue() until it stops finding new
                    // requests on the queue.
                    loop {
                        vring.disable_notification().unwrap();
                        self.process_control_queue(vring)?;
                        if !vring.enable_notification().unwrap() {
                            break;
                        }
                    }
                } else {
                    // Without EVENT_IDX, a single call is enough.
                    self.process_control_queue(vring)?;
                }
            }

            CURSOR_QUEUE => {
                let vring = &vrings
                    .get(device_event as usize)
                    .ok_or_else(|| Error::HandleEventUnknown)?;

                if self.event_idx {
                    // vm-virtio's Queue implementation only checks avail_index
                    // once, so to properly support EVENT_IDX we need to keep
                    // calling process_queue() until it stops finding new
                    // requests on the queue.
                    loop {
                        vring.disable_notification().unwrap();
                        self.process_cursor_queue(vring)?;
                        if !vring.enable_notification().unwrap() {
                            break;
                        }
                    }
                } else {
                    // Without EVENT_IDX, a single call is enough.
                    self.process_cursor_queue(vring)?;
                }
            }

            _ => {
                warn!("unhandled device_event: {}", device_event);
                return Err(Error::HandleEventUnknown.into());
            }
        }
        Ok(())
    }

    fn get_config(&self, offset: u32, size: u32) -> Vec<u8> {
        let offset = offset as usize;
        let size = size as usize;

        let buf = self.virtio_cfg.as_slice();

        if offset + size > buf.len() {
            return Vec::new();
        }

        buf[offset..offset + size].to_vec()
    }

    fn exit_event(&self, _thread_index: usize) -> Option<EventFd> {
        self.exit_event.try_clone().ok()
    }
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;
    use vhost_user_backend::{VhostUserBackendMut, VringRwLock, VringT};
    use virtio_bindings::bindings::virtio_ring::{VRING_DESC_F_NEXT, VRING_DESC_F_WRITE};
    use virtio_queue::{mock::MockSplitQueue, Descriptor, Queue};
    use vm_memory::{Address, ByteValued, Bytes, GuestAddress, GuestMemoryAtomic, GuestMemoryMmap};

    use super::*;

    const SOCKET_PATH: &str = "vgpu.socket";

    #[derive(Copy, Clone, Default)]
    #[repr(C)]
    struct VirtioGpuOutHdr {
        a: u16,
        b: u16,
        c: u32,
    }
    // SAFETY: The layout of the structure is fixed and can be initialized by
    // reading its content from byte array.
    unsafe impl ByteValued for VirtioGpuOutHdr {}

    #[derive(Copy, Clone, Default)]
    #[repr(C)]
    struct VirtioGpuInHdr {
        d: u8,
    }
    // SAFETY: The layout of the structure is fixed and can be initialized by
    // reading its content from byte array.
    unsafe impl ByteValued for VirtioGpuInHdr {}

    fn init() -> (
        VhostUserGpuBackend,
        GuestMemoryAtomic<GuestMemoryMmap>,
        VringRwLock,
    ) {
        let backend = VhostUserGpuBackend::new(GpuConfig::new(SOCKET_PATH.into())).unwrap();
        let mem = GuestMemoryAtomic::new(
            GuestMemoryMmap::<()>::from_ranges(&[(GuestAddress(0), 0x1000)]).unwrap(),
        );
        let vring = VringRwLock::new(mem.clone(), 16).unwrap();

        (backend, mem, vring)
    }

    // Prepares a single chain of descriptors
    fn prepare_descriptors(
        mut next_addr: u64,
        mem: &GuestMemoryLoadGuard<GuestMemoryMmap<()>>,
        buf: &mut Vec<u8>,
    ) -> Vec<Descriptor> {
        let mut descriptors = Vec::new();
        let mut index = 0;

        // Out header descriptor
        let out_hdr = VirtioGpuOutHdr {
            a: 0x10,
            b: 0x11,
            c: 0x20,
        };

        let desc_out = Descriptor::new(
            next_addr,
            size_of::<VirtioGpuOutHdr>() as u32,
            VRING_DESC_F_NEXT as u16,
            index + 1,
        );
        next_addr += desc_out.len() as u64;
        index += 1;

        mem.write_obj::<VirtioGpuOutHdr>(out_hdr, desc_out.addr())
            .unwrap();
        descriptors.push(desc_out);

        // Buf descriptor: optional
        if !buf.is_empty() {
            let desc_buf = Descriptor::new(
                next_addr,
                buf.len() as u32,
                (VRING_DESC_F_WRITE | VRING_DESC_F_NEXT) as u16,
                index + 1,
            );
            next_addr += desc_buf.len() as u64;

            mem.write(buf, desc_buf.addr()).unwrap();
            descriptors.push(desc_buf);
        }

        // In response descriptor
        let desc_in = Descriptor::new(
            next_addr,
            size_of::<VirtioGpuInHdr>() as u32,
            VRING_DESC_F_WRITE as u16,
            0,
        );
        descriptors.push(desc_in);
        descriptors
    }

    // Prepares a single chain of descriptors
    fn prepare_desc_chain(buf: &mut Vec<u8>) -> (VhostUserGpuBackend, VringRwLock) {
        let (mut backend, mem, vring) = init();
        let mem_handle = mem.memory();
        let vq = MockSplitQueue::new(&*mem_handle, 16);
        let next_addr = vq.desc_table().total_size() + 0x100;

        let descriptors = prepare_descriptors(next_addr, &mem_handle, buf);

        vq.build_desc_chain(&descriptors).unwrap();

        // Put the descriptor index 0 in the first available ring position.
        mem_handle
            .write_obj(0u16, vq.avail_addr().unchecked_add(4))
            .unwrap();

        // Set `avail_idx` to 1.
        mem_handle
            .write_obj(1u16, vq.avail_addr().unchecked_add(2))
            .unwrap();

        vring.set_queue_size(16);
        vring
            .set_queue_info(vq.desc_table_addr().0, vq.avail_addr().0, vq.used_addr().0)
            .unwrap();
        vring.set_queue_ready(true);

        backend.update_memory(mem).unwrap();

        (backend, vring)
    }

    // Prepares a chain of descriptors
    fn prepare_desc_chains(
        mem: &GuestMemoryAtomic<GuestMemoryMmap>,
        buf: &mut Vec<u8>,
    ) -> GpuDescriptorChain {
        let mem_handle = mem.memory();
        let vq = MockSplitQueue::new(&*mem_handle, 16);
        let next_addr = vq.desc_table().total_size() + 0x100;

        let descriptors = prepare_descriptors(next_addr, &mem_handle, buf);

        for (idx, desc) in descriptors.iter().enumerate() {
            vq.desc_table().store(idx as u16, *desc).unwrap();
        }

        // Put the descriptor index 0 in the first available ring position.
        mem_handle
            .write_obj(0u16, vq.avail_addr().unchecked_add(4))
            .unwrap();

        // Set `avail_idx` to 1.
        mem_handle
            .write_obj(1u16, vq.avail_addr().unchecked_add(2))
            .unwrap();

        // Create descriptor chain from pre-filled memory
        vq.create_queue::<Queue>()
            .unwrap()
            .iter(mem_handle)
            .unwrap()
            .next()
            .unwrap()
    }

    #[test]
    fn process_requests_no_desc() {
        let (mut backend, _, vring) = init();

        // Descriptor chain size zero, shouldn't fail
        // backend
        //     .process_requests(Vec::<GpuDescriptorChain>::new(), &vring)
        //     .unwrap();
    }

    #[test]
    fn process_request_single() {
        // Single valid descriptor
        let mut buf: Vec<u8> = vec![0; 30];
        let (mut backend, vring) = prepare_desc_chain(&mut buf);
        backend.process_control_queue(&vring).unwrap();
    }

    #[test]
    fn process_requests_multi() {
        // Multiple valid descriptors
        let (mut backend, mem, vring) = init();

        let mut bufs: Vec<Vec<u8>> = vec![vec![0; 30]; 6];
        let desc_chains = vec![
            prepare_desc_chains(&mem, &mut bufs[0]),
            prepare_desc_chains(&mem, &mut bufs[1]),
            prepare_desc_chains(&mem, &mut bufs[2]),
            prepare_desc_chains(&mem, &mut bufs[3]),
            prepare_desc_chains(&mem, &mut bufs[4]),
            prepare_desc_chains(&mem, &mut bufs[5]),
        ];

        // backend
        //     .process_requests(desc_chains.clone(), &vring)
        //     .unwrap();
    }

    #[test]
    fn verify_backend() {
        let gpu_config = GpuConfig::new(SOCKET_PATH.into());
        let mut backend = VhostUserGpuBackend::new(gpu_config).unwrap();

        assert_eq!(backend.num_queues(), NUM_QUEUES);
        assert_eq!(backend.max_queue_size(), QUEUE_SIZE);
        assert_eq!(backend.features(), 0x171000000);
        assert_eq!(backend.protocol_features(), VhostUserProtocolFeatures::MQ);

        assert_eq!(backend.queues_per_thread(), vec![0xffff_ffff]);
        assert_eq!(backend.get_config(0, 0), vec![]);

        backend.set_event_idx(true);
        assert!(backend.event_idx);

        assert!(backend.exit_event(0).is_some());

        let mem = GuestMemoryAtomic::new(
            GuestMemoryMmap::<()>::from_ranges(&[(GuestAddress(0), 0x1000)]).unwrap(),
        );
        backend.update_memory(mem.clone()).unwrap();

        let vring = VringRwLock::new(mem, 0x1000).unwrap();
        vring.set_queue_info(0x100, 0x200, 0x300).unwrap();
        vring.set_queue_ready(true);

        assert_eq!(
            backend
                .handle_event(0, EventSet::OUT, &[vring.clone()], 0)
                .unwrap_err()
                .kind(),
            io::ErrorKind::Other
        );

        assert_eq!(
            backend
                .handle_event(1, EventSet::IN, &[vring.clone()], 0)
                .unwrap_err()
                .kind(),
            io::ErrorKind::Other
        );

        // Hit the loop part
        backend.set_event_idx(true);
        backend
            .handle_event(0, EventSet::IN, &[vring.clone()], 0)
            .unwrap();

        // Hit the non-loop part
        backend.set_event_idx(false);
        backend.handle_event(0, EventSet::IN, &[vring], 0).unwrap();
    }
}
