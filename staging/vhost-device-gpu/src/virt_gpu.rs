use std::collections::BTreeMap;
use std::env;
use std::io::IoSliceMut;
use std::os::fd::AsRawFd;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::vhu_gpu::Error;
use libc::c_void;
use log::{debug, error, trace};
use rutabaga_gfx::{
    ResourceCreate3D, ResourceCreateBlob, Rutabaga, RutabagaBuilder, RutabagaChannel,
    RutabagaFence, RutabagaFenceHandler, RutabagaIovec, RutabagaResult, Transfer3D,
    RUTABAGA_CHANNEL_TYPE_WAYLAND, RUTABAGA_MAP_ACCESS_MASK, RUTABAGA_MAP_ACCESS_READ,
    RUTABAGA_MAP_ACCESS_RW, RUTABAGA_MAP_ACCESS_WRITE, RUTABAGA_MAP_CACHE_MASK,
    RUTABAGA_MEM_HANDLE_TYPE_OPAQUE_FD,
};
//use utils::eventfd::EventFd;
use vhost::vhost_user::gpu_message::{
    VhostUserGpuCursorPos, VhostUserGpuCursorUpdate, VhostUserGpuEdidRequest, VhostUserGpuScanout,
    VhostUserGpuUpdate, VirtioGpuRespDisplayInfo,
};
use vhost_user_backend::{VringRwLock, VringT};
use vm_memory::{GuestAddress, GuestMemory, GuestMemoryMmap, VolatileSlice};

//use super::super::Queue as VirtQueue;
use super::protocol::GpuResponse::*;
use super::protocol::{
    virtio_gpu_rect, virtio_gpu_set_scanout, GpuResponse, GpuResponsePlaneInfo, VirtioGpuResult,
    VIRTIO_GPU_BLOB_FLAG_CREATE_GUEST_HANDLE, VIRTIO_GPU_BLOB_MEM_HOST3D,
};
use crate::protocol::VIRTIO_GPU_FLAG_INFO_RING_IDX;
use crate::virtio_gpu::VIRTIO_GPU_MAX_SCANOUTS;
use std::result::Result;
use vhost::vhost_user::GpuBackend;

fn sglist_to_rutabaga_iovecs(
    vecs: &[(GuestAddress, usize)],
    mem: &GuestMemoryMmap,
) -> Result<Vec<RutabagaIovec>, ()> {
    if vecs
        .iter()
        .any(|&(addr, len)| mem.get_slice(addr, len).is_err())
    {
        return Err(());
    }

    let mut rutabaga_iovecs: Vec<RutabagaIovec> = Vec::new();
    for &(addr, len) in vecs {
        let slice = mem.get_slice(addr, len).unwrap();
        rutabaga_iovecs.push(RutabagaIovec {
            base: slice.ptr_guard_mut().as_ptr() as *mut c_void,
            len,
        });
    }
    Ok(rutabaga_iovecs)
}

#[derive(Clone)]
pub struct VirtioShmRegion {
    pub host_addr: u64,
    pub guest_addr: u64,
    pub size: usize,
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub enum VirtioGpuRing {
    Global,
    ContextSpecific { ctx_id: u32, ring_idx: u8 },
}

struct FenceDescriptor {
    ring: VirtioGpuRing,
    fence_id: u64,
    desc_index: u16,
    len: u32,
}

#[derive(Default)]
pub struct FenceState {
    descs: Vec<FenceDescriptor>,
    completed_fences: BTreeMap<VirtioGpuRing, u64>,
}

#[derive(Copy, Clone, Debug, Default)]
struct AssociatedScanouts(u32);

impl AssociatedScanouts {
    fn enable(&mut self, scanout_id: u32) {
        self.0 |= 1 << scanout_id;
    }

    fn disable(&mut self, scanout_id: u32) {
        self.0 ^= 1 << scanout_id;
    }

    fn has_any(&self) -> bool {
        self.0 == 0
    }

    fn iter_enabled(self) -> impl Iterator<Item = u32> {
        (0..VIRTIO_GPU_MAX_SCANOUTS)
            .filter(move |i| ((self.0 >> i) & 1) == 1)
            .map(|n| n as u32)
    }
}

struct VirtioGpuResource {
    size: u64,
    shmem_offset: Option<u64>,
    rutabaga_external_mapping: bool,
    /// Stores information about which scanouts are associated with the given resource.
    /// Resource could be used for multiple scanouts (the displays are mirrored).
    scanouts: AssociatedScanouts,
}

impl VirtioGpuResource {
    /// Creates a new VirtioGpuResource with the given metadata.  Width and height are used by the
    /// display, while size is useful for hypervisor mapping.
    pub fn new(_resource_id: u32, _width: u32, _height: u32, size: u64) -> VirtioGpuResource {
        VirtioGpuResource {
            size,
            shmem_offset: None,
            rutabaga_external_mapping: false,
            scanouts: Default::default(),
        }
    }
}

#[derive(Default, Debug, Copy, Clone)]
pub struct Rectangle {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

impl From<virtio_gpu_rect> for Rectangle {
    fn from(r: virtio_gpu_rect) -> Self {
        Self {
            x: r.x,
            y: r.y,
            width: r.width,
            height: r.height,
        }
    }
}

struct VirtioGpuScanout {
    resource_id: u32,
    rect: Rectangle,
}

pub struct VirtioGpu {
    rutabaga: Rutabaga,
    resources: BTreeMap<u32, VirtioGpuResource>,
    fence_state: Arc<Mutex<FenceState>>,
    scanouts: [Option<VirtioGpuScanout>; VIRTIO_GPU_MAX_SCANOUTS],
}

impl VirtioGpu {
    fn create_fence_handler(
        queue_ctl: VringRwLock,
        fence_state: Arc<Mutex<FenceState>>,
        // interrupt_status: Arc<AtomicUsize>,
        // interrupt_evt: EventFd,
        // intc: Option<Arc<Mutex<Gic>>>,
        // irq_line: Option<u32>,
    ) -> RutabagaFenceHandler {
        RutabagaFenceHandler::new(move |completed_fence: RutabagaFence| {
            debug!(
                "XXX - fence called: id={}, ring_idx={}",
                completed_fence.fence_id, completed_fence.ring_idx
            );

            //let mut queue = queue_ctl.lock().unwrap();
            let mut fence_state = fence_state.lock().unwrap();
            let mut i = 0;

            let ring = match completed_fence.flags & VIRTIO_GPU_FLAG_INFO_RING_IDX {
                0 => VirtioGpuRing::Global,
                _ => VirtioGpuRing::ContextSpecific {
                    ctx_id: completed_fence.ctx_id,
                    ring_idx: completed_fence.ring_idx,
                },
            };

            while i < fence_state.descs.len() {
                debug!("XXX - fence_id: {}", fence_state.descs[i].fence_id);
                if fence_state.descs[i].ring == ring
                    && fence_state.descs[i].fence_id <= completed_fence.fence_id
                {
                    let completed_desc = fence_state.descs.remove(i);
                    debug!(
                        "XXX - found fence: desc_index={}",
                        completed_desc.desc_index
                    );

                    queue_ctl
                        .add_used(completed_desc.desc_index, completed_desc.len)
                        .unwrap();

                    queue_ctl
                        .signal_used_queue()
                        .map_err(Error::NotificationFailed)
                        .unwrap();
                    debug!("Notification sent");
                    // interrupt_status.fetch_or(VIRTIO_MMIO_INT_VRING as usize, Ordering::SeqCst);
                    // if let Some(intc) = &intc {
                    //     intc.lock().unwrap().set_irq(irq_line.unwrap());
                    // } else if let Err(e) = interrupt_evt.write(1) {
                    //     error!("Failed to signal used queue: {:?}", e);
                    // }
                } else {
                    i += 1;
                }
            }
            // Update the last completed fence for this context
            fence_state
                .completed_fences
                .insert(ring, completed_fence.fence_id);
        })
    }

    pub fn new(
        queue_ctl: &VringRwLock,
        // interrupt_status: Arc<AtomicUsize>,
        // interrupt_evt: EventFd,
        // intc: Option<Arc<Mutex<Gic>>>,
        // irq_line: Option<u32>,
    ) -> Self {
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

        let fence_state = Arc::new(Mutex::new(Default::default()));
        let fence = Self::create_fence_handler(
            queue_ctl.clone(),
            fence_state.clone(),
            // interrupt_status,
            // interrupt_evt,
            // intc,
            // irq_line,
        );
        let rutabaga = builder
            .build(fence, None)
            .expect("Rutabaga initialization failed!");

        Self {
            rutabaga,
            resources: Default::default(),
            fence_state,
            scanouts: Default::default(),
        }
    }

    // Non-public function -- no doc comment needed!
    fn result_from_query(&mut self, resource_id: u32) -> GpuResponse {
        match self.rutabaga.query(resource_id) {
            Ok(query) => {
                let mut plane_info = Vec::with_capacity(4);
                for plane_index in 0..4 {
                    plane_info.push(GpuResponsePlaneInfo {
                        stride: query.strides[plane_index],
                        offset: query.offsets[plane_index],
                    });
                }
                let format_modifier = query.modifier;
                OkResourcePlaneInfo {
                    format_modifier,
                    plane_info,
                }
            }
            Err(_) => OkNoData,
        }
    }

    fn read_2d_resource(
        &mut self,
        ctx_id: u32,
        resource_id: u32,
        rect: &Rectangle,
        output: &mut [u8],
    ) -> RutabagaResult<()> {
        const BYTES_PER_PIXEL: usize = 4;
        let result_len = rect.width as usize * rect.height as usize * BYTES_PER_PIXEL;
        assert!(output.len() >= result_len);

        let transfer = Transfer3D {
            x: rect.x,
            y: rect.y,
            z: 0,
            w: rect.width,
            h: rect.height,
            d: 1,
            level: 0,
            stride: rect.width * BYTES_PER_PIXEL as u32,
            layer_stride: 0,
            offset: 0,
        };

        self.rutabaga
            .transfer_read(ctx_id, resource_id, transfer, Some(IoSliceMut::new(output)))?;

        Ok(())
    }

    pub fn force_ctx_0(&self) {
        self.rutabaga.force_ctx_0()
    }

    /// Gets the list of supported display resolutions as a slice of `(width, height, enabled)` tuples.
    pub fn display_info(&self, display_info: VirtioGpuRespDisplayInfo) -> Vec<(u32, u32, bool)> {
        display_info
            .pmodes
            .iter()
            .map(|display| (display.r.width, display.r.height, display.enabled == 1))
            .collect::<Vec<_>>()
    }

    /// Gets the EDID for the specified scanout ID. If that scanout is not enabled, it would return
    /// the EDID of a default display.
    pub fn get_edid(
        &self,
        gpu_backend: &mut GpuBackend,
        edid_req: VhostUserGpuEdidRequest,
    ) -> VirtioGpuResult {
        debug!("edid request: {edid_req:?}");
        let edid = gpu_backend.get_edid(&edid_req).map_err(|e| {
            error!("Failed to get edid from frontend: {}", e);
            ErrUnspec
        })?;

        Ok(OkEdid {
            blob: Box::from(&edid.edid[..edid.size as usize]),
        })
    }

    /// Sets the given resource id as the source of scanout to the display.
    pub fn set_scanout(
        &mut self,
        gpu_backend: &mut GpuBackend,
        info: virtio_gpu_set_scanout,
    ) -> VirtioGpuResult {
        let scanout = self
            .scanouts
            .get_mut(info.scanout_id as usize)
            .ok_or(ErrInvalidScanoutId)?;

        // Virtio spec: "The driver can use resource_id = 0 to disable a scanout."
        if info.resource_id == 0 {
            gpu_backend
                .set_scanout(&VhostUserGpuScanout {
                    scanout_id: info.scanout_id,
                    width: info.r.width,
                    height: info.r.height,
                })
                .map_err(|e| {
                    error!("Failed to set_scanout: {e:?}");
                    ErrUnspec
                })?;

            if let Some(resource_id) = scanout.as_ref().map(|scanout| scanout.resource_id) {
                let resource = self
                    .resources
                    .get_mut(&resource_id)
                    .ok_or(ErrInvalidResourceId)?;

                resource.scanouts.disable(info.scanout_id);
            }
            debug!("Disabling scanout {}", info.scanout_id);
            *scanout = None;
            return Ok(OkNoData);
        }

        let resource = self
            .resources
            .get_mut(&info.resource_id)
            .ok_or(ErrInvalidResourceId)?;

        resource.scanouts.enable(info.scanout_id);
        let rect = info.r.into();
        debug!(
            "Enabling scanout: scanout_id: {}, resource_id: {}, {:?}",
            info.scanout_id, info.resource_id, rect
        );

        *scanout = Some(VirtioGpuScanout {
            resource_id: info.resource_id,
            rect,
        });

        Ok(OkNoData)
    }

    /// Creates a 3D resource with the given properties and resource_id.
    pub fn resource_create_3d(
        &mut self,
        resource_id: u32,
        resource_create_3d: ResourceCreate3D,
    ) -> VirtioGpuResult {
        self.rutabaga
            .resource_create_3d(resource_id, resource_create_3d)?;

        let resource = VirtioGpuResource::new(
            resource_id,
            resource_create_3d.width,
            resource_create_3d.height,
            0,
        );

        // Rely on rutabaga to check for duplicate resource ids.
        self.resources.insert(resource_id, resource);
        Ok(self.result_from_query(resource_id))
    }

    /// Releases guest kernel reference on the resource.
    pub fn unref_resource(&mut self, resource_id: u32) -> VirtioGpuResult {
        let resource = self
            .resources
            .remove(&resource_id)
            .ok_or(ErrInvalidResourceId)?;

        if resource.rutabaga_external_mapping {
            self.rutabaga.unmap(resource_id)?;
        }

        self.rutabaga.unref_resource(resource_id)?;
        Ok(OkNoData)
    }

    /// If the resource is the scanout resource, flush it to the display.
    pub fn flush_resource(
        &mut self,
        ctx_id: u32,
        resource_id: u32,
        gpu_backend: &mut GpuBackend,
        rect: Rectangle,
    ) -> VirtioGpuResult {
        if resource_id == 0 {
            return Ok(OkNoData);
        }

        let resource = self
            .resources
            .get(&resource_id)
            .ok_or(ErrInvalidResourceId)?;

        for scanout_id in resource.scanouts.iter_enabled() {
            let scanout  = self.scanouts[scanout_id as usize].as_mut().unwrap();
            let scanout_rect = scanout.rect;
            trace!("Updating scanout {scanout_id} with resource {resource_id}. Updating {rect:?} of {scanout_rect:?}");

            let mut image = vec![0; rect.width as usize * rect.height as usize * 4];
            if let Err(e) = self.read_2d_resource(ctx_id, resource_id, &rect, &mut image) {
                log::error!("Failed to read resource {resource_id} for scanout {scanout_id}: {e}");
                continue;
            }

            gpu_backend
                .update_scanout(
                    &VhostUserGpuUpdate {
                        scanout_id: scanout_id,
                        x: rect.x,
                        y: rect.y,
                        width: rect.width,
                        height: rect.height,
                    },
                    &image,
                )
                .map_err(|e| {
                    error!("Failed to update_scanout: {e:?}");
                    ErrUnspec
                })?
        }

        #[cfg(windows)]
        match self.rutabaga.resource_flush(resource_id) {
            Ok(_) => return Ok(OkNoData),
            Err(RutabagaError::Unsupported) => {}
            Err(e) => return Err(ErrRutabaga(e)),
        }

        Ok(OkNoData)
    }

    /// Copies data to host resource from the attached iovecs. Can also be used to flush caches.
    pub fn transfer_write(
        &mut self,
        ctx_id: u32,
        resource_id: u32,
        mut transfer: Transfer3D,
    ) -> VirtioGpuResult {
        trace!("transfer_write ctx_id {ctx_id}, resource_id {resource_id}, {transfer:?}");

        let resource = self
            .resources
            .get_mut(&resource_id)
            .ok_or(ErrInvalidResourceId)?;

        // FIXME: horrible hack, that makes transfer_read work for scanout!
        if let Some(scanout_id) = resource.scanouts.iter_enabled().next() {
            let rect = self.scanouts[scanout_id as usize].as_ref().unwrap().rect;
            transfer.x = 0;
            transfer.y = 0;
            transfer.w = rect.width;
            transfer.h = rect.height;
        }

        self.rutabaga
            .transfer_write(ctx_id, resource_id, transfer)?;
        Ok(OkNoData)
    }

    /// Copies data from the host resource to:
    ///    1) To the optional volatile slice
    ///    2) To the host resource's attached iovecs
    ///
    /// Can also be used to invalidate caches.
    pub fn transfer_read(
        &mut self,
        _ctx_id: u32,
        _resource_id: u32,
        _transfer: Transfer3D,
        _buf: Option<VolatileSlice>,
    ) -> VirtioGpuResult {
        panic!("virtio_gpu: transfer_read unimplemented");
    }

    /// Attaches backing memory to the given resource, represented by a `Vec` of `(address, size)`
    /// tuples in the guest's physical address space. Converts to RutabagaIovec from the memory
    /// mapping.
    pub fn attach_backing(
        &mut self,
        resource_id: u32,
        mem: &GuestMemoryMmap,
        vecs: Vec<(GuestAddress, usize)>,
    ) -> VirtioGpuResult {
        let rutabaga_iovecs = sglist_to_rutabaga_iovecs(&vecs[..], mem).map_err(|_| ErrUnspec)?;
        self.rutabaga.attach_backing(resource_id, rutabaga_iovecs)?;
        Ok(OkNoData)
    }

    /// Detaches any previously attached iovecs from the resource.
    pub fn detach_backing(&mut self, resource_id: u32) -> VirtioGpuResult {
        self.rutabaga.detach_backing(resource_id)?;
        Ok(OkNoData)
    }

    /// Updates the cursor's memory to the given resource_id, and sets its position to the given
    /// coordinates.
    pub fn update_cursor(
        &mut self,
        _resource_id: u32,
        gpu_backend: &mut GpuBackend,
        cursor_pos: VhostUserGpuCursorPos,
        hot_x: u32,
        hot_y: u32,
        _mem: &GuestMemoryMmap,
    ) -> VirtioGpuResult {
        let size: usize = 4096;
        //TODO: copy data associated with the resource_id
        let data = vec![0u32; size];
        let mut data_array = [0u32; 4096];
        data_array.copy_from_slice(&data[..]);
        let cursor_update: VhostUserGpuCursorUpdate = VhostUserGpuCursorUpdate {
            pos: cursor_pos,
            hot_x,
            hot_y,
            data: data_array,
        };
        gpu_backend.cursor_update(&cursor_update).map_err(|e| {
            error!("Failed to update cursor pos from frontend: {}", e);
            ErrUnspec
        })?;
        //TODO: flush resource
        //self.flush_resource(resource_id, gpu_backend, mem)
        Ok(OkNoData)
    }

    /// Moves the cursor's position to the given coordinates.
    pub fn move_cursor(
        &mut self,
        resource_id: u32,
        gpu_backend: &mut GpuBackend,
        cursor: VhostUserGpuCursorPos,
    ) -> VirtioGpuResult {
        if resource_id == 0 {
            gpu_backend.cursor_pos_hide(&cursor).map_err(|e| {
                error!("Failed to set cursor pos from frontend: {}", e);
                ErrUnspec
            })?;
        } else {
            gpu_backend.cursor_pos(&cursor).map_err(|e| {
                error!("Failed to set cursor pos from frontend: {}", e);
                ErrUnspec
            })?;
        }

        Ok(OkNoData)
    }

    /// Returns a uuid for the resource.
    pub fn resource_assign_uuid(&self, resource_id: u32) -> VirtioGpuResult {
        if !self.resources.contains_key(&resource_id) {
            return Err(ErrInvalidResourceId);
        }

        // TODO(stevensd): use real uuids once the virtio wayland protocol is updated to
        // handle more than 32 bits. For now, the virtwl driver knows that the uuid is
        // actually just the resource id.
        let mut uuid: [u8; 16] = [0; 16];
        for (idx, byte) in resource_id.to_be_bytes().iter().enumerate() {
            uuid[12 + idx] = *byte;
        }
        Ok(OkResourceUuid { uuid })
    }

    /// Gets rutabaga's capset information associated with `index`.
    pub fn get_capset_info(&self, index: u32) -> VirtioGpuResult {
        let (capset_id, version, size) = self.rutabaga.get_capset_info(index)?;
        Ok(OkCapsetInfo {
            capset_id,
            version,
            size,
        })
    }

    /// Gets a capset from rutabaga.
    pub fn get_capset(&self, capset_id: u32, version: u32) -> VirtioGpuResult {
        let capset = self.rutabaga.get_capset(capset_id, version)?;
        Ok(OkCapset(capset))
    }

    /// Creates a rutabaga context.
    pub fn create_context(
        &mut self,
        ctx_id: u32,
        context_init: u32,
        context_name: Option<&str>,
    ) -> VirtioGpuResult {
        self.rutabaga
            .create_context(ctx_id, context_init, context_name)?;
        Ok(OkNoData)
    }

    /// Destroys a rutabaga context.
    pub fn destroy_context(&mut self, ctx_id: u32) -> VirtioGpuResult {
        self.rutabaga.destroy_context(ctx_id)?;
        Ok(OkNoData)
    }

    /// Attaches a resource to a rutabaga context.
    pub fn context_attach_resource(&mut self, ctx_id: u32, resource_id: u32) -> VirtioGpuResult {
        self.rutabaga.context_attach_resource(ctx_id, resource_id)?;
        Ok(OkNoData)
    }

    /// Detaches a resource from a rutabaga context.
    pub fn context_detach_resource(&mut self, ctx_id: u32, resource_id: u32) -> VirtioGpuResult {
        self.rutabaga.context_detach_resource(ctx_id, resource_id)?;
        Ok(OkNoData)
    }

    /// Submits a command buffer to a rutabaga context.
    pub fn submit_command(
        &mut self,
        ctx_id: u32,
        commands: &mut [u8],
        fence_ids: &[u64],
    ) -> VirtioGpuResult {
        self.rutabaga.submit_command(ctx_id, commands, fence_ids)?;
        Ok(OkNoData)
    }

    /// Creates a fence with the RutabagaFence that can be used to determine when the previous
    /// command completed.
    pub fn create_fence(&mut self, rutabaga_fence: RutabagaFence) -> VirtioGpuResult {
        self.rutabaga.create_fence(rutabaga_fence)?;
        Ok(OkNoData)
    }

    pub fn process_fence(
        &mut self,
        ring: VirtioGpuRing,
        fence_id: u64,
        desc_index: u16,
        len: u32,
    ) -> bool {
        // In case the fence is signaled immediately after creation, don't add a return
        // FenceDescriptor.
        let mut fence_state = self.fence_state.lock().unwrap();
        if fence_id > *fence_state.completed_fences.get(&ring).unwrap_or(&0) {
            fence_state.descs.push(FenceDescriptor {
                ring,
                fence_id,
                desc_index,
                len,
            });

            false
        } else {
            true
        }
    }

    /// Creates a blob resource using rutabaga.
    pub fn resource_create_blob(
        &mut self,
        ctx_id: u32,
        resource_id: u32,
        resource_create_blob: ResourceCreateBlob,
        vecs: Vec<(GuestAddress, usize)>,
        mem: &GuestMemoryMmap,
    ) -> VirtioGpuResult {
        let mut rutabaga_iovecs = None;

        if resource_create_blob.blob_flags & VIRTIO_GPU_BLOB_FLAG_CREATE_GUEST_HANDLE != 0 {
            panic!("GUEST_HANDLE unimplemented");
        } else if resource_create_blob.blob_mem != VIRTIO_GPU_BLOB_MEM_HOST3D {
            rutabaga_iovecs =
                Some(sglist_to_rutabaga_iovecs(&vecs[..], mem).map_err(|_| ErrUnspec)?);
        }

        self.rutabaga.resource_create_blob(
            ctx_id,
            resource_id,
            resource_create_blob,
            rutabaga_iovecs,
            None,
        )?;

        let resource = VirtioGpuResource::new(resource_id, 0, 0, resource_create_blob.size);

        // Rely on rutabaga to check for duplicate resource ids.
        self.resources.insert(resource_id, resource);
        Ok(self.result_from_query(resource_id))
    }

    /// Uses the hypervisor to map the rutabaga blob resource.
    ///
    /// When sandboxing is disabled, external_blob is unset and opaque fds are mapped by
    /// rutabaga as ExternalMapping.
    /// When sandboxing is enabled, external_blob is set and opaque fds must be mapped in the
    /// hypervisor process by Vulkano using metadata provided by Rutabaga::vulkan_info().
    pub fn resource_map_blob(
        &mut self,
        resource_id: u32,
        shm_region: &VirtioShmRegion,
        offset: u64,
    ) -> VirtioGpuResult {
        let resource = self
            .resources
            .get_mut(&resource_id)
            .ok_or(ErrInvalidResourceId)?;

        let map_info = self.rutabaga.map_info(resource_id).map_err(|_| ErrUnspec)?;

        if let Ok(export) = self.rutabaga.export_blob(resource_id) {
            if export.handle_type != RUTABAGA_MEM_HANDLE_TYPE_OPAQUE_FD {
                let prot = match map_info & RUTABAGA_MAP_ACCESS_MASK {
                    RUTABAGA_MAP_ACCESS_READ => libc::PROT_READ,
                    RUTABAGA_MAP_ACCESS_WRITE => libc::PROT_WRITE,
                    RUTABAGA_MAP_ACCESS_RW => libc::PROT_READ | libc::PROT_WRITE,
                    _ => panic!("unexpected prot mode for mapping"),
                };

                if offset + resource.size > shm_region.size as u64 {
                    error!("mapping DOES NOT FIT");
                }
                let addr = shm_region.host_addr + offset;
                debug!(
                    "mapping: host_addr={:x}, addr={:x}, size={}",
                    shm_region.host_addr, addr, resource.size
                );
                let ret = unsafe {
                    libc::mmap(
                        addr as *mut libc::c_void,
                        resource.size as usize,
                        prot,
                        libc::MAP_SHARED | libc::MAP_FIXED,
                        export.os_handle.as_raw_fd(),
                        0 as libc::off_t,
                    )
                };
                if ret == libc::MAP_FAILED {
                    return Err(ErrUnspec);
                }
            } else {
                return Err(ErrUnspec);
            }
        } else {
            return Err(ErrUnspec);
        }

        resource.shmem_offset = Some(offset);
        // Access flags not a part of the virtio-gpu spec.
        Ok(OkMapInfo {
            map_info: map_info & RUTABAGA_MAP_CACHE_MASK,
        })
    }

    /// Uses the hypervisor to unmap the blob resource.
    pub fn resource_unmap_blob(
        &mut self,
        resource_id: u32,
        shm_region: &VirtioShmRegion,
    ) -> VirtioGpuResult {
        let resource = self
            .resources
            .get_mut(&resource_id)
            .ok_or(ErrInvalidResourceId)?;

        let shmem_offset = resource.shmem_offset.ok_or(ErrUnspec)?;

        let addr = shm_region.host_addr + shmem_offset;

        let ret = unsafe {
            libc::mmap(
                addr as *mut libc::c_void,
                resource.size as usize,
                libc::PROT_NONE,
                libc::MAP_ANONYMOUS | libc::MAP_PRIVATE | libc::MAP_FIXED,
                -1,
                0_i64,
            )
        };
        if ret == libc::MAP_FAILED {
            panic!("UNMAP failed");
        }

        resource.shmem_offset = None;

        Ok(OkNoData)
    }
}
