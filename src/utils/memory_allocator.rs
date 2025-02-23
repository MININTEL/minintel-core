//! Machine Intelligence Node - High-Performance Memory Allocator
//!
//! This module optimizes memory allocation for AI workloads, ensuring 
//! efficient use of CPU and GPU resources.
//!
//! Author: Machine Intelligence Node Development Team

use std::alloc::{alloc, dealloc, Layout};
use std::collections::HashMap;
use std::ptr::null_mut;
use std::sync::Mutex;
use lazy_static::lazy_static;

/// A thread-safe memory allocator that optimizes AI workload efficiency.
pub struct MemoryAllocator {
    allocations: Mutex<HashMap<*mut u8, Layout>>,
}

lazy_static! {
    static ref GLOBAL_ALLOCATOR: MemoryAllocator = MemoryAllocator::new();
}

impl MemoryAllocator {
    /// Initializes a new memory allocator instance.
    pub const fn new() -> Self {
        MemoryAllocator {
            allocations: Mutex::new(HashMap::new()),
        }
    }

    /// Allocates memory with a given size and alignment.
    ///
    /// # Arguments
    /// * `size` - The size of the memory block.
    /// * `alignment` - The memory alignment requirement.
    ///
    /// # Returns
    /// * A pointer to the allocated memory block.
    pub fn allocate(&self, size: usize, alignment: usize) -> *mut u8 {
        let layout = Layout::from_size_align(size, alignment).unwrap();
        let ptr = unsafe { alloc(layout) };

        if ptr.is_null() {
            panic!("Memory allocation failed");
        }

        self.allocations.lock().unwrap().insert(ptr, layout);
        ptr
    }

    /// Deallocates a previously allocated memory block.
    ///
    /// # Arguments
    /// * `ptr` - A pointer to the memory block to be deallocated.
    pub fn deallocate(&self, ptr: *mut u8) {
        let mut allocations = self.allocations.lock().unwrap();
        if let Some(layout) = allocations.remove(&ptr) {
            unsafe {
                dealloc(ptr, layout);
            }
        }
    }

    /// Returns the number of active allocations.
    pub fn active_allocations(&self) -> usize {
        self.allocations.lock().unwrap().len()
    }
}

/// GPU Memory Tracking (Optional CUDA Integration)
#[cfg(feature = "cuda")]
mod gpu_memory {
    use cuda_sys::cuda::*;
    use std::sync::Mutex;

    lazy_static! {
        static ref GPU_MEMORY_USAGE: Mutex<u64> = Mutex::new(0);
    }

    pub fn allocate_gpu_memory(size: usize) -> *mut std::ffi::c_void {
        let mut dev_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        unsafe {
            if cudaMalloc(&mut dev_ptr, size) != cudaError::CUDA_SUCCESS {
                panic!("GPU memory allocation failed");
            }
        }
        *GPU_MEMORY_USAGE.lock().unwrap() += size as u64;
        dev_ptr
    }

    pub fn deallocate_gpu_memory(ptr: *mut std::ffi::c_void, size: usize) {
        unsafe {
            cudaFree(ptr);
        }
        *GPU_MEMORY_USAGE.lock().unwrap() -= size as u64;
    }

    pub fn get_gpu_memory_usage() -> u64 {
        *GPU_MEMORY_USAGE.lock().unwrap()
    }
}

/// Example Usage
fn main() {
    let allocator = &GLOBAL_ALLOCATOR;

    // Allocate and deallocate CPU memory
    let mem_block = allocator.allocate(1024, 8);
    println!("Allocated 1024 bytes");
    allocator.deallocate(mem_block);
    println!("Deallocated memory");

    #[cfg(feature = "cuda")]
    {
        let gpu_mem = gpu_memory::allocate_gpu_memory(1024 * 1024);
        println!("Allocated 1MB on GPU");
        gpu_memory::deallocate_gpu_memory(gpu_mem, 1024 * 1024);
        println!("Deallocated GPU memory");
    }
}
