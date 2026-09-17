extern crate libc;
#[macro_use]
extern crate lazy_static;
extern crate bigint;
extern crate dirs;

extern crate progpow_cpu;

#[cfg(feature = "cuda")]
pub extern crate progpow_gpu_cuda as progpow_gpu;
#[cfg(feature = "opencl")]
pub extern crate progpow_gpu_opencl as progpow_gpu;

pub mod hardware;
pub mod types;

#[cfg(test)]
mod test {
	use super::*;

	use bigint::uint::U256;
	use hardware::PpCPU;
	use types::PpCompute;

	#[test]
	fn test_compute_cpu() {
		let height: u64 = 20;
		let nonce: u64 = 10123012301;
		let header_hash: [u8; 32] = [0; 32];
		let pp_cpu = PpCPU::new();
		let (value, mix) = pp_cpu.verify(&header_hash, height, nonce).unwrap();
		assert_eq!(
			mix,
			[
				1067276040, 109748694, 1270962088, 3616890847, 2528371908, 2524623649, 1191460869,
				2529877558
			]
		);
	}

	#[test]
	#[cfg(feature = "cuda")]
	fn test_compute_gpu() {
		use hardware::PpGPU;
		use progpow_gpu::utils::get_gpu_solution;

		let header = [20u8; 32];
		let epoch: i32 = 0;
		let height: u64 = 1;
		let boundary: u64 = u64::MAX;

		let (nonce, mix) = get_gpu_solution(header.clone(), height, epoch, boundary);

		let cpu = PpCPU::new();
		let (_value, cpu_mix) = cpu.verify(&header, height, nonce).unwrap();

		let gpu_mix: [u32; 8] = unsafe { ::std::mem::transmute(mix) };

		println!("CUDA nonce: {}", nonce);
		println!("CUDA mix:   {:?}", gpu_mix);
		println!("CPU mix:    {:?}", cpu_mix);

		assert_eq!(
			gpu_mix,
			cpu_mix,
			"CUDA ProgPoW mix does not match CPU reference"
		);
	}
}
