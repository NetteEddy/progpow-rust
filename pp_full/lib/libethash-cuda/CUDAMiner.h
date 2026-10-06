/*
This file is part of cpp-ethereum.

cpp-ethereum is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

cpp-ethereum is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with cpp-ethereum.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <iostream>
#include <vector>
#include <time.h>
#include <fstream>
#include <functional>
#include <libethash/ethash.h>
#include <libethcore/EthashAux.h>
#include <libdevcore/Worker.h>
#include <libethcore/miner.h>
#include <libethcore/work.h>
#include "libethash/internal.h"
#include <libprogpow/ProgPow.h>
#include <cuda.h>
#include "CUDAMiner_cuda.h"

using namespace std;

#define DAG_LOAD_MODE_PARALLEL	 0
#define DAG_LOAD_MODE_SEQUENTIAL 1
#define DAG_LOAD_MODE_SINGLE	 2

namespace dev
{
namespace eth
{

struct Solution {
	uint64_t nonce;
	h256 mixHash;

	Solution(uint64_t n, h256 mix){
		nonce = n;
		mixHash = mix;
	}
};

class CUDAMiner: public Miner
{

public:
	Work current;
	uint64_t old_period_seed = -1;

	CUDAMiner(unsigned index);
	~CUDAMiner();

	static unsigned instances()
	{
		return s_numInstances > 0 ? s_numInstances : 1;
	}
	static unsigned getNumDevices();
	static void listDevices();
	static void setParallelHash(unsigned _parallelHash);
	static bool configureGPU(
		unsigned _blockSize,
		unsigned _gridSize,
		unsigned _numStreams,
		unsigned _scheduleFlag,
		uint64_t _currentBlock,
		unsigned _dagLoadMode,
		unsigned _dagCreateDevice,
		bool _noeval,
		bool _exit
		);
	static void setNumInstances(unsigned _instances);
	static void setDevices(const vector<unsigned>& _devices, unsigned _selectedDeviceCount);
	static bool cuda_configureGPU(
		size_t numDevices,
		const vector<int>& _devices,
		unsigned _blockSize,
		unsigned _gridSize,
		unsigned _numStreams,
		unsigned _scheduleFlag,
		uint64_t _currentBlock,
		bool _noeval
		);

	static void cuda_setParallelHash(unsigned _parallelHash);

	bool cuda_init(
		size_t numDevices,
		ethash_light_t _light,
		uint8_t const* _lightData,
		uint64_t _lightSize,
		unsigned _deviceId,
		bool _cpyToHost,
		uint8_t * &hostDAG,
		unsigned dagCreateDevice);

	void search(
		uint8_t const* header,
		uint64_t target,
		bool _ethStratum,
		uint64_t _startN);

	/* -- default values -- */
	/// Default value of the block size. Also known as workgroup size.
	static unsigned const c_defaultBlockSize;
	/// Default value of the grid size
	static unsigned const c_defaultGridSize;
	// default number of CUDA streams
	static unsigned const c_defaultNumStreams;

	static unsigned s_dagCreateDevice;
	static unsigned s_dagLoadMode;

	void compute(const void* header, uint64_t height, int epoch, uint64_t target, uint64_t startNonce) override;
	bool get_solutions(void* data) override;

	static uint8_t* s_dagInHostMemory;

protected:
	void kick_miner();

private:
	//atomic<bool> m_new_work = {false};

	bool init(int epoch);

	uint32_t index;

	hash32_t m_current_header;
	uint64_t m_current_target;
	uint64_t m_current_nonce;
	uint64_t m_starting_nonce;
	uint64_t m_current_index;

	///Constants on GPU
	hash64_t* m_dag = nullptr;
	std::vector<hash64_t*> *m_light;
	uint32_t m_dag_elms = -1;
	uint32_t m_device_num;

	Solution* solution = nullptr;

	CUmodule m_module;
	CUfunction m_kernel;
	volatile search_results** m_search_buf = nullptr;
	cudaStream_t* m_streams = nullptr;

	// Nonce base used by the kernel currently associated with each
	// CUDA stream. This allows completed results to be reconstructed
	// correctly even when the next search window starts elsewhere.
	uint64_t* m_stream_nonce_base = nullptr;

	/// The local work size for the search
	static unsigned s_blockSize;
	/// The initial global work size for the searches
	static unsigned s_gridSize;
	/// The number of CUDA streams
	static unsigned s_numStreams;
	/// CUDA schedule flag
	static unsigned s_scheduleFlag;

	static unsigned m_parallelHash;

	static unsigned s_numInstances;
	static vector<int> s_devices;

	static bool s_noeval;

	void compileKernel(uint64_t block_number, uint64_t dag_words);

};


}
}
