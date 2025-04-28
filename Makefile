CUTLASS_DIR := csrc/cutlass
CUTLASS_BUILD_DIR := $(CUTLASS_DIR)/build
CUTLASS_BUILD_FLAGS := -DCUTLASS_NVCC_ARCHS="89" -DCUTLASS_ENABLE_EXAMPLES=OFF -DCUTLASS_ENABLE_TOOLS=OFF -DCUTLASS_LIBRARY_KERNELS=ON -DCUTLASS_ENABLE_PROFILER=OFF -DCUTLASS_ENABLE_PERFORMANCE=OFF -DCUTLASS_ENABLE_TESTS=OFF

NPROC := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
THREADS := $(shell echo $$(($(NPROC) / 2)))

cutlass: update-submodule-cutlass create-cutlass-build-dir configure-cutlass build-cutlass

update-submodule-cutlass:
	@echo "Initializing and updating CUTLASS submodule."
	git submodule update --init --recursive $(CUTLASS_DIR)

create-cutlass-build-dir:
	@echo "Making CUTLASS build directory."
	mkdir -p $(CUTLASS_BUILD_DIR)

configure-cutlass:
	@echo "Configuring CUTLASS."
	cd $(CUTLASS_BUILD_DIR) && cmake .. -DCMAKE_BUILD_TYPE=Release $(CUTLASS_BUILD_FLAGS)

build-cutlass:
	@echo "Building CUTLASS."
	cd $(CUTLASS_BUILD_DIR) && make -j$(THREADS)

clean-cutlass:
	@echo "Cleaning CUTLASS build."
	rm -rf $(CUTLASS_BUILD_DIR)

.PHONY: cutlass update-submodule-cutlass create-cutlass-build-dir configure-cutlass build-cutlass clean-cutlass