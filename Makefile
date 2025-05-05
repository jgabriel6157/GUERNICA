# ---------------------
# Paths and Settings
# ---------------------

MFEM_DIR        = external/mfem
MFEM_BUILD_DIR  = $(MFEM_DIR)/build
MFEM_LIB        = $(MFEM_BUILD_DIR)/libmfem.a
MFEM_INC        = -I$(MFEM_DIR) -I$(MFEM_BUILD_DIR)/config
MFEM_FLAGS      = -std=c++17 -g -O2 $(MFEM_INC)

# You can add other libs (e.g., -lcuda, -fopenmp) as needed
LDFLAGS         = -L$(MFEM_BUILD_DIR) -lmfem -lrt -ldl -lpthread

# ---------------------
# Files
# ---------------------

TARGET          = guernica
SOURCES         = guernica.cxx

# ---------------------
# Build Rules
# ---------------------

all: $(TARGET)

$(TARGET): $(SOURCES) $(MFEM_LIB)
	$(CXX) $(MFEM_FLAGS) -o $@ $^ $(LDFLAGS)

$(MFEM_LIB):
	@echo "MFEM not yet built. Building MFEM with default options..."
	cd $(MFEM_DIR) && \
		mkdir -p build && cd build && \
		cmake .. && make -j2

clean:
	rm -f $(TARGET)
