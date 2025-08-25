# ---------------------
# Paths and Settings
# ---------------------

MFEM_DIR        = external/mfem
MFEM_BUILD_DIR  = $(MFEM_DIR)/build
MFEM_LIB        = $(MFEM_BUILD_DIR)/libmfem.a
MFEM_INC        = -I. -I$(MFEM_DIR) -I$(MFEM_BUILD_DIR)/config
MFEM_FLAGS      = -std=c++17 -g -O2 $(MFEM_INC)

# You can add other libs (e.g., -lcuda, -fopenmp) as needed
LDFLAGS         = -L$(MFEM_BUILD_DIR) -lmfem -lrt -ldl -lpthread

# ---------------------
# Files
# ---------------------

COMMON_SRC      = InputConfig.cxx DG_Advection.cxx IonizationOperator.cxx
COMMON_OBJ      = $(COMMON_SRC:.cxx=.o)

GUERNICA_SRC    = guernica.cxx
GUERNICA1X_SRC  = guernica1X.cxx

GUERNICA_OBJ    = $(GUERNICA_SRC:.cxx=.o)
GUERNICA1X_OBJ  = $(GUERNICA1X_SRC:.cxx=.o)

# Default target
all: guernica

# ---------------------
# Build Targets
# ---------------------

guernica: $(GUERNICA_OBJ) $(COMMON_OBJ) $(MFEM_LIB)
	$(CXX) $(MFEM_FLAGS) -o $@ $^ $(LDFLAGS)

guernica1X: $(GUERNICA1X_OBJ) $(COMMON_OBJ) $(MFEM_LIB)
	$(CXX) $(MFEM_FLAGS) -o $@ $^ $(LDFLAGS)

# ---------------------
# Build MFEM if needed
# ---------------------

$(MFEM_LIB):
	@echo "MFEM not yet built. Building MFEM with default options..."
	cd $(MFEM_DIR) && \
		mkdir -p build && cd build && \
		cmake .. && make -j2

# ---------------------
# Compilation Rule
# ---------------------

%.o: %.cxx
	$(CXX) $(MFEM_FLAGS) -c $< -o $@

# ---------------------
# Clean Rule
# ---------------------

clean:
	rm -f *.o guernica guernica1X *.gf *.mesh *.vtk
