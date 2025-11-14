BUILD_DIR = priv
C_SRC_DIR = c_src

SRC = $(C_SRC_DIR)/gpu_nifs.cpp
TARGET = $(BUILD_DIR)/gpu_nifs.so
DEPENDENCIES = $(C_SRC_DIR)/ocl_interface/OCLInterface.cpp
HEADER_DEPENDENCIES = $(C_SRC_DIR)/cldef.hpp $(C_SRC_DIR)/ocl_interface/OCLInterface.hpp

BMP_SRC = $(C_SRC_DIR)/bmp_nifs.cpp
BMP_TARGET = $(BUILD_DIR)/bmp_nifs.so
BMP_DEPENDENCIES = $(C_SRC_DIR)/bmp/BMP.cpp
BMP_HEADER_DEPENDENCIES = $(C_SRC_DIR)/bmp/BMP.hpp

CXX = g++
CXXFLAGS = -shared -fPIC -Wall -Wextra -std=c++17
LINKFLAGS = -lOpenCL

all: $(TARGET)
bmp: $(BUILD_DIR) $(BMP_TARGET)

$(TARGET): $(SRC) $(DEPENDENCIES) $(HEADER_DEPENDENCIES)
	$(CXX) $(CXXFLAGS) $(DEPENDENCIES) $(SRC) -o $(TARGET) $(LINKFLAGS)

$(BMP_TARGET): $(BMP_SRC) $(BMP_DEPENDENCIES) $(BMP_HEADER_DEPENDENCIES)
	$(CXX) $(CXXFLAGS) $(BMP_DEPENDENCIES) $(BMP_SRC) -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)/*.so

.PHONY: all bmp clean
