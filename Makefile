BUILD_DIR = priv
C_SRC_DIR = c_src

SRC = $(C_SRC_DIR)/gpu_nifs.cpp
TARGET = $(BUILD_DIR)/gpu_nifs.so
DEPENDENCIES = $(C_SRC_DIR)/ocl_interface/OCLInterface.cpp

CXX = g++
CXXFLAGS = -shared -fPIC -Wall -Wextra -std=c++17
LINKFLAGS = -lOpenCL

all: $(BUILD_DIR) $(TARGET)

$(TARGET):
	$(CXX) $(CXXFLAGS) $(DEPENDENCIES) $(SRC) -o $@ $(LINKFLAGS)

# bmp: c_src/bmp_nifs.cu 
# 	nvcc --shared -g --compiler-options '-fPIC' -o priv/bmp_nifs.so c_src/bmp_nifs.cu

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)/*.so

.PHONY: all clean
