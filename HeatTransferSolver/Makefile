# Compiler
CXX = g++

# VTK Paths (adjust to your installation paths)
VTK_INC = "C:\Users\srite\Desktop\VCPKG\vcpkg\packages\vtk_x64-windows\include\vtk-9.3"
VTK_LIB = "C:\Users\srite\Desktop\VCPKG\vcpkg\packages\vtk_x64-windows\lib"

# Eigen Paths
EIGEN_INC = "C:\Users\srite\Desktop\VCPKG\vcpkg\packages\eigen3_x64-windows\include\eigen3\Eigen\src"

# VTK Libraries (adjust to your installation)
VTK_LIBS = -lvtkCommonCore-9.0 -lvtkRenderingCore-9.0 -lvtkRenderingOpenGL2-9.0 -lvtkInteractionStyle-9.0 -lvtkRenderingFreeType-9.0 -lvtkFiltersSources-9.0

# Source files
SRCS = main.cpp

# Output binary
TARGET = main

# Compiler flags
CXXFLAGS = -I$(VTK_INC) -I$(EIGEN_INC)
LDFLAGS = -L$(VTK_LIB) $(VTK_LIBS)

# Build target
$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(TARGET) $(SRCS)

# Clean target
clean:
	rm -f $(TARGET)
