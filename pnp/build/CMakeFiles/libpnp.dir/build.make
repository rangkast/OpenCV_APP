# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/rangkast.jeong/.local/lib/python3.8/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/rangkast.jeong/.local/lib/python3.8/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/rangkast.jeong/Project/pnp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/rangkast.jeong/Project/pnp/build

# Include any dependencies generated for this target.
include CMakeFiles/libpnp.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/libpnp.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/libpnp.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/libpnp.dir/flags.make

CMakeFiles/libpnp.dir/utils/mlibtime.cpp.o: CMakeFiles/libpnp.dir/flags.make
CMakeFiles/libpnp.dir/utils/mlibtime.cpp.o: /home/rangkast.jeong/Project/pnp/utils/mlibtime.cpp
CMakeFiles/libpnp.dir/utils/mlibtime.cpp.o: CMakeFiles/libpnp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rangkast.jeong/Project/pnp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/libpnp.dir/utils/mlibtime.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/libpnp.dir/utils/mlibtime.cpp.o -MF CMakeFiles/libpnp.dir/utils/mlibtime.cpp.o.d -o CMakeFiles/libpnp.dir/utils/mlibtime.cpp.o -c /home/rangkast.jeong/Project/pnp/utils/mlibtime.cpp

CMakeFiles/libpnp.dir/utils/mlibtime.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/libpnp.dir/utils/mlibtime.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/rangkast.jeong/Project/pnp/utils/mlibtime.cpp > CMakeFiles/libpnp.dir/utils/mlibtime.cpp.i

CMakeFiles/libpnp.dir/utils/mlibtime.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/libpnp.dir/utils/mlibtime.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/rangkast.jeong/Project/pnp/utils/mlibtime.cpp -o CMakeFiles/libpnp.dir/utils/mlibtime.cpp.s

CMakeFiles/libpnp.dir/pnp_ransac.cpp.o: CMakeFiles/libpnp.dir/flags.make
CMakeFiles/libpnp.dir/pnp_ransac.cpp.o: /home/rangkast.jeong/Project/pnp/pnp_ransac.cpp
CMakeFiles/libpnp.dir/pnp_ransac.cpp.o: CMakeFiles/libpnp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rangkast.jeong/Project/pnp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/libpnp.dir/pnp_ransac.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/libpnp.dir/pnp_ransac.cpp.o -MF CMakeFiles/libpnp.dir/pnp_ransac.cpp.o.d -o CMakeFiles/libpnp.dir/pnp_ransac.cpp.o -c /home/rangkast.jeong/Project/pnp/pnp_ransac.cpp

CMakeFiles/libpnp.dir/pnp_ransac.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/libpnp.dir/pnp_ransac.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/rangkast.jeong/Project/pnp/pnp_ransac.cpp > CMakeFiles/libpnp.dir/pnp_ransac.cpp.i

CMakeFiles/libpnp.dir/pnp_ransac.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/libpnp.dir/pnp_ransac.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/rangkast.jeong/Project/pnp/pnp_ransac.cpp -o CMakeFiles/libpnp.dir/pnp_ransac.cpp.s

CMakeFiles/libpnp.dir/p4p.cpp.o: CMakeFiles/libpnp.dir/flags.make
CMakeFiles/libpnp.dir/p4p.cpp.o: /home/rangkast.jeong/Project/pnp/p4p.cpp
CMakeFiles/libpnp.dir/p4p.cpp.o: CMakeFiles/libpnp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rangkast.jeong/Project/pnp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/libpnp.dir/p4p.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/libpnp.dir/p4p.cpp.o -MF CMakeFiles/libpnp.dir/p4p.cpp.o.d -o CMakeFiles/libpnp.dir/p4p.cpp.o -c /home/rangkast.jeong/Project/pnp/p4p.cpp

CMakeFiles/libpnp.dir/p4p.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/libpnp.dir/p4p.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/rangkast.jeong/Project/pnp/p4p.cpp > CMakeFiles/libpnp.dir/p4p.cpp.i

CMakeFiles/libpnp.dir/p4p.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/libpnp.dir/p4p.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/rangkast.jeong/Project/pnp/p4p.cpp -o CMakeFiles/libpnp.dir/p4p.cpp.s

# Object files for target libpnp
libpnp_OBJECTS = \
"CMakeFiles/libpnp.dir/utils/mlibtime.cpp.o" \
"CMakeFiles/libpnp.dir/pnp_ransac.cpp.o" \
"CMakeFiles/libpnp.dir/p4p.cpp.o"

# External object files for target libpnp
libpnp_EXTERNAL_OBJECTS =

liblibpnp.a: CMakeFiles/libpnp.dir/utils/mlibtime.cpp.o
liblibpnp.a: CMakeFiles/libpnp.dir/pnp_ransac.cpp.o
liblibpnp.a: CMakeFiles/libpnp.dir/p4p.cpp.o
liblibpnp.a: CMakeFiles/libpnp.dir/build.make
liblibpnp.a: CMakeFiles/libpnp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/rangkast.jeong/Project/pnp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX static library liblibpnp.a"
	$(CMAKE_COMMAND) -P CMakeFiles/libpnp.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/libpnp.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/libpnp.dir/build: liblibpnp.a
.PHONY : CMakeFiles/libpnp.dir/build

CMakeFiles/libpnp.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/libpnp.dir/cmake_clean.cmake
.PHONY : CMakeFiles/libpnp.dir/clean

CMakeFiles/libpnp.dir/depend:
	cd /home/rangkast.jeong/Project/pnp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/rangkast.jeong/Project/pnp /home/rangkast.jeong/Project/pnp /home/rangkast.jeong/Project/pnp/build /home/rangkast.jeong/Project/pnp/build /home/rangkast.jeong/Project/pnp/build/CMakeFiles/libpnp.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/libpnp.dir/depend
