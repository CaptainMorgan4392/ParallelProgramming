# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.19

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
CMAKE_COMMAND = "/Users/captainmorgan/Library/Application Support/JetBrains/Toolbox/apps/CLion/ch-0/211.6693.114/CLion.app/Contents/bin/cmake/mac/bin/cmake"

# The command to remove a file.
RM = "/Users/captainmorgan/Library/Application Support/JetBrains/Toolbox/apps/CLion/ch-0/211.6693.114/CLion.app/Contents/bin/cmake/mac/bin/cmake" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/captainmorgan/CLionProjects/parallelLab1

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/captainmorgan/CLionProjects/parallelLab1/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/parallelLab1.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/parallelLab1.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/parallelLab1.dir/flags.make

CMakeFiles/parallelLab1.dir/main.c.o: CMakeFiles/parallelLab1.dir/flags.make
CMakeFiles/parallelLab1.dir/main.c.o: ../main.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/captainmorgan/CLionProjects/parallelLab1/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/parallelLab1.dir/main.c.o"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/parallelLab1.dir/main.c.o -c /Users/captainmorgan/CLionProjects/parallelLab1/main.c

CMakeFiles/parallelLab1.dir/main.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/parallelLab1.dir/main.c.i"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/captainmorgan/CLionProjects/parallelLab1/main.c > CMakeFiles/parallelLab1.dir/main.c.i

CMakeFiles/parallelLab1.dir/main.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/parallelLab1.dir/main.c.s"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/captainmorgan/CLionProjects/parallelLab1/main.c -o CMakeFiles/parallelLab1.dir/main.c.s

# Object files for target parallelLab1
parallelLab1_OBJECTS = \
"CMakeFiles/parallelLab1.dir/main.c.o"

# External object files for target parallelLab1
parallelLab1_EXTERNAL_OBJECTS =

parallelLab1: CMakeFiles/parallelLab1.dir/main.c.o
parallelLab1: CMakeFiles/parallelLab1.dir/build.make
parallelLab1: CMakeFiles/parallelLab1.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/captainmorgan/CLionProjects/parallelLab1/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable parallelLab1"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/parallelLab1.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/parallelLab1.dir/build: parallelLab1

.PHONY : CMakeFiles/parallelLab1.dir/build

CMakeFiles/parallelLab1.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/parallelLab1.dir/cmake_clean.cmake
.PHONY : CMakeFiles/parallelLab1.dir/clean

CMakeFiles/parallelLab1.dir/depend:
	cd /Users/captainmorgan/CLionProjects/parallelLab1/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/captainmorgan/CLionProjects/parallelLab1 /Users/captainmorgan/CLionProjects/parallelLab1 /Users/captainmorgan/CLionProjects/parallelLab1/cmake-build-debug /Users/captainmorgan/CLionProjects/parallelLab1/cmake-build-debug /Users/captainmorgan/CLionProjects/parallelLab1/cmake-build-debug/CMakeFiles/parallelLab1.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/parallelLab1.dir/depend

