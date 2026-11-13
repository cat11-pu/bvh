# CMake generated Testfile for 
# Source directory: /home/cat/gsb/uid3676/test
# Build directory: /home/cat/gsb/uid3676/build-dbg/test
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[simple_example]=] "/home/cat/gsb/uid3676/build-dbg/bin/simple_example")
set_tests_properties([=[simple_example]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/cat/gsb/uid3676/test/CMakeLists.txt;8;add_test;/home/cat/gsb/uid3676/test/CMakeLists.txt;0;")
add_test([=[threadpool_test]=] "/home/cat/gsb/uid3676/build-dbg/bin/threadpool_test")
set_tests_properties([=[threadpool_test]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/cat/gsb/uid3676/test/CMakeLists.txt;13;add_test;/home/cat/gsb/uid3676/test/CMakeLists.txt;0;")
add_test([=[serialize]=] "/home/cat/gsb/uid3676/build-dbg/bin/serialize")
set_tests_properties([=[serialize]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/cat/gsb/uid3676/test/CMakeLists.txt;18;add_test;/home/cat/gsb/uid3676/test/CMakeLists.txt;0;")
add_test([=[benchmark]=] "/home/cat/gsb/uid3676/build-dbg/bin/benchmark" "/home/cat/gsb/uid3676/test/scenes/cornell_box.obj" "--eye" "0" "1" "2" "--dir" "0" "0" "-1" "--up" "0" "1" "0")
set_tests_properties([=[benchmark]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/cat/gsb/uid3676/test/CMakeLists.txt;23;add_test;/home/cat/gsb/uid3676/test/CMakeLists.txt;0;")
add_test([=[motion_blur_test]=] "/home/cat/gsb/uid3676/build-dbg/bin/motion_blur_test")
set_tests_properties([=[motion_blur_test]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/cat/gsb/uid3676/test/CMakeLists.txt;34;add_test;/home/cat/gsb/uid3676/test/CMakeLists.txt;0;")
