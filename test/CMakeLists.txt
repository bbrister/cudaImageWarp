# Try to SIFT3D
find_package (SIFT3D)

# If SIFT3D was found, build the C test program
if (SIFT3D_FOUND)
    add_executable (c_test tester.c)
    target_include_directories (c_test PRIVATE ${SIFT3D_INCLUDE_DIRS})
    target_link_libraries (c_test PRIVATE cudaImageWarp ${SIFT3D_LIBRARIES})
    set_target_properties (c_test PROPERTIES LINKER_LANGUAGE CXX)
else ()
       message (STATUS "SIFT3D not found. Skipping the C test program.") 
endif ()

