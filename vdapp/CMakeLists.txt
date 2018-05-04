set(SOURCES
    main.cpp
)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_executable(vdapp ${SOURCES})

target_link_libraries(vdapp ${X11_LIBRARIES} ${X11_Xxf86vm_LIB} ${VDPAU_LIBRARIES} rt dl)
