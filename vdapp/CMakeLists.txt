set(SOURCES
    main.cpp
)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_executable(vdapp ${SOURCES})

target_link_libraries(vdapp ${OpenCV_LIBS} ${X11_LIBRARIES} ${X11_Xxf86vm_LIB} ${VDPAU_LIBRARIES} ${TF_LIBRARY} ${TF_FWK_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY} rt dl)

configure_file(car.pb ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/car.pb COPYONLY)
configure_file(car.png ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/car.png COPYONLY)
