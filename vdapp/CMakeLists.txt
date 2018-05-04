set(SOURCES
    main.cpp
)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_executable(vdapp ${SOURCES})

target_link_libraries(vdapp)
