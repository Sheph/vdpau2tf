set(SOURCES
    cuviddec.h
    FFmpegDemuxer.h
    Logger.h
    NvCodecUtils.h
    nvcuvid.h
    Resize.cu
    main.cpp
)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

cuda_add_executable(cuvidapp ${SOURCES} OPTIONS -std=c++11)

target_link_libraries(cuvidapp ${OpenCV_LIBS} ${TF_LIBRARY} ${TF_FWK_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY} ${AVCODEC_LIBRARIES} ${AVFORMAT_LIBRARIES} ${AVUTIL_LIBRARIES} ${NVIDIA_ML_LIBRARY} rt dl cuda nvcuvid)

configure_file(car_cuvid.pb ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/car_cuvid.pb COPYONLY)
