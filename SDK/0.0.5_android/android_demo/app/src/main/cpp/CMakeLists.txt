cmake_minimum_required(VERSION 3.22.1)
project(asr_jni LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(asr_sdk SHARED IMPORTED)
set_target_properties(asr_sdk PROPERTIES
  IMPORTED_LOCATION
  "${CMAKE_CURRENT_LIST_DIR}/../jniLibs/${ANDROID_ABI}/libasr_sdk.so"
)
target_include_directories(asr_sdk INTERFACE
  "${CMAKE_CURRENT_LIST_DIR}/../../../../../include"
)

find_library(log-lib log)

add_library(asr_jni SHARED asr_jni.cpp)
target_link_libraries(asr_jni PRIVATE asr_sdk ${log-lib})
