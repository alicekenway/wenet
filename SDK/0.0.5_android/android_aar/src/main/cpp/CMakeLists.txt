cmake_minimum_required(VERSION 3.22.1)
project(wenet_asr_jni LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(ASR_JNI_SDK_ROOT "" CACHE PATH "Path to SDK root")
set(ASR_JNI_LIB_DIR "" CACHE PATH "Path containing ABI subdirectories with libasr_sdk.so")

if(NOT ASR_JNI_SDK_ROOT)
  message(FATAL_ERROR "ASR_JNI_SDK_ROOT is required")
endif()
if(NOT ASR_JNI_LIB_DIR)
  message(FATAL_ERROR "ASR_JNI_LIB_DIR is required")
endif()

add_library(asr_sdk SHARED IMPORTED)
set_target_properties(asr_sdk PROPERTIES
  IMPORTED_LOCATION "${ASR_JNI_LIB_DIR}/${ANDROID_ABI}/libasr_sdk.so"
)
target_include_directories(asr_sdk INTERFACE
  "${ASR_JNI_SDK_ROOT}/include"
)

find_library(log-lib log)

add_library(asr_jni SHARED asr_jni.cpp)
target_link_libraries(asr_jni PRIVATE asr_sdk ${log-lib})
