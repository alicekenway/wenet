include(GNUInstallDirs)

install(TARGETS asr_sdk
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/asr_sdk
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if(ASR_SDK_BUILD_TOOLS)
  install(TARGETS asr_stream_file asr_batch_decode inspect_package print_build_info
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
endif()

install(FILES "${ASR_SDK_ONNXRUNTIME_LIB}"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RENAME libonnxruntime.so.1.23.2
)

install(CODE "
  execute_process(COMMAND \${CMAKE_COMMAND} -E create_symlink libonnxruntime.so.1.23.2 \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/libonnxruntime.so.1\")
  execute_process(COMMAND \${CMAKE_COMMAND} -E create_symlink libonnxruntime.so.1.23.2 \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/libonnxruntime.so\")
")
