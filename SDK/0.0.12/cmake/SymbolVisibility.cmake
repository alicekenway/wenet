function(asr_sdk_apply_hidden_visibility target)
  set_target_properties(${target} PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    C_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
  )
  target_compile_definitions(${target} PRIVATE ASR_SDK_BUILDING_LIBRARY=1)
  if(UNIX AND NOT APPLE)
    target_link_options(${target} PRIVATE
      "LINKER:--exclude-libs,ALL"
    )
  endif()
endfunction()
