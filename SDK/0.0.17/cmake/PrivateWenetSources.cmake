# Rebuild private dependencies so compiler path mapping covers every object.
set(_fc "${ASR_SDK_WENET_ROOT}/runtime/onnxruntime/fc_base")
set(_onnx "${ASR_SDK_WENET_ROOT}/runtime/onnxruntime")
add_library(asr_private_gflags STATIC
  "${_fc}/gflags-src/src/gflags.cc"
  "${_fc}/gflags-src/src/gflags_reporting.cc"
  "${_fc}/gflags-src/src/gflags_completions.cc"
)
target_include_directories(asr_private_gflags PRIVATE
  "${_fc}/gflags-src/src" "${_fc}/gflags-build/include" "${_fc}/gflags-build/include/gflags"
  "${_fc}/glog-src/src" "${_fc}/glog-build" "${_fc}/openfst-src/src/include")
set_target_properties(asr_private_gflags PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(asr_private_glog STATIC
  "${_fc}/glog-src/src/demangle.cc"
  "${_fc}/glog-src/src/logging.cc"
  "${_fc}/glog-src/src/raw_logging.cc"
  "${_fc}/glog-src/src/symbolize.cc"
  "${_fc}/glog-src/src/utilities.cc"
  "${_fc}/glog-src/src/vlog_is_on.cc"
  "${_fc}/glog-src/src/signalhandler.cc"
)
target_include_directories(asr_private_glog PRIVATE
  "${_fc}/gflags-src/src" "${_fc}/gflags-build/include" "${_fc}/gflags-build/include/gflags"
  "${_fc}/glog-src/src" "${_fc}/glog-build" "${_fc}/openfst-src/src/include")
set_target_properties(asr_private_glog PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(asr_private_fst STATIC
  "${_fc}/openfst-src/src/lib/compat.cc"
  "${_fc}/openfst-src/src/lib/flags.cc"
  "${_fc}/openfst-src/src/lib/fst-types.cc"
  "${_fc}/openfst-src/src/lib/fst.cc"
  "${_fc}/openfst-src/src/lib/mapped-file.cc"
  "${_fc}/openfst-src/src/lib/properties.cc"
  "${_fc}/openfst-src/src/lib/symbol-table.cc"
  "${_fc}/openfst-src/src/lib/symbol-table-ops.cc"
  "${_fc}/openfst-src/src/lib/util.cc"
  "${_fc}/openfst-src/src/lib/weight.cc"
)
target_include_directories(asr_private_fst PRIVATE
  "${_fc}/gflags-src/src" "${_fc}/gflags-build/include" "${_fc}/gflags-build/include/gflags"
  "${_fc}/glog-src/src" "${_fc}/glog-build" "${_fc}/openfst-src/src/include")
set_target_properties(asr_private_fst PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(asr_private_gflags PRIVATE GFLAGS_IS_A_DLL=0 NO_THREADS)
target_include_directories(asr_private_glog BEFORE PRIVATE "${_fc}/glog-src/src" "${_fc}/glog-build")
target_compile_definitions(asr_private_glog PRIVATE "GOOGLE_GLOG_DLL_DECL=")
target_link_libraries(asr_private_glog PUBLIC asr_private_gflags pthread)
add_library(asr_private_wenet STATIC
  "${_onnx}/decoder/asr_decoder.cc"
  "${_onnx}/decoder/asr_model.cc"
  "${_onnx}/decoder/context_graph.cc"
  "${_onnx}/decoder/ctc_prefix_beam_search.cc"
  "${_onnx}/decoder/ctc_wfst_beam_search.cc"
  "${_onnx}/decoder/ctc_endpoint.cc"
  "${_onnx}/kaldi/base/kaldi-error.cc"
  "${_onnx}/kaldi/base/kaldi-math.cc"
  "${_onnx}/kaldi/util/kaldi-io.cc"
  "${_onnx}/kaldi/util/parse-options.cc"
  "${_onnx}/kaldi/util/simple-io-funcs.cc"
  "${_onnx}/kaldi/util/text-utils.cc"
  "${_onnx}/kaldi/lat/determinize-lattice-pruned.cc"
  "${_onnx}/kaldi/lat/lattice-functions.cc"
  "${_onnx}/kaldi/decoder/lattice-faster-decoder.cc"
  "${_onnx}/kaldi/decoder/lattice-faster-online-decoder.cc"
  "${_onnx}/frontend/feature_pipeline.cc"
  "${_onnx}/frontend/fft.cc"
  "${_onnx}/post_processor/post_processor.cc"
  "${_onnx}/utils/string.cc"
  "${_onnx}/utils/utils.cc"
)
target_include_directories(asr_private_wenet BEFORE PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/src/wenet_bridge/private_wenet")
target_include_directories(asr_private_wenet PRIVATE ${ASR_SDK_WENET_INCLUDE_DIRS}
  "${ASR_SDK_WETEXT_ROOT}/runtime")
target_compile_definitions(asr_private_wenet PRIVATE USE_ONNX)
set_target_properties(asr_private_wenet PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(asr_private_wenet PUBLIC asr_private_fst asr_private_glog asr_private_gflags)
