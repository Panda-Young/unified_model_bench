/* COPYRIGHT HEADER GOES HERE: No CopyRight Header String Passed During Model Conversion */

/* Command Line used:
qnn-onnx-converter; act_bitwidth=8; act_quantizer=tf; act_quantizer_calibration=min-max; act_quantizer_schema=asymmetric; adjust_nms_features_dims=True; algorithms=[]; align_matmul_ranks=True; apply_masked_softmax=uncompressed; arch_checker=False; backend=None; batch=None; bias_bitwidth=8; calc_static_encodings=False; converter_op_package_lib=; copyright_file=None; custom_io=; custom_op_config_paths=None; debug=None; defer_loading=False; define_symbol=None; disable_batchnorm_folding=False; disable_defer_loading=False; disable_node_validation=False; disable_qnn_op_config_validation=False; disable_relu_squashing=False; dry_run=None; dumpIR=False; dump_custom_io_config_template=; dump_encoding_json=False; dump_inferred_model=False; dump_ir=; dump_ir_optimizer_config_template=False; dump_optimization_pass_mode_config=False; dump_pass_trace_info=False; dump_qairt_io_config_yaml=; dump_qairt_quantizer_command=None; dump_value_info=False; enable_framework_trace=False; enable_match_gathernd=False; enable_match_topk=False; enable_per_row_quantized_bias=False; exclude_named_tensors=False; expand_gru_op_structure=True; expand_lstm_op_structure=False; expand_sparse_op_structure=False; export_format=cpp; extract_color_transform=True; float_bias_bitwidth=0; float_bias_bw=0; float_bitwidth=32; float_bw=32; float_fallback=False; force_prune_cast_ops=False; handle_gather_negative_indices=True; ignore_encodings=False; include_data_invariant_ops=False; inject_cast_for_gather=True; input_dim=None; input_dtype=[]; input_encoding=[]; input_layout=[]; input_list=None; input_type=[]; ir_optimizer_config=; keep_disconnected_nodes=False; keep_int64_inputs=False; keep_quant_nodes=False; keep_weights_quantized=False; match_caffe_ssd_to_tf=True; model_version=None; multi_time_steps_gru=False; multi_time_steps_lstm=False; no_simplification=False; op_package_lib=; optimization_pass_mode=ir_optimizer_mainline; optimization_pass_mode_config=; out_names=['output']; overwrite_model_prefix=False; pack_4_bit_weights=False; package_name=None; packed_masked_softmax_inputs=[]; packed_max_seq=1; param_quantizer=None; param_quantizer_calibration=min-max; param_quantizer_schema=asymmetric; percentile_calibration_value=99.99; perform_axes_to_spatial_first_order=True; perform_layout_transformation=False; prepare_inputs_as_params=False; preprocess_roi_pool_inputs=True; preserve_io=[['layout']]; preserve_onnx_output_order=False; quantization_overrides=; quantizer_log=None; quantizer_log_level=LogLevel.NONE; restrict_quantization_steps=[]; squash_box_decoder=True; unroll_gru_time_steps=True; unroll_lstm_time_steps=True; use_aimet_quantizer=False; use_convert_quantization_nodes=False; use_dynamic_16_bit_weights=False; use_native_dtype=False; use_native_input_files=False; use_native_output_files=False; use_per_channel_quantization=False; use_per_row_quantization=False; use_quantize_v2=False; validate_models=False; weights_bitwidth=8
*/

#include "QnnOpDef.h"
#include "QnnModel.hpp"

// Flag to determine if Backend should do node validation for each opNode added
#define DO_GRAPH_NODE_VALIDATIONS 1

using namespace qnn_wrapper_api;
const __attribute__((visibility("default"))) char* QNN_SDK_VERSION = "qaisw-v2.42.0.251225135753_193295";
extern "C" {
static ModelError_t addTensor_input_a(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_input_a[] = {1, 3, 224, 224};
  VALIDATE(model.addTensor("input_a", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "input_a",
                                 .type= QNN_TENSOR_TYPE_APP_WRITE,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_input_a,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=nullptr,
                                                .dataSize=0}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_input_b(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_input_b[] = {1, 48, 1, 1};
  VALIDATE(model.addTensor("input_b", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "input_b",
                                 .type= QNN_TENSOR_TYPE_APP_WRITE,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions_input_b,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=nullptr,
                                                .dataSize=0}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode_input_a_nhwc(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR input_a_nhwc */
  uint32_t dimensions_input_a_nhwc_perm[] = {4};
  uint32_t input_a_nhwc_perm[] = {0, 2, 3, 1};
  Qnn_Param_t params_input_a_nhwc[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "input_a_nhwc_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_input_a_nhwc_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)input_a_nhwc_perm,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs_input_a_nhwc[] = {
    "input_a"
  };
  uint32_t dimensions_input_a_nhwc[] = {1, 224, 224, 3};
  Qnn_Tensor_t outputs_input_a_nhwc[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "input_a_nhwc",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions_input_a_nhwc,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "input_a_nhwc", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params_input_a_nhwc, // Node Params
                         1, // Num Node Params
                         inputs_input_a_nhwc, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_input_a_nhwc, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_input_b_nhwc(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR input_b_nhwc */
  uint32_t dimensions_input_b_nhwc_perm[] = {4};
  uint32_t input_b_nhwc_perm[] = {0, 2, 3, 1};
  Qnn_Param_t params_input_b_nhwc[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "input_b_nhwc_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_input_b_nhwc_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)input_b_nhwc_perm,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs_input_b_nhwc[] = {
    "input_b"
  };
  uint32_t dimensions_input_b_nhwc[] = {1, 1, 1, 48};
  Qnn_Tensor_t outputs_input_b_nhwc[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "input_b_nhwc",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions_input_b_nhwc,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "input_b_nhwc", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params_input_b_nhwc, // Node Params
                         1, // Num Node Params
                         inputs_input_b_nhwc, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_input_b_nhwc, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor__v_78(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions__v_78[] = {3, 3, 3, 16};
  VALIDATE(model.addTensor("_v_78", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "_v_78",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions__v_78,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(_v_78),
                                                .dataSize=BINLEN(_v_78)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor__v_83(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions__v_83[] = {16};
  VALIDATE(model.addTensor("_v_83", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "_v_83",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions__v_83,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(_v_83),
                                                .dataSize=BINLEN(_v_83)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode_sng_Conv_0(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR sng_Conv_0 */
  uint32_t dimensions_sng_Conv_0_dilation[] = {2};
  uint32_t sng_Conv_0_dilation[] = {1, 1};
  uint32_t dimensions_sng_Conv_0_pad_amount[] = {2, 2};
  uint32_t sng_Conv_0_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions_sng_Conv_0_stride[] = {2};
  uint32_t sng_Conv_0_stride[] = {1, 1};
  Qnn_Param_t params_sng_Conv_0[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "sng_Conv_0_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_sng_Conv_0_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)sng_Conv_0_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "sng_Conv_0_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_sng_Conv_0_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)sng_Conv_0_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "sng_Conv_0_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_sng_Conv_0_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)sng_Conv_0_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="reuse_sparse_indices",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}
  };
  const char*  inputs_sng_Conv_0[] = {
    "input_a_nhwc",
    "_v_78",
    "_v_83"
  };
  uint32_t dimensions_conv1[] = {1, 224, 224, 16};
  Qnn_Tensor_t outputs_sng_Conv_0[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "conv1",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions_conv1,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "sng_Conv_0", // Node Name
                         "qti.aisw", // Package Name
                         "Conv2d", // Qnn Node Type
                         params_sng_Conv_0, // Node Params
                         5, // Num Node Params
                         inputs_sng_Conv_0, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs_sng_Conv_0, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_sng_Relu_2(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR sng_Relu_2 */
  Qnn_Param_t params_sng_Relu_2[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 4}}}}
  };
  const char*  inputs_sng_Relu_2[] = {
    "conv1"
  };
  uint32_t dimensions_r1[] = {1, 224, 224, 16};
  Qnn_Tensor_t outputs_sng_Relu_2[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "r1",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions_r1,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "sng_Relu_2", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params_sng_Relu_2, // Node Params
                         1, // Num Node Params
                         inputs_sng_Relu_2, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_sng_Relu_2, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor__v_104(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions__v_104[] = {3, 3, 16, 24};
  VALIDATE(model.addTensor("_v_104", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "_v_104",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions__v_104,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(_v_104),
                                                .dataSize=BINLEN(_v_104)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor__v_109(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions__v_109[] = {24};
  VALIDATE(model.addTensor("_v_109", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "_v_109",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions__v_109,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(_v_109),
                                                .dataSize=BINLEN(_v_109)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode_sng_Conv_3(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR sng_Conv_3 */
  uint32_t dimensions_sng_Conv_3_dilation[] = {2};
  uint32_t sng_Conv_3_dilation[] = {1, 1};
  uint32_t dimensions_sng_Conv_3_pad_amount[] = {2, 2};
  uint32_t sng_Conv_3_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions_sng_Conv_3_stride[] = {2};
  uint32_t sng_Conv_3_stride[] = {1, 1};
  Qnn_Param_t params_sng_Conv_3[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "sng_Conv_3_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_sng_Conv_3_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)sng_Conv_3_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "sng_Conv_3_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_sng_Conv_3_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)sng_Conv_3_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "sng_Conv_3_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_sng_Conv_3_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)sng_Conv_3_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="reuse_sparse_indices",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}
  };
  const char*  inputs_sng_Conv_3[] = {
    "r1",
    "_v_104",
    "_v_109"
  };
  uint32_t dimensions_conv2a[] = {1, 224, 224, 24};
  Qnn_Tensor_t outputs_sng_Conv_3[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "conv2a",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions_conv2a,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "sng_Conv_3", // Node Name
                         "qti.aisw", // Package Name
                         "Conv2d", // Qnn Node Type
                         params_sng_Conv_3, // Node Params
                         5, // Num Node Params
                         inputs_sng_Conv_3, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs_sng_Conv_3, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor__v_130(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions__v_130[] = {5, 5, 16, 24};
  VALIDATE(model.addTensor("_v_130", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "_v_130",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions__v_130,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(_v_130),
                                                .dataSize=BINLEN(_v_130)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor__v_135(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions__v_135[] = {24};
  VALIDATE(model.addTensor("_v_135", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "_v_135",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions__v_135,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(_v_135),
                                                .dataSize=BINLEN(_v_135)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode_sng_Conv_4(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR sng_Conv_4 */
  uint32_t dimensions_sng_Conv_4_dilation[] = {2};
  uint32_t sng_Conv_4_dilation[] = {1, 1};
  uint32_t dimensions_sng_Conv_4_pad_amount[] = {2, 2};
  uint32_t sng_Conv_4_pad_amount[] = {2, 2, 2, 2};
  uint32_t dimensions_sng_Conv_4_stride[] = {2};
  uint32_t sng_Conv_4_stride[] = {1, 1};
  Qnn_Param_t params_sng_Conv_4[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "sng_Conv_4_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_sng_Conv_4_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)sng_Conv_4_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "sng_Conv_4_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_sng_Conv_4_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)sng_Conv_4_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "sng_Conv_4_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_sng_Conv_4_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)sng_Conv_4_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="reuse_sparse_indices",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}
  };
  const char*  inputs_sng_Conv_4[] = {
    "r1",
    "_v_130",
    "_v_135"
  };
  uint32_t dimensions_conv2b[] = {1, 224, 224, 24};
  Qnn_Tensor_t outputs_sng_Conv_4[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "conv2b",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions_conv2b,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "sng_Conv_4", // Node Name
                         "qti.aisw", // Package Name
                         "Conv2d", // Qnn Node Type
                         params_sng_Conv_4, // Node Params
                         5, // Num Node Params
                         inputs_sng_Conv_4, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs_sng_Conv_4, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_sng_Relu_7(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR sng_Relu_7 */
  Qnn_Param_t params_sng_Relu_7[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 4}}}}
  };
  const char*  inputs_sng_Relu_7[] = {
    "conv2a"
  };
  uint32_t dimensions_r2a[] = {1, 224, 224, 24};
  Qnn_Tensor_t outputs_sng_Relu_7[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "r2a",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions_r2a,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "sng_Relu_7", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params_sng_Relu_7, // Node Params
                         1, // Num Node Params
                         inputs_sng_Relu_7, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_sng_Relu_7, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_sng_Relu_8(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR sng_Relu_8 */
  Qnn_Param_t params_sng_Relu_8[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 4}}}}
  };
  const char*  inputs_sng_Relu_8[] = {
    "conv2b"
  };
  uint32_t dimensions_r2b[] = {1, 224, 224, 24};
  Qnn_Tensor_t outputs_sng_Relu_8[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "r2b",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions_r2b,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "sng_Relu_8", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params_sng_Relu_8, // Node Params
                         1, // Num Node Params
                         inputs_sng_Relu_8, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_sng_Relu_8, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_sng_Concat_9(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR sng_Concat_9 */
  Qnn_Param_t params_sng_Concat_9[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="axis",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 3}}}}
  };
  const char*  inputs_sng_Concat_9[] = {
    "r2a",
    "r2b"
  };
  uint32_t dimensions_cat1[] = {1, 224, 224, 48};
  Qnn_Tensor_t outputs_sng_Concat_9[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "cat1",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions_cat1,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "sng_Concat_9", // Node Name
                         "qti.aisw", // Package Name
                         "Concat", // Qnn Node Type
                         params_sng_Concat_9, // Node Params
                         1, // Num Node Params
                         inputs_sng_Concat_9, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs_sng_Concat_9, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_sng_Add_10(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR sng_Add_10 */
  Qnn_Param_t params_sng_Add_10[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 0}}}}
  };
  const char*  inputs_sng_Add_10[] = {
    "cat1",
    "input_b_nhwc"
  };
  uint32_t dimensions_cat_bias[] = {1, 224, 224, 48};
  Qnn_Tensor_t outputs_sng_Add_10[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "cat_bias",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions_cat_bias,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "sng_Add_10", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseBinary", // Qnn Node Type
                         params_sng_Add_10, // Node Params
                         1, // Num Node Params
                         inputs_sng_Add_10, // Input Tensor Names
                         2, // Num Input Tensor Names
                         outputs_sng_Add_10, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor__v_156(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions__v_156[] = {3, 3, 48, 32};
  VALIDATE(model.addTensor("_v_156", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "_v_156",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions__v_156,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(_v_156),
                                                .dataSize=BINLEN(_v_156)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor__v_161(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions__v_161[] = {32};
  VALIDATE(model.addTensor("_v_161", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "_v_161",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions__v_161,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(_v_161),
                                                .dataSize=BINLEN(_v_161)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode_sng_Conv_11(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR sng_Conv_11 */
  uint32_t dimensions_sng_Conv_11_dilation[] = {2};
  uint32_t sng_Conv_11_dilation[] = {1, 1};
  uint32_t dimensions_sng_Conv_11_pad_amount[] = {2, 2};
  uint32_t sng_Conv_11_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions_sng_Conv_11_stride[] = {2};
  uint32_t sng_Conv_11_stride[] = {1, 1};
  Qnn_Param_t params_sng_Conv_11[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "sng_Conv_11_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_sng_Conv_11_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)sng_Conv_11_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "sng_Conv_11_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_sng_Conv_11_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)sng_Conv_11_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "sng_Conv_11_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_sng_Conv_11_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)sng_Conv_11_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="reuse_sparse_indices",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}
  };
  const char*  inputs_sng_Conv_11[] = {
    "cat_bias",
    "_v_156",
    "_v_161"
  };
  uint32_t dimensions_conv3[] = {1, 224, 224, 32};
  Qnn_Tensor_t outputs_sng_Conv_11[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "conv3",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions_conv3,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "sng_Conv_11", // Node Name
                         "qti.aisw", // Package Name
                         "Conv2d", // Qnn Node Type
                         params_sng_Conv_11, // Node Params
                         5, // Num Node Params
                         inputs_sng_Conv_11, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs_sng_Conv_11, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_sng_Relu_13(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR sng_Relu_13 */
  Qnn_Param_t params_sng_Relu_13[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 4}}}}
  };
  const char*  inputs_sng_Relu_13[] = {
    "conv3"
  };
  uint32_t dimensions_r3[] = {1, 224, 224, 32};
  Qnn_Tensor_t outputs_sng_Relu_13[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "r3",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions_r3,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "sng_Relu_13", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params_sng_Relu_13, // Node Params
                         1, // Num Node Params
                         inputs_sng_Relu_13, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_sng_Relu_13, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_sng_MaxPool_14(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR sng_MaxPool_14 */
  uint32_t dimensions_sng_MaxPool_14_filter_size[] = {2};
  uint32_t sng_MaxPool_14_filter_size[] = {2, 2};
  uint32_t dimensions_sng_MaxPool_14_pad_amount[] = {2, 2};
  uint32_t sng_MaxPool_14_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions_sng_MaxPool_14_stride[] = {2};
  uint32_t sng_MaxPool_14_stride[] = {2, 2};
  Qnn_Param_t params_sng_MaxPool_14[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="filter_size",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "sng_MaxPool_14_filter_size",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_sng_MaxPool_14_filter_size,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)sng_MaxPool_14_filter_size,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "sng_MaxPool_14_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_sng_MaxPool_14_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)sng_MaxPool_14_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "sng_MaxPool_14_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_sng_MaxPool_14_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)sng_MaxPool_14_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs_sng_MaxPool_14[] = {
    "r3"
  };
  uint32_t dimensions_pool1[] = {1, 112, 112, 32};
  Qnn_Tensor_t outputs_sng_MaxPool_14[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "pool1",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions_pool1,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "sng_MaxPool_14", // Node Name
                         "qti.aisw", // Package Name
                         "PoolMax2d", // Qnn Node Type
                         params_sng_MaxPool_14, // Node Params
                         3, // Num Node Params
                         inputs_sng_MaxPool_14, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_sng_MaxPool_14, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor__v_182(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions__v_182[] = {3, 3, 32, 64};
  VALIDATE(model.addTensor("_v_182", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "_v_182",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 4,
                                 .dimensions=dimensions__v_182,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(_v_182),
                                                .dataSize=BINLEN(_v_182)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor__v_187(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions__v_187[] = {64};
  VALIDATE(model.addTensor("_v_187", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "_v_187",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions__v_187,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(_v_187),
                                                .dataSize=BINLEN(_v_187)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode_sng_Conv_15(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR sng_Conv_15 */
  uint32_t dimensions_sng_Conv_15_dilation[] = {2};
  uint32_t sng_Conv_15_dilation[] = {1, 1};
  uint32_t dimensions_sng_Conv_15_pad_amount[] = {2, 2};
  uint32_t sng_Conv_15_pad_amount[] = {1, 1, 1, 1};
  uint32_t dimensions_sng_Conv_15_stride[] = {2};
  uint32_t sng_Conv_15_stride[] = {1, 1};
  Qnn_Param_t params_sng_Conv_15[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="dilation",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "sng_Conv_15_dilation",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_sng_Conv_15_dilation,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)sng_Conv_15_dilation,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "sng_Conv_15_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_sng_Conv_15_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)sng_Conv_15_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "sng_Conv_15_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_sng_Conv_15_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)sng_Conv_15_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="group",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="reuse_sparse_indices",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}
  };
  const char*  inputs_sng_Conv_15[] = {
    "pool1",
    "_v_182",
    "_v_187"
  };
  uint32_t dimensions_conv4[] = {1, 112, 112, 64};
  Qnn_Tensor_t outputs_sng_Conv_15[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "conv4",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions_conv4,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "sng_Conv_15", // Node Name
                         "qti.aisw", // Package Name
                         "Conv2d", // Qnn Node Type
                         params_sng_Conv_15, // Node Params
                         5, // Num Node Params
                         inputs_sng_Conv_15, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs_sng_Conv_15, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_sng_Relu_17(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR sng_Relu_17 */
  Qnn_Param_t params_sng_Relu_17[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="operation",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 4}}}}
  };
  const char*  inputs_sng_Relu_17[] = {
    "conv4"
  };
  uint32_t dimensions_r4[] = {1, 112, 112, 64};
  Qnn_Tensor_t outputs_sng_Relu_17[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "r4",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions_r4,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "sng_Relu_17", // Node Name
                         "qti.aisw", // Package Name
                         "ElementWiseNeuron", // Qnn Node Type
                         params_sng_Relu_17, // Node Params
                         1, // Num Node Params
                         inputs_sng_Relu_17, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_sng_Relu_17, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_sng_GlobalAveragePool_18(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR sng_GlobalAveragePool_18 */
  uint32_t dimensions_sng_GlobalAveragePool_18_filter_size[] = {2};
  uint32_t sng_GlobalAveragePool_18_filter_size[] = {112, 112};
  uint32_t dimensions_sng_GlobalAveragePool_18_pad_amount[] = {2, 2};
  uint32_t sng_GlobalAveragePool_18_pad_amount[] = {0, 0, 0, 0};
  uint32_t dimensions_sng_GlobalAveragePool_18_stride[] = {2};
  uint32_t sng_GlobalAveragePool_18_stride[] = {112, 112};
  Qnn_Param_t params_sng_GlobalAveragePool_18[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="filter_size",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "sng_GlobalAveragePool_18_filter_size",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_sng_GlobalAveragePool_18_filter_size,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)sng_GlobalAveragePool_18_filter_size,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="pad_amount",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "sng_GlobalAveragePool_18_pad_amount",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_sng_GlobalAveragePool_18_pad_amount,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)sng_GlobalAveragePool_18_pad_amount,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="stride",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "sng_GlobalAveragePool_18_stride",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_sng_GlobalAveragePool_18_stride,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)sng_GlobalAveragePool_18_stride,
                           .dataSize=8}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="count_pad_for_edges",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_BOOL_8, {.bool8Value = 0}}}}
  };
  const char*  inputs_sng_GlobalAveragePool_18[] = {
    "r4"
  };
  uint32_t dimensions_gap[] = {1, 1, 1, 64};
  Qnn_Tensor_t outputs_sng_GlobalAveragePool_18[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "gap",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions_gap,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "sng_GlobalAveragePool_18", // Node Name
                         "qti.aisw", // Package Name
                         "PoolAvg2d", // Qnn Node Type
                         params_sng_GlobalAveragePool_18, // Node Params
                         4, // Num Node Params
                         inputs_sng_GlobalAveragePool_18, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_sng_GlobalAveragePool_18, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_gap_nchw(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR gap_nchw */
  uint32_t dimensions_gap_nchw_perm[] = {4};
  uint32_t gap_nchw_perm[] = {0, 3, 1, 2};
  Qnn_Param_t params_gap_nchw[] = {
    {.paramType=QNN_PARAMTYPE_TENSOR,
     .name="perm",
     {.tensorParam=(Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "gap_nchw_perm",
            .type= QNN_TENSOR_TYPE_STATIC,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_UINT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 1,
            .dimensions=dimensions_gap_nchw_perm,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=(uint8_t*)gap_nchw_perm,
                           .dataSize=16}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}}}
  };
  const char*  inputs_gap_nchw[] = {
    "gap"
  };
  uint32_t dimensions_gap_nchw[] = {1, 64, 1, 1};
  Qnn_Tensor_t outputs_gap_nchw[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "gap_nchw",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 4,
            .dimensions=dimensions_gap_nchw,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "gap_nchw", // Node Name
                         "qti.aisw", // Package Name
                         "Transpose", // Qnn Node Type
                         params_gap_nchw, // Node Params
                         1, // Num Node Params
                         inputs_gap_nchw, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_gap_nchw, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addTensor_fc_w(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_fc_w[] = {10, 64};
  VALIDATE(model.addTensor("fc_w", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "fc_w",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 2,
                                 .dimensions=dimensions_fc_w,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(fc_w),
                                                .dataSize=BINLEN(fc_w)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addTensor_fc_b(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;
  uint32_t dimensions_fc_b[] = {10};
  VALIDATE(model.addTensor("fc_b", // Tensor Name
                           (Qnn_Tensor_t) {
                               .version= QNN_TENSOR_VERSION_2,
                               {.v2= {
                                 .id=0,
                                 .name= "fc_b",
                                 .type= QNN_TENSOR_TYPE_STATIC,
                                 .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
                                 .dataType= QNN_DATATYPE_FLOAT_32,
                                 .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                    {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
                                 .rank= 1,
                                 .dimensions=dimensions_fc_b,
                                 .memType= QNN_TENSORMEMTYPE_RAW,
                                 {.clientBuf= { .data=BINVARSTART(fc_b),
                                                .dataSize=BINLEN(fc_b)}},
                                 .isDynamicDimensions= nullptr,
                                 .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                                                  .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
                                 .isProduced= 0}}}
  ), err);
  return err;
}

static ModelError_t addNode_Gemm_0(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR Gemm_0 */
  const char*  inputs_Gemm_0[] = {
    "gap_nchw",
    "fc_w",
    "fc_b"
  };
  uint32_t dimensions_logits[] = {1, 10};
  Qnn_Tensor_t outputs_Gemm_0[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "logits",
            .type= QNN_TENSOR_TYPE_NATIVE,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_logits,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "Gemm_0", // Node Name
                         "qti.aisw", // Package Name
                         "FullyConnected", // Qnn Node Type
                         nullptr, // Node Params
                         0, // Num Node Params
                         inputs_Gemm_0, // Input Tensor Names
                         3, // Num Input Tensor Names
                         outputs_Gemm_0, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

static ModelError_t addNode_sng_Softmax_22(QnnModel& model){
  ModelError_t err = MODEL_NO_ERROR;

  /* ADDING NODE FOR sng_Softmax_22 */
  Qnn_Param_t params_sng_Softmax_22[] = {
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="axis",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_UINT_32, {.uint32Value = 1}}}},
    {.paramType=QNN_PARAMTYPE_SCALAR,
     .name="beta",
     {.scalarParam= (Qnn_Scalar_t) {QNN_DATATYPE_FLOAT_32, {.floatValue = 1.0000000000000000000000000000000000000000f}}}}
  };
  const char*  inputs_sng_Softmax_22[] = {
    "logits"
  };
  uint32_t dimensions_output[] = {1, 10};
  Qnn_Tensor_t outputs_sng_Softmax_22[] = {
    (Qnn_Tensor_t) {
          .version= QNN_TENSOR_VERSION_2,
          {.v2= {
            .id=0,
            .name= "output",
            .type= QNN_TENSOR_TYPE_APP_READ,
            .dataFormat= QNN_TENSOR_DATA_FORMAT_DENSE,
            .dataType= QNN_DATATYPE_FLOAT_32,
            .quantizeParams= { QNN_DEFINITION_UNDEFINED,
                               QNN_QUANTIZATION_ENCODING_UNDEFINED,
                               {.scaleOffsetEncoding= {.scale= 0.0000000000000000000000000000000000000000f, .offset= 0}}},
            .rank= 2,
            .dimensions=dimensions_output,
            .memType= QNN_TENSORMEMTYPE_RAW,
            {.clientBuf= { .data=nullptr,
                           .dataSize=0}},
            .isDynamicDimensions= nullptr,
            .sparseParams= { QNN_SPARSE_LAYOUT_UNDEFINED,
                             .hybridCoo= {.numSpecifiedElements= 0, .numSparseDimensions= 0}},
            .isProduced= 0}}}
  };
  VALIDATE(model.addNode(QNN_OPCONFIG_VERSION_1, // Op_Config_t Version
                         "sng_Softmax_22", // Node Name
                         "qti.aisw", // Package Name
                         "Softmax", // Qnn Node Type
                         params_sng_Softmax_22, // Node Params
                         2, // Num Node Params
                         inputs_sng_Softmax_22, // Input Tensor Names
                         1, // Num Input Tensor Names
                         outputs_sng_Softmax_22, // Output Tensors 
                         1// Num Output Tensors 
  ), err);
  return err;
}

QNN_API
ModelError_t QnnModel_composeGraphs(Qnn_BackendHandle_t backendHandle,
                                    QNN_INTERFACE_VER_TYPE interface,
                                    Qnn_ContextHandle_t contextHandle,
                                    const GraphConfigInfo_t** graphsConfigInfo,
                                    const uint32_t numGraphsConfigInfo,
                                    GraphInfoPtr_t** graphsInfo,
                                    uint32_t* numGraphsInfo,
                                    bool debug,
                                    QnnLog_Callback_t logCallback,
                                    QnnLog_Level_t maxLogLevel) {

  ModelError_t err = MODEL_NO_ERROR;

  /* model/graph for test_model*/
  QnnModel test_model;
  const QnnGraph_Config_t** graphConfigs = nullptr;
  VALIDATE(getQnnGraphConfigFromInfo("test_model", graphsConfigInfo, numGraphsConfigInfo, graphConfigs), err);
  VALIDATE(test_model.initialize(backendHandle, interface, contextHandle, "test_model", debug, DO_GRAPH_NODE_VALIDATIONS, graphConfigs), err);
  VALIDATE(addTensor_input_a(test_model), err);
  VALIDATE(addTensor_input_b(test_model), err);
  VALIDATE(addNode_input_a_nhwc(test_model), err);
  VALIDATE(addNode_input_b_nhwc(test_model), err);
  VALIDATE(addTensor__v_78(test_model), err);
  VALIDATE(addTensor__v_83(test_model), err);
  VALIDATE(addNode_sng_Conv_0(test_model), err);
  VALIDATE(addNode_sng_Relu_2(test_model), err);
  VALIDATE(addTensor__v_104(test_model), err);
  VALIDATE(addTensor__v_109(test_model), err);
  VALIDATE(addNode_sng_Conv_3(test_model), err);
  VALIDATE(addTensor__v_130(test_model), err);
  VALIDATE(addTensor__v_135(test_model), err);
  VALIDATE(addNode_sng_Conv_4(test_model), err);
  VALIDATE(addNode_sng_Relu_7(test_model), err);
  VALIDATE(addNode_sng_Relu_8(test_model), err);
  VALIDATE(addNode_sng_Concat_9(test_model), err);
  VALIDATE(addNode_sng_Add_10(test_model), err);
  VALIDATE(addTensor__v_156(test_model), err);
  VALIDATE(addTensor__v_161(test_model), err);
  VALIDATE(addNode_sng_Conv_11(test_model), err);
  VALIDATE(addNode_sng_Relu_13(test_model), err);
  VALIDATE(addNode_sng_MaxPool_14(test_model), err);
  VALIDATE(addTensor__v_182(test_model), err);
  VALIDATE(addTensor__v_187(test_model), err);
  VALIDATE(addNode_sng_Conv_15(test_model), err);
  VALIDATE(addNode_sng_Relu_17(test_model), err);
  VALIDATE(addNode_sng_GlobalAveragePool_18(test_model), err);
  VALIDATE(addNode_gap_nchw(test_model), err);
  VALIDATE(addTensor_fc_w(test_model), err);
  VALIDATE(addTensor_fc_b(test_model), err);
  VALIDATE(addNode_Gemm_0(test_model), err);
  VALIDATE(addNode_sng_Softmax_22(test_model), err);

  // Add all models to array to get graphsInfo
  QnnModel* models [] = {&test_model};
  uint32_t numModels = 1;

  // Populate the constructed graphs in provided output variables
  VALIDATE(getGraphInfoFromModels(*models, numModels, graphsInfo), err);
  *numGraphsInfo = numModels;

  return err;

} // PREPARE_GRAPHS

QNN_API
ModelError_t QnnModel_freeGraphsInfo(GraphInfoPtr_t** graphsInfo, uint32_t numGraphsInfo){
  return qnn_wrapper_api::freeGraphsInfo(graphsInfo, numGraphsInfo);
} // FREEGRAPHINFO

}