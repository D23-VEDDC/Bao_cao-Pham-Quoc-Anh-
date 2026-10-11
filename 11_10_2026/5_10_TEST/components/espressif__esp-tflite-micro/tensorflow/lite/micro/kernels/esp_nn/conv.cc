/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/micro/kernels/conv.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/kernels/internal/reference/conv.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

#include <esp_timer.h>

#ifndef VEDC_FORCE_TFLM_REFERENCE_CONV
#define VEDC_FORCE_TFLM_REFERENCE_CONV 0
#endif

#if ESP_NN && !VEDC_FORCE_TFLM_REFERENCE_CONV
#include <esp_nn.h>
#endif

long long conv_total_time = 0;

namespace tflite {
namespace {

struct NodeData {
  OpDataConv op_data;
#if ESP_NN && !VEDC_FORCE_TFLM_REFERENCE_CONV
  int buffer_idx;
#endif
};

static void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(NodeData));
}

static TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  NodeData* data = static_cast<NodeData*>(node->user_data);
  const auto& params =
      *(static_cast<const TfLiteConvParams*>(node->builtin_data));

  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kConvOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kConvInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);

  TfLiteTensor* filter =
      micro_context->AllocateTempInputTensor(node, kConvWeightsTensor);
  TF_LITE_ENSURE(context, filter != nullptr);

  /* Check input channels matching filter */
  const int input_channels = input->dims->data[3];
  const int filter_input_channels = filter->dims->data[3];

  TF_LITE_ENSURE(context, filter_input_channels > 0);
  TF_LITE_ENSURE_EQ(
      context,
      input_channels % filter_input_channels,
      0);

  TF_LITE_ENSURE_EQ(context, input->type, output->type);

  TF_LITE_ENSURE_MSG(
      context,
      (input->type == kTfLiteFloat32 && filter->type == kTfLiteFloat32) ||
          (input->type == kTfLiteInt16 && filter->type == kTfLiteInt8) ||
          (input->type == kTfLiteInt8 &&
           (filter->type == kTfLiteInt4 || filter->type == kTfLiteInt8)),
      "Hybrid models are not supported on TFLite Micro.");

  const int input_width = input->dims->data[2];
  const int input_height = input->dims->data[1];

  const int filter_width = filter->dims->data[2];
  const int filter_height = filter->dims->data[1];

  const int output_width = output->dims->data[2];
  const int output_height = output->dims->data[1];

  /* Dynamically allocate per-channel quantization parameters. */
  const int num_channels =
      filter->dims->data[kConvQuantizedDimension];

  data->op_data.per_channel_output_multiplier =
      static_cast<int32_t*>(
          context->AllocatePersistentBuffer(
              context,
              num_channels * sizeof(int32_t)));

  data->op_data.per_channel_output_shift =
      static_cast<int32_t*>(
          context->AllocatePersistentBuffer(
              context,
              num_channels * sizeof(int32_t)));

  /* All per-channel quantized tensors need valid zero point and scale arrays. */
  if (input->type == kTfLiteInt8 ||
      input->type == kTfLiteInt16) {

    TF_LITE_ENSURE_EQ(
        context,
        filter->quantization.type,
        kTfLiteAffineQuantization);

    const auto* affine_quantization =
        static_cast<TfLiteAffineQuantization*>(
            filter->quantization.params);

    TFLITE_DCHECK(affine_quantization != nullptr);
    TFLITE_DCHECK(affine_quantization->scale != nullptr);
    TFLITE_DCHECK(affine_quantization->zero_point != nullptr);

    TF_LITE_ENSURE(
        context,
        affine_quantization->scale->size == 1 ||
            affine_quantization->scale->size ==
                filter->dims->data[kConvQuantizedDimension]);
  }

  TF_LITE_ENSURE_STATUS(
      CalculateOpDataConv(
          context,
          node,
          params,
          input_width,
          input_height,
          filter_width,
          filter_height,
          output_width,
          output_height,
          input->type,
          &data->op_data));

  if (filter->type == kTfLiteInt4) {
    int filter_size =
        RuntimeShape(
            filter->dims->size,
            reinterpret_cast<const int32_t*>(
                filter->dims->data))
            .FlatSize();

    context->RequestScratchBufferInArena(
        context,
        filter_size,
        &data->op_data.filter_buffer_index);
  }

#if ESP_NN && !VEDC_FORCE_TFLM_REFERENCE_CONV

  if (input->type == kTfLiteInt8) {
    data_dims_t input_dims = {
        .width = input_width,
        .height = input_height,
        .channels = input_channels,
        .extra = 1
    };

    data_dims_t output_dims = {
        .width = output_width,
        .height = output_height,
        .channels = output->dims->data[3],
        .extra = 1
    };

    data_dims_t filter_dims = {
        .width = filter_width,
        .height = filter_height,
        .channels = filter->dims->data[3],
        .extra = 0
    };

    conv_params_t conv_params = {
        .in_offset = 0,
        .out_offset = 0,
        .stride = {
            params.stride_width,
            params.stride_height
        },
        .padding = {
            data->op_data.padding.width,
            data->op_data.padding.height
        },
        .dilation = {
            0,
            0
        },
        .activation = {
            -128,
            127
        }
    };

    int scratch_buf_size =
        esp_nn_get_conv_scratch_size(
            &input_dims,
            &filter_dims,
            &output_dims,
            &conv_params);

    if (scratch_buf_size > 0) {
      TF_LITE_ENSURE_STATUS(
          context->RequestScratchBufferInArena(
              context,
              scratch_buf_size,
              &data->buffer_idx));
    } else {
      data->buffer_idx = -1;
    }
  }

#endif

  micro_context->DeallocateTempTfLiteTensor(output);
  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(filter);

  return kTfLiteOk;
}

#if ESP_NN && !VEDC_FORCE_TFLM_REFERENCE_CONV

/* Fixed-point per-channel-quantization convolution Int8 function wrapper. */
inline void EvalQuantizedPerChannel(
    TfLiteContext* context,
    TfLiteNode* node,
    const TfLiteConvParams& params,
    const NodeData& data,
    const TfLiteEvalTensor* input,
    const TfLiteEvalTensor* filter,
    const TfLiteEvalTensor* bias,
    TfLiteEvalTensor* output) {

  const int dilation_width_factor =
      params.dilation_width_factor;

  const int dilation_height_factor =
      params.dilation_height_factor;

  /*
   * VEDC profiling:
   * Count how many times the reference kernel and ESP-NN
   * kernel are actually used.
   */
  static int reference_profile_count = 0;
  static int optimized_profile_count = 0;
  static bool dispatch_banner_logged = false;

  if (!dispatch_banner_logged) {
    MicroPrintf(
        "[VEDC CONV] SELECTIVE_DISPATCH_V2");
    dispatch_banner_logged = true;
  }

  if (dilation_width_factor == 1 &&
      dilation_height_factor == 1) {

    /* Get tensor shapes. */
    RuntimeShape filter_shape =
        tflite::micro::GetTensorShape(filter);

    RuntimeShape input_shape =
        tflite::micro::GetTensorShape(input);

    RuntimeShape output_shape =
        tflite::micro::GetTensorShape(output);

    RuntimeShape bias_shape =
        tflite::micro::GetTensorShape(bias);

    const int8_t* input_data =
        tflite::micro::GetTensorData<int8_t>(input);

    int8_t* output_data =
        tflite::micro::GetTensorData<int8_t>(output);

    const int32_t input_offset =
        -data.op_data.input_zero_point;

    const int32_t output_offset =
        data.op_data.output_zero_point;

    const int stride_width =
        params.stride_width;

    const int stride_height =
        params.stride_height;

    const int pad_width =
        data.op_data.padding.width;

    const int pad_height =
        data.op_data.padding.height;

    const int input_height =
        input_shape.Dims(1);

    const int input_width =
        input_shape.Dims(2);

    const int filter_height =
        filter_shape.Dims(1);

    const int filter_width =
        filter_shape.Dims(2);

    const int output_height =
        output_shape.Dims(1);

    const int output_width =
        output_shape.Dims(2);

    const int input_depth =
        input_shape.Dims(3);

    const int output_depth =
        MatchingDim(
            filter_shape,
            0,
            output_shape,
            3);

    const int batch_size =
        MatchingDim(
            input_shape,
            0,
            output_shape,
            0);

    const int32_t activation_min =
        data.op_data.output_activation_min;

    const int32_t activation_max =
        data.op_data.output_activation_max;

    TFLITE_DCHECK_LE(
        activation_min,
        activation_max);

    TFLITE_DCHECK_EQ(
        input_shape.DimensionsCount(),
        4);

    TFLITE_DCHECK_EQ(
        filter_shape.DimensionsCount(),
        4);

    TFLITE_DCHECK_EQ(
        output_shape.DimensionsCount(),
        4);

    /*
     * ============================================================
     * VEDC SELECTIVE REFERENCE FALLBACK
     * ============================================================
     *
     * Do NOT fallback all 1x1 convolutions whose channels are
     * not multiples of 16.
     *
     * The ESP-NN implementation used by this project supports
     * the 1x1 path whenever input channels are a multiple of 8.
     *
     * Only keep the exact shapes previously observed to generate
     * incorrect feature maps / duplicate vectors.
     */

    bool force_reference = false;

    if (filter_shape.Dims(1) == 1 &&
        filter_shape.Dims(2) == 1) {

      /*
       * Confirmed problematic:
       * 16 input channels -> 8 output channels.
       */
      if (input_depth == 16 &&
          output_depth == 8) {
        force_reference = true;
      }

      /*
       * Confirmed problematic:
       * 48 input channels -> 8 output channels.
       */
      if (input_depth == 48 &&
          output_depth == 8) {
        force_reference = true;
      }

      /*
       * Confirmed problematic:
       * 8 input channels -> 48 output channels
       * specifically at 20x20 feature map.
       */
      if (input_depth == 8 &&
          output_depth == 48 &&
          input_shape.Dims(1) == 20 &&
          input_shape.Dims(2) == 20) {
        force_reference = true;
      }
    }

    if (force_reference) {

      int64_t ref_t0 =
          esp_timer_get_time();

      /*
       * Log every different fallback shape for the first
       * few occurrences. This must replace the old:
       *
       * [ESP-NN FIX] 1x1 Conv channels not multiple of 16 ...
       */
      if (reference_profile_count < 30) {
        MicroPrintf(
            "[ESP-NN PROFILE] REFERENCE "
            "input=%dx%dx%d "
            "output=%dx%dx%d "
            "filter=%dx%d",
            input_width,
            input_height,
            input_depth,
            output_width,
            output_height,
            output_depth,
            filter_width,
            filter_height);
      }

      reference_integer_ops::ConvPerChannel(
          ConvParamsQuantized(
              params,
              data.op_data),
          data.op_data.per_channel_output_multiplier,
          data.op_data.per_channel_output_shift,
          input_shape,
          input_data,
          filter_shape,
          tflite::micro::GetTensorData<int8_t>(
              filter),
          bias_shape,
          tflite::micro::GetOptionalTensorData<int32_t>(
              bias),
          output_shape,
          output_data);

      int64_t ref_us =
          esp_timer_get_time() - ref_t0;

      if (reference_profile_count < 30) {
        MicroPrintf(
            "[ESP-NN PROFILE] REFERENCE DONE "
            "%lld us",
            (long long)ref_us);
      }

      reference_profile_count++;

      return;
    }

    /*
     * ============================================================
     * 2x2 POINTWISE FALLBACK
     * ============================================================
     *
     * Keep this fallback because the current VEDC model/source
     * already uses it as a known-safe path.
     *
     * We profile it so we can later determine whether it is
     * relevant to the total inference time.
     */
    /*
    if (input_shape.Dims(0) == 1 &&
        input_shape.Dims(1) == 2 &&
        input_shape.Dims(2) == 2 &&
        filter_shape.Dims(1) == 1 &&
        filter_shape.Dims(2) == 1 &&
        filter_shape.Dims(3) == input_depth &&
        output_shape.Dims(0) == 1 &&
        output_shape.Dims(1) == 2 &&
        output_shape.Dims(2) == 2 &&
        params.stride_width == 1 &&
        params.stride_height == 1) {

        int64_t ref_2x2_t0 = esp_timer_get_time();

        if (reference_profile_count < 30) {
          MicroPrintf(
              "[ESP-NN PROFILE] REFERENCE 2x2 "
              "input=%dx%dx%d "
              "output=%dx%dx%d",
              input_width,
              input_height,
              input_depth,
              output_width,
              output_height,
              output_depth);
        }

        reference_integer_ops::ConvPerChannel(
            ConvParamsQuantized(
                params,
                data.op_data),
            data.op_data.per_channel_output_multiplier,
            data.op_data.per_channel_output_shift,
            input_shape,
            input_data,
            filter_shape,
            tflite::micro::GetTensorData<int8_t>(filter),
            bias_shape,
            tflite::micro::GetOptionalTensorData<int32_t>(bias),
            output_shape,
            output_data);

        int64_t ref_2x2_us =
            esp_timer_get_time() - ref_2x2_t0;

        if (reference_profile_count < 30) {
          MicroPrintf(
              "[ESP-NN PROFILE] REFERENCE 2x2 DONE "
              "%lld us",
              (long long)ref_2x2_us);
        }

        reference_profile_count++;

        return;
    }
    */	

    /*
     * ============================================================
     * ESP-NN OPTIMIZED PATH
     * ============================================================
     */

    if (tflite::micro::GetTensorData<int8_t>(bias)) {
      TFLITE_DCHECK_EQ(
          bias_shape.FlatSize(),
          output_depth);
    }

    void* scratch_buf = NULL;

    if (data.buffer_idx > -1) {
      scratch_buf =
          context->GetScratchBuffer(
              context,
              data.buffer_idx);
    }

    esp_nn_set_conv_scratch_buf(
        scratch_buf);

    const int input_size =
        input_width *
        input_height *
        input_depth;

    const int output_size =
        output_width *
        output_height *
        output_depth;

    data_dims_t input_dims = {
        .width = input_width,
        .height = input_height,
        .channels = input_depth,
        .extra = 1
    };

    data_dims_t output_dims = {
        .width = output_width,
        .height = output_height,
        .channels = output_depth,
        .extra = 1
    };

    data_dims_t filter_dims = {
        .width = filter_width,
        .height = filter_height,
        .channels = filter->dims->data[3],
        .extra = 0
    };

    conv_params_t conv_params = {
        .in_offset = input_offset,
        .out_offset = output_offset,
        .stride = {
            stride_width,
            stride_height
        },
        .padding = {
            pad_width,
            pad_height
        },
        .dilation = {
            0,
            0
        },
        .activation = {
            activation_min,
            activation_max
        }
    };

    quant_data_t quant_data = {
        .shift =
            data.op_data.per_channel_output_shift,

        .mult =
            data.op_data.per_channel_output_multiplier
    };

    for (int i_batch = 0;
         i_batch < batch_size;
         i_batch++) {

      int64_t nn_t0 =
          esp_timer_get_time();

      esp_nn_conv_s8(
          &input_dims,
          input_data +
              i_batch * input_size,

          &filter_dims,
          tflite::micro::GetTensorData<int8_t>(
              filter),

          tflite::micro::GetTensorData<int32_t>(
              bias),

          &output_dims,
          output_data +
              i_batch * output_size,

          &conv_params,
          &quant_data);

      int64_t nn_us =
          esp_timer_get_time() - nn_t0;

      if (optimized_profile_count < 40) {
        MicroPrintf(
            "[ESP-NN PROFILE] OPT "
            "input=%dx%dx%d "
            "output=%dx%dx%d "
            "filter=%dx%d "
            "time=%lld us",
            input_width,
            input_height,
            input_depth,
            output_width,
            output_height,
            output_depth,
            filter_width,
            filter_height,
            (long long)nn_us);

        optimized_profile_count++;
      }
    }

  } else {

    /*
     * Dilation != 1
     *
     * ESP-NN path above is only used for standard convolution
     * without dilation.
     */
    reference_integer_ops::ConvPerChannel(
        ConvParamsQuantized(
            params,
            data.op_data),
        data.op_data.per_channel_output_multiplier,
        data.op_data.per_channel_output_shift,
        tflite::micro::GetTensorShape(input),
        tflite::micro::GetTensorData<int8_t>(input),
        tflite::micro::GetTensorShape(filter),
        tflite::micro::GetTensorData<int8_t>(filter),
        tflite::micro::GetTensorShape(bias),
        tflite::micro::GetTensorData<int32_t>(bias),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<int8_t>(output));
  }
}

#endif

static TfLiteStatus Eval(
    TfLiteContext* context,
    TfLiteNode* node) {

  /*
   * Build marker.
   *
   * This is intentionally inside Eval() so that it proves
   * the exact conv.cc compiled into the firmware is the new
   * selective-dispatch version.
   */
  static bool conv_build_logged = false;

  if (!conv_build_logged) {
    MicroPrintf(
        "[VEDC CONV] SELECTIVE_DISPATCH_V2");
    conv_build_logged = true;
  }

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(
          context,
          node,
          kConvInputTensor);

  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(
          context,
          node,
          kConvWeightsTensor);

  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 3)
          ? tflite::micro::GetEvalInput(
                context,
                node,
                kConvBiasTensor)
          : nullptr;

  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(
          context,
          node,
          kConvOutputTensor);

  TFLITE_DCHECK(
      node->builtin_data != nullptr);

  const auto& params =
      *(reinterpret_cast<TfLiteConvParams*>(
          node->builtin_data));

  TFLITE_DCHECK(
      node->user_data != nullptr);

  const auto& data =
      *(static_cast<const NodeData*>(
          node->user_data));

  long long start_time =
      esp_timer_get_time();

  switch (input->type) {

    case kTfLiteFloat32: {

      tflite::reference_ops::Conv(
          ConvParamsFloat(
              params,
              data.op_data),

          tflite::micro::GetTensorShape(
              input),

          tflite::micro::GetTensorData<float>(
              input),

          tflite::micro::GetTensorShape(
              filter),

          tflite::micro::GetTensorData<float>(
              filter),

          tflite::micro::GetTensorShape(
              bias),

          tflite::micro::GetOptionalTensorData<float>(
              bias),

          tflite::micro::GetTensorShape(
              output),

          tflite::micro::GetTensorData<float>(
              output),

          tflite::micro::GetTensorShape(
              nullptr),

          nullptr);

      break;
    }

    case kTfLiteInt16: {

      if (bias == nullptr ||
          bias->type == kTfLiteInt32) {

        reference_integer_ops::ConvPerChannel(
            ConvParamsQuantized(
                params,
                data.op_data),

            data.op_data.per_channel_output_multiplier,

            data.op_data.per_channel_output_shift,

            tflite::micro::GetTensorShape(
                input),

            tflite::micro::GetTensorData<int16_t>(
                input),

            tflite::micro::GetTensorShape(
                filter),

            tflite::micro::GetTensorData<int8_t>(
                filter),

            tflite::micro::GetTensorShape(
                bias),

            tflite::micro::GetOptionalTensorData<std::int32_t>(
                bias),

            tflite::micro::GetTensorShape(
                output),

            tflite::micro::GetTensorData<int16_t>(
                output));

      } else if (bias->type == kTfLiteInt64) {

        reference_integer_ops::ConvPerChannel(
            ConvParamsQuantized(
                params,
                data.op_data),

            data.op_data.per_channel_output_multiplier,

            data.op_data.per_channel_output_shift,

            tflite::micro::GetTensorShape(
                input),

            tflite::micro::GetTensorData<int16_t>(
                input),

            tflite::micro::GetTensorShape(
                filter),

            tflite::micro::GetTensorData<int8_t>(
                filter),

            tflite::micro::GetTensorShape(
                bias),

            tflite::micro::GetOptionalTensorData<std::int64_t>(
                bias),

            tflite::micro::GetTensorShape(
                output),

            tflite::micro::GetTensorData<int16_t>(
                output));

      } else {

        MicroPrintf(
            "Bias type %s (%d) not supported.",
            TfLiteTypeGetName(bias->type),
            bias->type);

        return kTfLiteError;
      }

      break;
    }

    case kTfLiteInt8: {

      switch (filter->type) {

        case kTfLiteInt4: {

          int8_t* unpacked_filter_data =
              static_cast<int8_t*>(
                  context->GetScratchBuffer(
                      context,
                      data.op_data.filter_buffer_index));

          tflite::tensor_utils::UnpackDenseInt4IntoInt8(
              tflite::micro::GetTensorData<int8_t>(
                  filter),

              tflite::micro::GetTensorShape(
                  filter)
                  .FlatSize(),

              unpacked_filter_data);

          reference_integer_ops::ConvPerChannel(
              ConvParamsQuantized(
                  params,
                  data.op_data),

              data.op_data.per_channel_output_multiplier,

              data.op_data.per_channel_output_shift,

              tflite::micro::GetTensorShape(
                  input),

              tflite::micro::GetTensorData<int8_t>(
                  input),

              tflite::micro::GetTensorShape(
                  filter),

              unpacked_filter_data,

              tflite::micro::GetTensorShape(
                  bias),

              tflite::micro::GetOptionalTensorData<int32_t>(
                  bias),

              tflite::micro::GetTensorShape(
                  output),

              tflite::micro::GetTensorData<int8_t>(
                  output));

          break;
        }

        case kTfLiteInt8: {

#if ESP_NN && !VEDC_FORCE_TFLM_REFERENCE_CONV

          EvalQuantizedPerChannel(
              context,
              node,
              params,
              data,
              input,
              filter,
              bias,
              output);

#else

          reference_integer_ops::ConvPerChannel(
              ConvParamsQuantized(
                  params,
                  data.op_data),

              data.op_data.per_channel_output_multiplier,

              data.op_data.per_channel_output_shift,

              tflite::micro::GetTensorShape(
                  input),

              tflite::micro::GetTensorData<int8_t>(
                  input),

              tflite::micro::GetTensorShape(
                  filter),

              tflite::micro::GetTensorData<int8_t>(
                  filter),

              tflite::micro::GetTensorShape(
                  bias),

              tflite::micro::GetTensorData<int32_t>(
                  bias),

              tflite::micro::GetTensorShape(
                  output),

              tflite::micro::GetTensorData<int8_t>(
                  output));

#endif

          break;
        }

        default:

          MicroPrintf(
              "Weight type %s (%d) not supported.",
              TfLiteTypeGetName(filter->type),
              filter->type);

          return kTfLiteError;
      }

      break;
    }

    default:

      MicroPrintf(
          "Type %s (%d) not supported.",
          TfLiteTypeGetName(input->type),
          input->type);

      return kTfLiteError;
  }

  long long time_this_instance =
      esp_timer_get_time() - start_time;

  conv_total_time +=
      time_this_instance;

  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_CONV_2D() {
  return tflite::micro::RegisterOp(
      Init,
      Prepare,
      Eval);
}

}  // namespace tflite