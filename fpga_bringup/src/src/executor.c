#include <animegan/runtime.h>

/* Node dispatch. Scalar kernels are the reference semantics for every build;
 * the AME backend runs CONV_2D on the matrix unit and (AG_RVV builds) the
 * other operators with RVV; a kernel that returns AG_AME_UNSUPPORTED for a
 * shape falls back to the scalar kernel, except in builds with
 * AG_NO_SCALAR_FALLBACK (the board: tensor data is RVV/AMU-only), where the
 * node fails with AG_NO_KERNEL. */
#ifdef AG_NO_SCALAR_FALLBACK
#define AG_NO_KERNEL (-30)
#define ame_fallback(model, node_id, tensor_data) AG_NO_KERNEL
#else
#define ame_fallback(model, node_id, tensor_data) ag_execute_node_scalar(model, node_id, tensor_data)
#endif
int ag_execute_node_scalar(const ag_model *model, uint32_t node_id, void *const tensor_data[]) {
    const ag_node_desc *node;
    const int32_t *inputs;
    const int32_t *outputs;
    if (!model || !tensor_data || node_id >= model->node_count) return -1;
    node = model->nodes + node_id;
    if (node->input_count < 1 || node->output_count != 1) return -2;
    inputs = ag_model_at(model, node->input_offset, node->input_count * sizeof(int32_t));
    outputs = ag_model_at(model, node->output_offset, sizeof(int32_t));
    if (!inputs || !outputs) return -3;
    switch (node->opcode) {
    case 100: return ag_mirror_pad(model, node, inputs, outputs, tensor_data);
    case 3: return ag_conv2d_scalar(model, node, inputs, outputs, tensor_data);
    case 0: return ag_add(model, node, inputs, outputs, tensor_data);
    case 23: return ag_resize_bilinear(model, node, inputs, outputs, tensor_data);
    case 98: return ag_leaky_relu(model, node, inputs, outputs, tensor_data);
    case 28: return ag_tanh(model, node, inputs, outputs, tensor_data);
    case 200: return ag_lade(model, node, inputs, outputs, tensor_data);
    default: return -7;
    }
}

int ag_runtime_init(ag_runtime *runtime, const ag_model *model, void **tensor_data,
                    size_t *tensor_size, uint32_t capacity, ag_backend backend) {
    uint32_t index;
    if (!runtime || !model || !tensor_data || !tensor_size ||
        capacity < model->tensor_count) return -1;
    runtime->model = model;
    runtime->tensor_data = tensor_data;
    runtime->tensor_size = tensor_size;
    runtime->capacity = capacity;
    runtime->backend = backend;
    runtime->scratch = 0;
    runtime->scratch_size = 0;
    for (index = 0; index < model->tensor_count; ++index) {
        const ag_tensor_desc *tensor = model->tensors + index;
        tensor_data[index] = 0;
        tensor_size[index] = 0;
        if (tensor->data_size) {
            tensor_data[index] = (void *)ag_model_at(model, tensor->data_offset, tensor->data_size);
            tensor_size[index] = (size_t)tensor->data_size;
        }
    }
    return 0;
}

int ag_runtime_bind(ag_runtime *runtime, uint32_t tensor_id, void *data, size_t size) {
    if (!runtime || tensor_id >= runtime->model->tensor_count || !data) return -1;
    if (size != ag_tensor_bytes(runtime->model->tensors + tensor_id)) return -2;
    runtime->tensor_data[tensor_id] = data;
    runtime->tensor_size[tensor_id] = size;
    return 0;
}

/* Workspace a node needs under the given backend; scalar kernels need none. */
size_t ag_node_scratch_bytes(const ag_model *model, uint32_t node_id, ag_backend backend) {
#ifdef AG_AME
    if (backend == AG_BACKEND_AME && node_id < model->node_count && model->nodes[node_id].opcode == 3)
        return ag_conv2d_ame_scratch(model, model->nodes + node_id);
#else
    (void)model; (void)node_id; (void)backend;
#endif
    return 0;
}

int ag_runtime_execute_node(ag_runtime *runtime, uint32_t node_id) {
    if (!runtime) return -1;
#ifdef AG_AME
    if (runtime->backend == AG_BACKEND_AME) {
        const ag_model *model = runtime->model;
        const ag_node_desc *node;
        const int32_t *inputs, *outputs;
        int rc;
        if (node_id >= model->node_count) return -1;
        node = model->nodes + node_id;
        inputs = ag_model_at(model, node->input_offset, node->input_count * sizeof(int32_t));
        outputs = ag_model_at(model, node->output_offset, sizeof(int32_t));
        if (!inputs || !outputs) return -3;
        if (node->opcode == 3) {
            rc = ag_conv2d_ame(model, node, inputs, outputs, runtime->tensor_data,
                               runtime->scratch, runtime->scratch_size);
            if (rc != AG_AME_UNSUPPORTED) return rc;
            return ame_fallback(model, node_id, runtime->tensor_data);
        }
#ifdef AG_RVV
        rc = ag_execute_node_rvv(model, node, inputs, outputs, runtime->tensor_data);
        if (rc != AG_AME_UNSUPPORTED) return rc;
#endif
        return ame_fallback(model, node_id, runtime->tensor_data);
    }
#endif
    if (runtime->backend != AG_BACKEND_SCALAR) return -2;
    return ag_execute_node_scalar(runtime->model, node_id, runtime->tensor_data);
}
