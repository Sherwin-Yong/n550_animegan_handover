#include <animegan/runtime.h>
#include <string.h>

static const uint8_t ag_magic[8] = {'A','G','V','3','P','K','G',0};
typedef char ag_header_size_check[(sizeof(ag_pack_header) == 56) ? 1 : -1];
typedef char ag_tensor_size_check[(sizeof(ag_tensor_desc) == 80) ? 1 : -1];
typedef char ag_node_size_check[(sizeof(ag_node_desc) == 80) ? 1 : -1];

int ag_model_check_range(const ag_model *model, uint64_t offset, uint64_t size) {
    uint64_t end;
    if (!model || offset > model->size) return 0;
    end = offset + size;
    if (end < offset || end > model->size) return 0;
    return 1;
}

const void *ag_model_at(const ag_model *model, uint64_t offset, uint64_t size) {
    if (!ag_model_check_range(model, offset, size)) return 0;
    return model->data + offset;
}

int ag_model_open(ag_model *model, const void *data, size_t size) {
    const ag_pack_header *header;
    uint64_t tensor_bytes;
    uint64_t node_bytes;
    uint32_t index;
    if (!model || !data || size < sizeof(ag_pack_header)) return -1;
    memset(model, 0, sizeof(*model));
    model->data = (const uint8_t *)data;
    model->size = size;
    header = (const ag_pack_header *)data;
    if (memcmp(header->magic, ag_magic, sizeof(ag_magic)) != 0) return -2;
    if (header->version != AG_PACK_VERSION || header->endian_tag != 0x01020304u) return -3;
    if (header->total_size != size) return -4;
    tensor_bytes = (uint64_t)header->tensor_count * sizeof(ag_tensor_desc);
    node_bytes = (uint64_t)header->node_count * sizeof(ag_node_desc);
    model->tensors = ag_model_at(model, header->tensor_offset, tensor_bytes);
    model->nodes = ag_model_at(model, header->node_offset, node_bytes);
    if (!model->tensors || !model->nodes) return -5;
    model->tensor_count = header->tensor_count;
    model->node_count = header->node_count;
    for (index = 0; index < model->tensor_count; ++index) {
        const ag_tensor_desc *tensor = model->tensors + index;
        uint32_t axis;
        if (tensor->id != index || tensor->rank > AG_MAX_RANK) return -6;
        for (axis = 0; axis < tensor->rank; ++axis)
            if (tensor->shape[axis] < 0) return -6;
        if (tensor->type != AG_TYPE_BF16 && tensor->type != AG_TYPE_F32 && tensor->type != AG_TYPE_I32) return -6;
        if (tensor->block && (tensor->block != AG_BLOCK || tensor->rank != 4 || tensor->type != AG_TYPE_BF16)) return -7;
        if ((tensor->pad_h || tensor->pad_w) && !tensor->block) return -7;
        if (tensor->pad_h < 0 || tensor->pad_w < 0) return -7;
        if (tensor->data_size &&
            !ag_model_check_range(model, tensor->data_offset, tensor->data_size)) return -9;
        if (tensor->data_size && ag_tensor_bytes(tensor) != tensor->data_size) return -9;
    }
    for (index = 0; index < model->node_count; ++index) {
        const ag_node_desc *node = model->nodes + index;
        const int32_t *inputs;
        const int32_t *outputs;
        uint32_t edge;
        if (node->id != index) return -10;
        if (node->input_count &&
            !ag_model_check_range(model, node->input_offset,
                                  (uint64_t)node->input_count * sizeof(int32_t))) return -11;
        if (node->output_count &&
            !ag_model_check_range(model, node->output_offset,
                                  (uint64_t)node->output_count * sizeof(int32_t))) return -12;
        inputs = ag_model_at(model, node->input_offset,
                             (uint64_t)node->input_count * sizeof(int32_t));
        outputs = ag_model_at(model, node->output_offset,
                              (uint64_t)node->output_count * sizeof(int32_t));
        for (edge = 0; edge < node->input_count; ++edge)
            if (inputs[edge] != -1 && (inputs[edge] < 0 || (uint32_t)inputs[edge] >= model->tensor_count)) return -13;
        for (edge = 0; edge < node->output_count; ++edge) {
            if (outputs[edge] < 0 || (uint32_t)outputs[edge] >= model->tensor_count) return -14;
            /* an alias output shares the padded buffer of the first input: same storage size */
            if ((model->tensors[outputs[edge]].flags & 8) &&
                (node->input_count < 1 || inputs[0] < 0 ||
                 ag_tensor_bytes(model->tensors + outputs[edge]) != ag_tensor_bytes(model->tensors + inputs[0]))) return -18;
        }
    }
    return 0;
}

size_t ag_tensor_elements(const ag_tensor_desc *tensor) {
    size_t count = 1;
    uint32_t axis;
    if (!tensor) return 0;
    for (axis = 0; axis < tensor->rank; ++axis) count *= (size_t)tensor->shape[axis];
    return count;
}

/* storage bytes: blocked tensors round the last axis up to whole channel
 * blocks and include the pad border; dense tensors are element count x width */
size_t ag_tensor_bytes(const ag_tensor_desc *tensor) {
    size_t width;
    if (!tensor) return 0;
    if (tensor->type == AG_TYPE_F32 || tensor->type == AG_TYPE_I32) width = 4;
    else if (tensor->type == AG_TYPE_BF16) width = 2;
    else return 0;
    if (tensor->block) {
        size_t blocks = ((size_t)tensor->shape[3] + AG_BLOCK - 1) / AG_BLOCK;
        return (size_t)tensor->shape[0] * (size_t)(tensor->shape[1] + 2 * tensor->pad_h) *
               (size_t)(tensor->shape[2] + 2 * tensor->pad_w) * blocks * AG_BLOCK * width;
    }
    return ag_tensor_elements(tensor) * width;
}

ag_view ag_view_of(const ag_tensor_desc *t) {
    ag_view v;
    v.H = t->shape[1]; v.W = t->shape[2]; v.C = t->shape[3];
    v.block = t->block;
    v.blocks = t->block ? (v.C + AG_BLOCK - 1) / AG_BLOCK : 1;
    v.pitch = v.W + 2 * t->pad_w;
    v.plane = (v.H + 2 * t->pad_h) * v.pitch;
    v.origin = t->pad_h * v.pitch + t->pad_w;
    return v;
}
