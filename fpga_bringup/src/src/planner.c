#include <animegan/platform.h>
#include <animegan/runtime.h>
#include <string.h>

/* Static activation plan: every runtime tensor except the graph input
 * gets a 64-byte aligned offset in one arena so that no two tensors whose
 * lifetimes [producer, last use] overlap share bytes. The operator scratch
 * (largest per-node requirement for the backend) is a block that lives for
 * the whole run. Lifetimes are taken from the node order. First-fit, largest
 * tensor first. The arena gets AG_PLAN_TAIL extra readable bytes at the end
 * because an AME tile load reads past the last row it uses. */
#define AG_PLAN_ALIGN 64u

static uint64_t align_up(uint64_t v) {
    return (v + AG_PLAN_ALIGN - 1) / AG_PLAN_ALIGN * AG_PLAN_ALIGN;
}

typedef struct {
    uint64_t offset, size;
    uint32_t start, end;
} ag_block;

/* lowest offset where [size] does not collide with any live block overlapping [start,end] */
static uint64_t first_fit(const ag_block *blocks, uint32_t count, uint32_t start, uint32_t end, uint64_t size) {
    uint64_t offset = 0;
    int moved = 1;
    while (moved) {
        uint32_t i;
        moved = 0;
        for (i = 0; i < count; ++i) {
            const ag_block *b = blocks + i;
            if (b->end < start || b->start > end) continue;             /* disjoint lifetimes */
            if (offset < b->offset + b->size && b->offset < offset + size) {
                offset = align_up(b->offset + b->size);                 /* collide: jump past */
                moved = 1;
            }
        }
    }
    return offset;
}

/* an alias tensor (flag 8) is the padded buffer of its producer's first
 * input: it is not placed itself, its uses extend that tensor's lifetime and
 * it receives the same offset */
static uint32_t alias_root(const uint32_t *alias_of, uint32_t t) {
    while (alias_of[t] != UINT32_MAX) t = alias_of[t];
    return t;
}

static int lifetimes(const ag_model *model, uint32_t *start, uint32_t *end, uint32_t *alias_of) {
    uint32_t t, node_id, edge;
    for (t = 0; t < model->tensor_count; ++t) { start[t] = UINT32_MAX; end[t] = 0; alias_of[t] = UINT32_MAX; }
    for (node_id = 0; node_id < model->node_count; ++node_id) {
        const ag_node_desc *node = model->nodes + node_id;
        const int32_t *inputs = ag_model_at(model, node->input_offset, node->input_count * sizeof(int32_t));
        const int32_t *outputs = ag_model_at(model, node->output_offset, node->output_count * sizeof(int32_t));
        if (!inputs || !outputs) return -3;
        for (edge = 0; edge < node->output_count; ++edge)
            if (outputs[edge] >= 0 && (model->tensors[outputs[edge]].flags & 8) && inputs[0] >= 0) alias_of[outputs[edge]] = (uint32_t)inputs[0];
        for (edge = 0; edge < node->input_count; ++edge)
            if (inputs[edge] >= 0) {
                uint32_t root = alias_root(alias_of, (uint32_t)inputs[edge]);
                if (end[root] < node_id) end[root] = node_id;
            }
        for (edge = 0; edge < node->output_count; ++edge)
            if (outputs[edge] >= 0 && alias_of[outputs[edge]] == UINT32_MAX && start[outputs[edge]] == UINT32_MAX) start[outputs[edge]] = node_id;
    }
    return 0;
}

int ag_plan_build(const ag_model *model, ag_backend backend, uint64_t offsets[], ag_plan *plan) {
    uint32_t *start, *end, *alias_of;
    ag_block *blocks;
    uint32_t t, node_id, count = 0;
    uint64_t scratch = 0;
    int rc = 0;
    if (!model || !offsets || !plan) return -1;
    start = ag_plat_alloc(model->tensor_count * sizeof(uint32_t));
    end = ag_plat_alloc(model->tensor_count * sizeof(uint32_t));
    alias_of = ag_plat_alloc(model->tensor_count * sizeof(uint32_t));
    blocks = ag_plat_alloc((model->tensor_count + 1) * sizeof(ag_block));
    if (!start || !end || !alias_of || !blocks) { rc = -2; goto done; }
    for (t = 0; t < model->tensor_count; ++t) offsets[t] = UINT64_MAX;
    if ((rc = lifetimes(model, start, end, alias_of)) != 0) goto done;
    for (node_id = 0; node_id < model->node_count; ++node_id) {
        size_t need = ag_node_scratch_bytes(model, node_id, backend);
        if (need > scratch) scratch = need;
    }
    /* scratch first, alive for the whole run */
    plan->scratch_bytes = align_up(scratch);
    plan->scratch_offset = 0;
    if (scratch) {
        blocks[count].offset = 0; blocks[count].size = plan->scratch_bytes;
        blocks[count].start = 0; blocks[count].end = model->node_count;
        ++count;
    }
    plan->arena_bytes = plan->scratch_bytes;
    plan->placed = 0;
    /* runtime tensors with a producer, largest first (ties: lower id first);
     * size-descending first-fit packs closer to the live peak than placing in
     * producer order */
    {
        uint32_t *order = ag_plat_alloc(model->tensor_count * sizeof(uint32_t)), n = 0, i, j;
        if (!order) { rc = -2; goto done; }
        for (t = 0; t < model->tensor_count; ++t)
            if (!model->tensors[t].data_size && start[t] != UINT32_MAX) order[n++] = t;
        for (i = 1; i < n; ++i) {                               /* insertion sort by size desc */
            uint32_t key = order[i];
            uint64_t ks = ag_tensor_bytes(model->tensors + key);
            for (j = i; j > 0 && ag_tensor_bytes(model->tensors + order[j - 1]) < ks; --j) order[j] = order[j - 1];
            order[j] = key;
        }
        for (i = 0; i < n; ++i) {
            const ag_tensor_desc *desc;
            uint64_t size, s, e;
            t = order[i];
            desc = model->tensors + t;
            size = align_up(ag_tensor_bytes(desc));
            s = start[t];
            e = (desc->flags & 4) ? model->node_count : (end[t] > start[t] ? end[t] : start[t]);
            offsets[t] = first_fit(blocks, count, (uint32_t)s, (uint32_t)e, size);
            blocks[count].offset = offsets[t]; blocks[count].size = size;
            blocks[count].start = (uint32_t)s; blocks[count].end = (uint32_t)e;
            ++count;
            if (offsets[t] + size > plan->arena_bytes) plan->arena_bytes = offsets[t] + size;
            ++plan->placed;
        }
        ag_plat_free(order);
    }
    for (t = 0; t < model->tensor_count; ++t)
        if (alias_of[t] != UINT32_MAX) offsets[t] = offsets[alias_root(alias_of, t)];
    plan->arena_bytes += AG_PLAN_TAIL;
done:
    ag_plat_free(start); ag_plat_free(end); ag_plat_free(alias_of); ag_plat_free(blocks);
    return rc;
}

/* Check a plan: every pair of placed tensors with overlapping lifetimes must
 * occupy disjoint bytes; used by the runtime test. Returns the number of
 * violations (0 = consistent). */
uint32_t ag_plan_check(const ag_model *model, const uint64_t offsets[], const ag_plan *plan) {
    uint32_t *start, *end, *alias_of, a, b, bad = 0;
    start = ag_plat_alloc(model->tensor_count * sizeof(uint32_t));
    end = ag_plat_alloc(model->tensor_count * sizeof(uint32_t));
    alias_of = ag_plat_alloc(model->tensor_count * sizeof(uint32_t));
    if (!start || !end || !alias_of) return UINT32_MAX;
    if (lifetimes(model, start, end, alias_of) != 0) return UINT32_MAX;
    for (a = 0; a < model->tensor_count; ++a) {
        uint64_t sa, ea_off;
        uint32_t ea;
        if (offsets[a] == UINT64_MAX) continue;
        if (alias_of[a] != UINT32_MAX) {   /* shares its root's buffer by design; sizes must agree */
            if (offsets[a] != offsets[alias_root(alias_of, a)] || ag_tensor_bytes(model->tensors + a) != ag_tensor_bytes(model->tensors + alias_root(alias_of, a))) ++bad;
            continue;
        }
        sa = ag_tensor_bytes(model->tensors + a);
        ea = (model->tensors[a].flags & 4) ? model->node_count : (end[a] > start[a] ? end[a] : start[a]);
        ea_off = offsets[a] + sa;
        if (offsets[a] % AG_PLAN_ALIGN || ea_off > plan->arena_bytes) ++bad;
        if (plan->scratch_bytes && offsets[a] < plan->scratch_offset + plan->scratch_bytes && plan->scratch_offset < ea_off) ++bad;
        for (b = a + 1; b < model->tensor_count; ++b) {
            uint32_t eb;
            if (offsets[b] == UINT64_MAX || alias_of[b] != UINT32_MAX) continue;
            eb = (model->tensors[b].flags & 4) ? model->node_count : (end[b] > start[b] ? end[b] : start[b]);
            if (ea < start[b] || eb < start[a]) continue;
            if (offsets[a] < offsets[b] + ag_tensor_bytes(model->tensors + b) && offsets[b] < ea_off) ++bad;
        }
    }
    ag_plat_free(start); ag_plat_free(end); ag_plat_free(alias_of);
    return bad;
}
