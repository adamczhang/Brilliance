#!/usr/bin/env python
"""Golden values for tests/ref_hy4_ops_test.cpp: the Hy4-flash ops against Hugging Face transformers
at c2ec8505 (HYV4Flash*), on small random f32 modules.

  HY4_REF=/root/work/claude-scratch/hy4flash/ref  $HY4_REF/venv/bin/python gen_hy4_golden.py > ../../hy4_ref_golden.h

f32 everywhere: the ops' own roundings vanish, so what is compared is the arithmetic and the layouts.
The harness (ref/) supplies the config builder (minicfg.py, the radiance fixture's geometry) and the
patched-in transformers checkout. Nothing here needs a checkpoint.
"""
import copy, os, sys, types
import numpy as np, torch

REF = os.environ.get("HY4_REF", "/root/work/claude-scratch/hy4flash/ref")
sys.path.insert(0, REF)
import minicfg
from transformers.models.hy_v4_flash import modeling_hy_v4_flash as m
from transformers.models.hy_v4_flash.configuration_hy_v4_flash import HYV4FlashConfig

torch.manual_seed(20261121)
rng = np.random.default_rng(20261121)
out = []

def emit(name, arr):
    a = arr.detach().numpy() if isinstance(arr, torch.Tensor) else np.asarray(arr)
    flat = a.reshape(-1)
    if a.dtype.kind == "f":
        def lit(x):
            t = "%.9g" % float(np.float32(x))
            if not any(c in t for c in ".eEni"): t += ".0"
            return t + "f"
        body = ", ".join(lit(x) for x in flat)
        out.append("static const float %s[%d] = {%s};" % (name, flat.size, body))
    else:
        body = ", ".join("%dLL" % int(x) for x in flat)
        out.append("static const long long %s[%d] = {%s};" % (name, flat.size, body))

def cfg(hidden=None):
    full = minicfg.full_config("mini")
    c = HYV4FlashConfig.from_dict(copy.deepcopy(full))
    t = c.text_config
    if hidden: t.hidden_size = hidden
    return t

def fill(mod, scale=0.5):
    for n, p in mod.named_parameters():
        p.data = (torch.randn(p.shape) * scale).float()
    for n, b in mod.named_buffers():
        pass

# ------------------------------------------------------------------ iHC
t = cfg(hidden=32)
hc, n, M = t.hc_mult, t.hidden_size, 3
conn = m.HYV4FlashHyperConnection(t).float()
conn.fn.data = torch.randn(2 * hc, hc * n) * 0.3
conn.base.data = torch.randn(2 * hc) * 1.5
conn.scale.data = torch.tensor([0.9, -0.6])
h = torch.randn(1, M, hc, n) * 2
with torch.no_grad():
    post, x = conn(h)
emit("ihc_n", [n, hc, M]); emit("ihc_h", h); emit("ihc_fn", conn.fn); emit("ihc_base", conn.base)
emit("ihc_scale", conn.scale); emit("ihc_post", post); emit("ihc_x", x)
emit("ihc_eps", [t.rms_norm_eps, t.hc_eps, t.hc_magnitude])
head = m.HYV4FlashHyperHead(t).float()
head.hc_fn.data = torch.randn(hc, hc * n) * 0.3; head.hc_base.data = torch.randn(hc); head.hc_scale.data = torch.tensor([1.3])
with torch.no_grad(): hx = head(h)
emit("head_fn", head.hc_fn); emit("head_base", head.hc_base); emit("head_scale", head.hc_scale); emit("head_x", hx)
# the residual write
y = torch.randn(1, M, n)
with torch.no_grad():
    new = (post.float().unsqueeze(-1) * y.float().unsqueeze(-2) + h.float())
emit("ihc_y", y); emit("ihc_hnew", new)

# ------------------------------------------------------------------ attention with a sink (eager, HF)
def mask_for(mq, nk, window):
    mk = torch.zeros(mq, nk)
    for i in range(mq):
        p = nk - mq + i
        for j in range(nk):
            if j > p or (window and p - j >= window): mk[i, j] = torch.finfo(torch.float32).min
    return mk
mod = types.SimpleNamespace(num_key_value_groups=1, training=False)
H, D, V, mq, nk = 2, 12, 8, 7, 12
q = torch.randn(1, H, mq, D); k = torch.randn(1, H, nk, D); v = torch.randn(1, H, nk, V)
mod.sinks = torch.randn(H)
emit("kvs_q", q.permute(0, 2, 1, 3)); emit("kvs_k", k.permute(0, 2, 1, 3)); emit("kvs_v", v.permute(0, 2, 1, 3)); emit("kvs_sinks", mod.sinks)
for w in (0, 4):
    o, _ = m.eager_attention_forward(mod, q, k, v, mask_for(mq, nk, w)[None, None], D ** -0.5)
    emit("kvs_out_w%d" % w, o)       # [1, mq, H, V]
# the latent form: key = [latent | rope], value = latent, one key row for every head
L, R, T = 16, 4, 11
lat = torch.randn(T, L); rope = torch.randn(T, R)
qa = torch.randn(1, H, 2, L); qr = torch.randn(1, H, 2, R)
bs = 4; pages = 4; order = [2, 0, 3]                   # page ids of blocks 0..2, pool of 4 pages
cache = torch.randn(pages, bs, L + R)
for t_ in range(T): cache[order[t_ // bs], t_ % bs] = torch.cat([lat[t_], rope[t_]])
mod2 = types.SimpleNamespace(num_key_value_groups=1, training=False, sinks=torch.randn(H))
key = torch.cat([lat, rope], -1)[None, None].expand(1, H, T, L + R)
val = lat[None, None].expand(1, H, T, L)
qq = torch.cat([qa, qr], -1)
emit("mla_q", qa.permute(0, 2, 1, 3)); emit("mla_qr", qr.permute(0, 2, 1, 3)); emit("mla_cache", cache)
emit("mla_sinks", mod2.sinks); emit("mla_order", order)
for w in (0, 5):
    o, _ = m.eager_attention_forward(mod2, qq, key.contiguous(), val.contiguous(), mask_for(2, T, w)[None, None], 256 ** -0.5)
    emit("mla_out_w%d" % w, o)     # [1, 2, H, L]

# ------------------------------------------------------------------ the compressed indexer
t = cfg(hidden=32)
ix = m.HYV4FlashCompressedIndexer(t).float()
for nme, p in ix.named_parameters():
    p.data = torch.randn(p.shape) * (1.0 if "ape" in nme or "norm" in nme else 0.5)
ix.token_key_norm.weight.data = torch.rand(ix.head_dim) + 0.5
ix.key_norm.weight.data = torch.rand(ix.head_dim) + 0.5
ratio, nd = ix.compress_ratio, ix.head_dim
Tn = 41; G = Tn // ratio
tk = torch.randn(1, Tn, nd) * 1.5
for mode in ("gptj", "imrope"):
    if mode == "gptj":
        pos = torch.arange(Tn)[None].expand(3, 1, Tn).contiguous()
    else:
        pos = torch.stack([torch.arange(Tn), torch.randint(0, 50, (Tn,)), torch.randint(0, 50, (Tn,))])[:, None]
    with torch.no_grad():
        pooled = ix.pool_groups(tk)
        gp = pos[..., : G * ratio : ratio]
        gc, gs = ix.rotary_emb(pooled, gp, layer_type="indexed_attention")
        keys = m._apply_partial_rotary_pos_emb(pooled.unsqueeze(2), gc, gs, ix.rope_dim, unsqueeze_dim=2).squeeze(2)
    if mode == "gptj":
        emit("pool_k", tk[0]); emit("pool_wv", ix.key_value_projection.weight); emit("pool_wg", ix.key_gate_projection.weight)
        emit("pool_ape", ix.intra_group_ape); emit("pool_gain", ix.key_norm.weight)
        emit("pool_dims", [nd, ratio, ix.rope_dim, G, Tn])
        emit("pool_eps_theta", [t.rms_norm_eps, float(t.rope_parameters["indexed_attention"]["rope_theta"])])
    emit("pool_pos_%s" % mode, pos[:, 0, : G * ratio : ratio] if mode == "imrope" else pos[0, 0, : G * ratio : ratio])
    emit("pool_out_%s" % mode, keys[0])
# scores and the recall of every query
hid = torch.randn(1, Tn, t.hidden_size); qres = torch.randn(1, Tn, t.q_lora_rank)
pos = torch.arange(Tn)[None].expand(3, 1, Tn).contiguous()
with torch.no_grad():
    pooled = ix.pool_groups(tk)
    gp = pos[..., : G * ratio : ratio]
    qgi = (torch.arange(Tn) // ratio)[None]
    recalled = ix(hid, qres, pooled, gp, pos, qgi)             # [1, Tn, group_topk]
    query = ix.query_projection(qres).view(1, Tn, ix.num_heads, nd)
    qc, qs = ix.rotary_emb(query, pos, layer_type="indexed_attention")
    query = m._apply_partial_rotary_pos_emb(query, qc, qs, ix.rope_dim, unsqueeze_dim=2)
    gc, gs = ix.rotary_emb(pooled, gp, layer_type="indexed_attention")
    keys = m._apply_partial_rotary_pos_emb(pooled.unsqueeze(2), gc, gs, ix.rope_dim, unsqueeze_dim=2).squeeze(2)
    gate = ix.head_weight_projection(hid)
    hw = gate.float() * (ix.num_heads ** -0.5 * nd ** -0.5)
    sc = torch.relu(torch.matmul(query.float(), keys.float().unsqueeze(1).transpose(-1, -2)))
    logits = torch.matmul(hw.unsqueeze(-2), sc).squeeze(-2)    # [1, Tn, G]
emit("ix_dims", [Tn, G, ix.num_heads, nd, ix.group_topk]); emit("ix_q", query[0]); emit("ix_gate", gate[0])
emit("ix_keys", keys[0]); emit("ix_logits", logits[0]); emit("ix_recalled", recalled[0].numpy())

# ------------------------------------------------------------------ Engram
t = cfg(hidden=32)
lay = m.HYV4FlashEngramLayer(t, t.engram_layer_ids[0]).float()
fill(lay, 0.4)
lay.lookup_table.copy_(torch.randint(0, 200, (t.engram_lookup_table_size,), dtype=torch.int32))
for nme in ("unified_pre_gamma",): getattr(lay, nme).data = torch.randn(t.hidden_size)
for p in lay.unified_per_hc_scales: p.data = torch.randn(t.hidden_size) * 0.5
for nm in list(lay.lookup_per_head_norms) + list(lay.norm_key) + list(lay.norm_query):
    nm.weight.data = torch.rand(nm.weight.shape) + 0.5
V_, MAP, PAD = 320, t.engram_lookup_table_size, t.engram_pad_id
docs = [[0, 5, 17, 301, 2, 311, 319], [0, 9, 9, 313, 250, 3]]            # ids past the map, the pad id inside
ids_in = torch.tensor([sum(docs, [])])
positions = torch.tensor([[*range(len(docs[0])), *range(len(docs[1]))]])
fake = types.SimpleNamespace(config=t)
ng = m.HYV4FlashTextModel.build_ngram_tokens(fake, ids_in, positions)
with torch.no_grad():
    hash_ids = lay.hash_ngrams(ng)
IMG = 311
mask = ((ids_in != IMG) & (ids_in != 313) & (ids_in != PAD))
emit("eg_tok", ids_in[0].numpy()); emit("eg_pos", positions[0].numpy()); emit("eg_map", lay.lookup_table.numpy().astype(np.int64))
emit("eg_mult", lay.hash_multipliers.numpy()); emit("eg_vocab", lay.head_vocab_sizes.numpy())
emit("eg_off", lay.multi_head_embedding.offsets.numpy()); emit("eg_dims", [PAD, MAP, t.engram_n_head_per_ngram])
emit("eg_ids", (hash_ids + lay.multi_head_embedding.offsets)[0].numpy()); emit("eg_mask", mask[0].numpy().astype(np.int64))
# norm, projections, gate: hooks on the module's own pieces
Tn = ids_in.shape[1]; hcm, hn = t.hc_mult, t.hidden_size
hs = torch.randn(1, Tn, hcm, hn) * 1.3
cap = {}
for c, kp in enumerate(lay.key_projs): kp.register_forward_hook(lambda mod_, i_, o_, c=c: cap.__setitem__("k%d" % c, o_.detach()))
lay.value_proj.register_forward_hook(lambda mod_, i_, o_: cap.__setitem__("v", o_.detach()))
with torch.no_grad():
    emb = lay.multi_head_embedding(hash_ids).to(hs.dtype)
    if lay.fp8_embedding: emb = m._fp8_round_trip(emb)
    emit("en_raw", emb.flatten(-2)[0])
    normed = torch.stack([nm(emb[:, :, hh]) for hh, nm in enumerate(lay.lookup_per_head_norms)], dim=2).flatten(-2)
    emit("en_gain", torch.cat([nm.weight for nm in lay.lookup_per_head_norms])); emit("en_out_f32eps", normed[0])
    emit("en_eps", [torch.finfo(torch.float32).eps, 0.0078125])
    for nm in lay.lookup_per_head_norms: nm.eps = 0.0078125
    normed2 = torch.stack([nm(emb[:, :, hh]) for hh, nm in enumerate(lay.lookup_per_head_norms)], dim=2).flatten(-2)
    emit("en_out_big", normed2[0])
    for nm in lay.lookup_per_head_norms: nm.eps = None
    outp = lay(hs, ng, mask)
    emit("eg_dims2", [Tn, hcm, hn])
    emit("eg_key", torch.cat([cap["k%d" % c] for c in range(hcm)], -1)[0]); emit("eg_val", cap["v"][0]); emit("eg_h", hs[0].reshape(Tn, -1))
    emit("eg_wkey", torch.cat([nm.weight for nm in lay.norm_key])); emit("eg_wq", torch.cat([nm.weight for nm in lay.norm_query]))
    emit("eg_wval", lay.unified_pre_gamma); emit("eg_wbr", torch.cat(list(lay.unified_per_hc_scales)))
    emit("eg_hnew_f32eps", (hs + outp)[0].reshape(Tn, -1))
    for nm in list(lay.norm_key) + list(lay.norm_query) + [lay.unified_rmsnorm]: nm.eps = 0.0078125
    emit("eg_hnew_big", (hs + lay(hs, ng, mask))[0].reshape(Tn, -1))

print("// Generated by tests/data/hy4/gen_hy4_golden.py from transformers c2ec8505 (HYV4Flash*); do not edit.")
print("#pragma once")
print("\n".join(out))
