-- tools/gen/index_semantics.lua
-- Operator index-semantics annotation table (C4/D13 1-based conversion).
-- Stage: P3 / M1 (WORKPLAN 2.3.1). MUST be written before the generator.
--
-- Why this file exists (process/conventions.md sec.5.2):
--   ops.yaml has NO information about which parameters/outputs are
--   index semantics -- that table is maintained by hand here. The
--   generator MUST error on any operator NOT present in this table
--   instead of defaulting to pass-through: a silently skipped
--   off-by-one is the hardest bug class in this project.
--
-- Annotation forms (conventions.md sec.5.1):
--   takes_index  = { "arg", ... }   args that are indices (label, ids,
--                                   token ids): their VALUES are 1-based
--                                   in Lua and must be decremented by 1
--                                   before the C++ call.
--   returns_index = { 1, ... }      1-based OUTPUT POSITIONS that are
--                                   indices: the C++ result's values are
--                                   0-based and must be incremented by 1
--                                   before returning to Lua.
--
-- Sources read 2026-10-24 (paddle/phi/ops/yaml/ops.yaml at
-- $PADDLE_ROOT, branch windows_dataloader_multiprocess, v3.5.0):
--   argmax  :346   args (Tensor x, Scalar(int64_t) axis, ...) -> Tensor(out)
--   argmin  :358   same shape as argmax
--   argsort :370   output Tensor(out), Tensor(indices)
--   index_select :2981  args (Tensor x, Tensor index, int axis=0)
--   index_sample :2968  args (Tensor x, Tensor index)
--   index_add    :2895  args (Tensor x, Tensor index, Tensor add_value, int axis=0)
--   gather  :2516   args (Tensor x, Tensor index, Scalar axis=0)
--   gather_nd :2527 args (Tensor x, Tensor index)
--   scatter :4868   args (Tensor x, Tensor index, Tensor updates, ...)
--   topk    :5646   output Tensor(out), Tensor(indices)
--   one_hot :4191   args (Tensor x, Scalar(int) num_classes)   -- x is label
--   cross_entropy_with_softmax :1275  args (Tensor input, Tensor label, ...)
--
-- This table covers ALL forward ops in the generator's input yamls
-- (2.3.4 full-generation annotation, 2026-10-24). Every entry carries the
-- yaml block line or paddle/python source it was verified against; see the
-- per-entry comments below. The generator refuses to emit any operator NOT
-- present here (require_annotated) -- a skipped annotation is a silent
-- off-by-one, conventions.md sec.5.2. Ops that cannot be expressed with
-- the per-argument -1/+1 conversion (graph-learning CSC structures, etc.)
-- are SKIPPED at emit level instead, via tools/gen/skip_rules.lua.

local M = {}

-- op -> { takes_index = {param names} }
-- Parameters whose VALUE carries index semantics (label / ids / token ids),
-- i.e. 1-based in Lua, decrement by 1 before the C++ call.
M.takes_index = {
  index_select = { "index" },
  index_sample = { "index" },
  index_add    = { "index" },
  gather       = { "index" },
  gather_nd    = { "index" },
  scatter      = { "index" },
  one_hot      = { "x" },        -- x is the class label (1..num_classes)
  cross_entropy_with_softmax = { "label" },

  -- 2.3.4 full-generation annotation (each entry verified against source,
  -- see process/open-questions.md and the 2.3.4 survey dump):
  --   ops.yaml block lines come from tools/gen/yaml_subset.lua IR dump;
  --   Python semantics from paddle/python/paddle (docstrings).

  -- class-id / token-id / expert-id / index-tensor args (ops.yaml lines)
  accuracy            = { "indices", "label" },  -- ops.yaml:24 top-k preds + truth
  beam_search         = { "pre_ids", "ids" },    -- ops.yaml:613 token ids
  c_embedding         = { "x" },                 -- dygraph_ops.yaml:59
  c_softmax_with_cross_entropy = { "label" },    -- ops.yaml:889
  chunk_eval          = { "label" },             -- ops.yaml:6152
  class_center_sample = { "label" },             -- ops.yaml:1030
  crf_decoding        = { "label" },             -- ops.yaml:1240
  cross_entropy_with_softmax_bwd_w_downcast = { "label" },  -- ops.yaml:1288
  embedding           = { "x" },                 -- dygraph_ops.yaml:128
  embedding_grad_add_to  = { "token_indices" },  -- ops.yaml:1744
  embedding_grad_dense  = { "x" },               -- dygraph_ops.yaml:142
  embedding_with_scaled_gradient = { "x" },      -- ops.yaml:1756
  expand_modality_expert_id = { "expert_id" },   -- ops.yaml:1884
  flashmask_attention = { "startend_row_indices" },   -- ops.yaml:2182 mask row idx
  flashmask_attention_v2 = { "startend_row_indices" },-- ops.yaml:2196
  gather_tree         = { "ids", "parents" },    -- ops.yaml:2539 beam ids
  hsigmoid_loss       = { "label" },             -- ops.yaml:2783 (path/code are huffman codes)
  index_elementwise_get  = { "index" },          -- ops.yaml:2907
  index_elementwise_put  = { "index" },          -- ops.yaml:2917
  index_elementwise_put_with_tensor = { "index" }, -- ops.yaml:2929
  index_fill          = { "index" },             -- ops.yaml:2941
  index_put           = { "indices" },           -- ops.yaml:2955
  index_select_strided = { "index" },            -- ops.yaml:2995 scalar index
  label_smooth        = { "label" },             -- ops.yaml:3143
  lookup_table_dequant = { "ids" },              -- ops.yaml:3475
  margin_cross_entropy = { "label" },            -- ops.yaml:3559
  moe_combine         = { "scatter_index" },     -- ops.yaml:3874
  moe_combine_auto    = { "scatter_index" },     -- ops.yaml:3884
  moe_combine_no_weight = { "scatter_index" },   -- ops.yaml:3895
  moe_gate_dispatch_partial_nosoftmaxtopk = { "expert_id" }, -- ops.yaml:3938
  moe_permute         = { "expert_routemap_topk" }, -- ops.yaml:3960
  moe_unpermute       = { "zipped_expertwise_rowmap" }, -- ops.yaml:3970
  multiplex           = { "index" },             -- ops.yaml:4038 selects from list
  nll_loss            = { "label" },             -- ops.yaml:4121
  prune_gate_by_capacity = { "gate_idx" },       -- ops.yaml:4402
  put_along_axis      = { "indices" },           -- ops.yaml:4425
  random_routing      = { "topk_idx" },          -- ops.yaml:4504
  scatter_nd_add      = { "index" },             -- ops.yaml:4881
  segment_pool        = { "segment_ids" },       -- ops.yaml:4904
  send_u_recv         = { "src_index", "dst_index" },  -- ops.yaml:4929
  send_ue_recv        = { "src_index", "dst_index" },  -- ops.yaml:4940
  send_uv             = { "src_index", "dst_index" },  -- ops.yaml:4951
  shard_index         = { "input" },             -- ops.yaml:5061 label ids
  take_along_axis     = { "indices" },           -- ops.yaml:5535
  unpool              = { "indices" },           -- ops.yaml:5879
  unpool3d            = { "indices" },           -- ops.yaml:5890
  warpctc             = { "label" },             -- ops.yaml:6004
  warprnnt            = { "label" },             -- ops.yaml:6017
  yolo_loss           = { "gt_label" },          -- ops.yaml:6120
}

-- op -> { output positions 1-based }
-- Outputs whose VALUES are indices (0-based from C++), increment by 1
-- before returning to Lua.
M.returns_index = {
  argmax = { 1 },
  argmin = { 1 },
  argsort = { 2 },   -- second output (indices)
  topk = { 2 },      -- second output (indices)

  -- 2.3.4 full-generation annotation; each entry verified against source:
  --   median  -> median_kernel.cc OutputAt(1) INT64 = median_index
  --   multinomial -> python/paddle/tensor/random.py:416 "sampled category index"
  --   moe_gate_dispatch -> python/paddle/incubate/nn/functional/moe_gate_dispatch.py
  --                        scatter_index/expert_offset/expert_id (int32/int64)
  --   reindex_graph -> python/paddle/geometric/reindex.py (indices into out_nodes)
  --   graph_*       -> graph_khop_sampler.py / neighbors.py (sample_index / eids)
  beam_search = { 1, 3 },        -- selected_ids, parent_idx (ops.yaml:613)
  bipartite_match = { 1 },       -- col_to_row_match_indices (ops.yaml:696)
  build_src_rank_and_local_expert_id = { 2 },  -- local_expert_id (ops.yaml:836)
  class_center_sample = { 1 },   -- remapped_label (ops.yaml:1030)
  crf_decoding = { 1 },          -- viterbi_path (ops.yaml:1240)
  cummax = { 2 },                -- indices (ops.yaml:1323)
  cummin = { 2 },                -- indices (ops.yaml:1335)
  distribute_fpn_proposals = { 3 },  -- restore_index (dygraph_ops.yaml:71)
  expand_modality_expert_id = { 1 }, -- expert_id_out (ops.yaml:1884)
  gather_tree = { 1 },           -- collected ids (ops.yaml:2539)
  kthvalue = { 2 },              -- indices (ops.yaml:3121)
  matrix_nms = { 2 },            -- index (ops.yaml:3627)
  max_pool2d_with_index = { 2 }, -- mask (ops.yaml:3690)
  max_pool3d_with_index = { 2 }, -- mask (ops.yaml:3700)
  max_with_index = { 2 },        -- indices (ops.yaml:3710)
  min_with_index = { 2 },        -- indices (ops.yaml:3839)
  mode = { 2 },                  -- indices (ops.yaml:3864)
  moe_gate_dispatch = { 3, 4, 5 },               -- scatter/expert_offset/expert_id (ops.yaml:3905)
  moe_gate_dispatch_and_quant = { 4, 5, 6 },     -- (ops.yaml:3916)
  moe_gate_dispatch_auto = { 3, 4, 5 },          -- (ops.yaml:3926)
  moe_gate_dispatch_partial_nosoftmaxtopk = { 3, 4, 5 },  -- scatter/rev/expert_offset (ops.yaml:3938)
  moe_gate_dispatch_permute = { 3, 4, 5 },       -- (ops.yaml:3949)
  moe_permute = { 2, 5 },        -- zipped_expertwise_rowmap, expert_indices (ops.yaml:3960)
  multiclass_nms3 = { 2 },       -- index (ops.yaml:4015)
  multinomial = { 1 },           -- sampled category index (random.py:416)
  nonzero = { 1 },               -- coordinates (ops.yaml:4143)
  prune_gate_by_capacity = { 1 }, -- out_gate_idx (ops.yaml:4402)
  randperm = { 1 },              -- permutation (ops.yaml:4516)
  restrict_nonzero = { 1 },      -- coordinates (ops.yaml:4701)
  sequence_pool = { 2 },         -- max_index (ops.yaml:4985)
  shuffle_batch = { 2 },         -- shuffle_idx (ops.yaml:5082)
  top_p_sampling = { 2, 4 },     -- ids, topk_ids (ops.yaml:5634)
  tril_indices = { 1 },          -- coordinates (ops.yaml:5712)
  triu_indices = { 1 },          -- coordinates (ops.yaml:5752)
  unique = { 2, 3 },             -- indices, inverse (dygraph_ops.yaml:415)
  unique_consecutive = { 2 },    -- index (ops.yaml:5867)
  viterbi_decode = { 2 },        -- path (ops.yaml:5993)
  median = { 2 },                -- median_index (median_kernel.cc:252 INT64)
  nanmedian = { 2 },             -- median_index, same kernel
  shard_index = { 1 },           -- output is the shard-local offset; -1 (ignore_value)
                                 -- becomes 0 after +1, used as Lua ignore marker
                                 -- (manipulation.py:799 shard_index docstring)
  random_routing = { 1 },        -- inplace topk_idx->out, output IS the index
                                 -- (ops.yaml:4504)
}

-- op -> true
-- Operators whose args/outputs were READ and found to carry NO index
-- semantics at all. The generator must still see the operator in this
-- table before emitting it (conventions.md sec.5.2: never default an
-- unannotated operator through); `no_index` is the explicit "I checked,
-- nothing to convert" annotation. An operator is only allowed to skip
-- conversion if a human has read its yaml and put it HERE (or in
-- takes_index / returns_index).
--
-- Verified 2026-10-24 (2.3.3 trial generation):
--   add    : inconsistent/dygraph_ops.yaml:4   (Tensor x, Tensor y) -> Tensor(out)
--   matmul : inconsistent/dygraph_ops.yaml:248 (Tensor x, Tensor y,
--            bool transpose_x=false, bool transpose_y=false) -> Tensor
--   relu   : ops.yaml:4626                     (Tensor x) -> Tensor(out)
M.no_index = {
  abs = true,
  accuracy_check = true,
  acos = true,
  acosh = true,
  add = true,
  add_n = true,
  add_position_encoding = true,
  addmm = true,
  affine_channel = true,
  affine_grid = true,
  all = true,
  all_reduce = true,
  all_to_all = true,
  allclose = true,
  amax = true,
  amin = true,
  aminmax = true,
  angle = true,
  any = true,
  ap_facade = true,
  ap_trivial_fusion_begin = true,
  ap_trivial_fusion_end = true,
  ap_variadic = true,
  apply_per_channel_scale = true,
  arange = true,
  as_complex = true,
  as_real = true,
  as_strided = true,
  asin = true,
  asinh = true,
  assign = true,
  assign_pos = true,
  atan = true,
  atan2 = true,
  atanh = true,
  attention_lstm = true,
  baddbmm = true,
  barrier = true,
  batch_fc = true,
  batch_norm = true,
  batched_gemm = true,
  bernoulli = true,
  bicubic_interp = true,
  bilinear = true,
  bilinear_interp = true,
  bincount = true,
  binomial = true,
  bitwise_and = true,
  bitwise_left_shift = true,
  bitwise_not = true,
  bitwise_or = true,
  bitwise_right_shift = true,
  bitwise_xor = true,
  bmm = true,
  box_clip = true,
  box_coder = true,
  broadcast = true,
  broadcast_tensors = true,
  c_concat = true,
  c_identity = true,
  c_split = true,
  calc_reduced_attn_scores = true,
  cast = true,
  ceil = true,
  celu = true,
  channel_shuffle = true,
  check_numerics = true,
  cholesky = true,
  cholesky_solve = true,
  clip = true,
  clip_by_norm = true,
  coalesce_tensor = true,
  collect_fpn_proposals = true,
  complex = true,
  concat = true,
  conj = true,
  conv2d = true,
  conv2d_transpose = true,
  conv2d_transpose_bias = true,
  conv3d = true,
  conv3d_transpose = true,
  copy_to = true,
  copysign = true,
  correlation = true,
  cos = true,
  cosh = true,
  crop = true,
  cross = true,
  ctc_align = true,
  cudnn_lstm = true,
  cumprod = true,
  cumsum = true,
  cvm = true,
  data = true,
  decayed_adagrad = true,
  decode_jpeg = true,
  deformable_conv = true,
  depend = true,
  depthwise_conv2d = true,
  depthwise_conv2d_bias = true,
  depthwise_conv2d_transpose = true,
  depthwise_conv3d_bias = true,
  dequantize_abs_max = true,
  dequantize_log = true,
  det = true,
  dgc_clip_by_norm = true,
  dgc_momentum = true,
  diag = true,
  diag_embed = true,
  diagonal = true,
  digamma = true,
  dirichlet = true,
  dist = true,
  div_scale = true,
  divide = true,
  dot = true,
  dpsgd = true,
  dropout = true,
  edit_distance = true,
  eig = true,
  eigh = true,
  eigvals = true,
  eigvalsh = true,
  einsum = true,
  elementwise_pow = true,
  elu = true,
  empty = true,
  empty_like = true,
  equal = true,
  equal_all = true,
  erf = true,
  erfinv = true,
  exp = true,
  expand = true,
  expand_as = true,
  expm1 = true,
  eye = true,
  fake_channel_wise_dequantize_max_abs = true,
  fake_channel_wise_quantize_abs_max = true,
  fake_channel_wise_quantize_dequantize_abs_max = true,
  fake_dequantize_max_abs = true,
  fake_quantize_abs_max = true,
  fake_quantize_dequantize_abs_max = true,
  fake_quantize_dequantize_moving_average_abs_max = true,
  fake_quantize_moving_average_abs_max = true,
  fake_quantize_range_abs_max = true,
  fast_ln = true,
  fast_rms_norm = true,
  fft_c2c = true,
  fft_c2r = true,
  fft_r2c = true,
  fill = true,
  fill_diagonal = true,
  fill_diagonal_tensor = true,
  flash_attn = true,
  flash_attn_qkvpacked = true,
  flash_attn_unpadded = true,
  flash_attn_v3 = true,
  flash_attn_v3_varlen = true,
  flash_attn_varlen_qkvpacked = true,
  flatten = true,
  flip = true,
  floor = true,
  floor_divide = true,
  fmax = true,
  fmin = true,
  fold = true,
  fp8_quant_blockwise = true,
  fractional_max_pool2d = true,
  fractional_max_pool3d = true,
  frame = true,
  frobenius_norm = true,
  ftrl = true,
  full = true,
  full_batch_size_like = true,
  full_int_array = true,
  full_like = true,
  full_with_tensor = true,
  fused_batch_norm_act = true,
  fused_bn_add_activation = true,
  fused_gemm_epilogue = true,
  fused_rms_norm_ext = true,
  fused_rms_norm_quant = true,
  fused_softmax_mask = true,
  fused_softmax_mask_upper_triangle = true,
  gammaincc = true,
  gammaln = true,
  gaussian = true,
  gaussian_inplace = true,
  gelu = true,
  generate_proposals = true,
  greater_equal = true,
  greater_than = true,
  group_norm = true,
  gru = true,
  gru_unit = true,
  hardshrink = true,
  hardsigmoid = true,
  hardswish = true,
  hardtanh = true,
  heaviside = true,
  histogram = true,
  i0 = true,
  i0e = true,
  i1 = true,
  i1e = true,
  identity_loss = true,
  im2sequence = true,
  imag = true,
  increment = true,
  instance_norm = true,
  int_bincount = true,
  interp_antialias = true,
  inverse = true,
  is_empty = true,
  isclose = true,
  isfinite = true,
  isinf = true,
  isnan = true,
  kron = true,
  l1_norm = true,
  layer_norm = true,
  leaky_relu = true,
  lerp = true,
  less_equal = true,
  less_than = true,
  lgamma = true,
  limit_by_capacity = true,
  linear_interp = true,
  linear_v2 = true,
  linspace = true,
  llm_int8_linear = true,
  log = true,
  log10 = true,
  log1p = true,
  log2 = true,
  log_softmax = true,
  logcumsumexp = true,
  logical_and = true,
  logical_not = true,
  logical_or = true,
  logical_xor = true,
  logit = true,
  logsigmoid = true,
  logspace = true,
  logsumexp = true,
  lp_pool2d = true,
  lstm = true,
  lstsq = true,
  lu = true,
  lu_solve = true,
  lu_unpack = true,
  masked_fill = true,
  masked_select = true,
  match_matrix_tensor = true,
  matmul = true,
  matrix_power = true,
  matrix_rank = true,
  matrix_rank_atol_rtol = true,
  matrix_rank_tol = true,
  max = true,
  maximum = true,
  maxout = true,
  mean = true,
  mean_all = true,
  memcpy_d2h = true,
  memcpy_h2d = true,
  memory_efficient_attention = true,
  merge_selected_rows = true,
  meshgrid = true,
  min = true,
  minimum = true,
  mish = true,
  mm_out_dtype = true,
  mp_allreduce_sum = true,
  multi_dot = true,
  multiply = true,
  mv = true,
  nansum = true,
  nearest_interp = true,
  nextafter = true,
  nms = true,
  norm = true,
  not_equal = true,
  npu_identity = true,
  number_count = true,
  numel = true,
  ones = true,
  ones_like = true,
  overlap_add = true,
  p_norm = true,
  pad = true,
  pad3d = true,
  partial_concat = true,
  partial_sum = true,
  pixel_shuffle = true,
  pixel_unshuffle = true,
  poisson = true,
  polygamma = true,
  pool2d = true,
  pool3d = true,
  pow = true,
  prelu = true,
  prior_box = true,
  prod = true,
  psroi_pool = true,
  pyramid_hash = true,
  qr = true,
  randint = true,
  random = true,
  range_v2 = true,
  rank_attention = true,
  read_file = true,
  real = true,
  reciprocal = true,
  reduce = true,
  reduce_as = true,
  relu = true,
  relu6 = true,
  remainder = true,
  renorm = true,
  repeat_interleave = true,
  reshape = true,
  reverse = true,
  rint = true,
  rms_norm = true,
  rnn = true,
  roi_align = true,
  roi_pool = true,
  roll = true,
  round = true,
  rrelu = true,
  rsqrt = true,
  scale = true,
  searchsorted = true,
  selu = true,
  sequence_conv = true,
  sequence_mask = true,
  set = true,
  set_value = true,
  set_value_with_tensor = true,
  shape = true,
  shape64 = true,
  share_data = true,
  shuffle_channel = true,
  sigmoid = true,
  sign = true,
  silu = true,
  sin = true,
  sinh = true,
  slice = true,
  slogdet = true,
  slogdet_v2 = true,
  slow_conv2d_dilated = true,
  slow_conv3d_dilated = true,
  softmax = true,
  softplus = true,
  softshrink = true,
  softsign = true,
  solve = true,
  spectral_norm = true,
  split = true,
  split_with_num = true,
  sqrt = true,
  square = true,
  squared_l2_norm = true,
  squeeze = true,
  stack = true,
  standard_gamma = true,
  stanh = true,
  std = true,
  stft = true,
  strided_slice = true,
  subtract = true,
  sum = true,
  svd = true,
  svdvals = true,
  swiglu = true,
  swish = true,
  sync_calc_stream = true,
  sync_comm_stream = true,
  tan = true,
  tanh = true,
  tanh_shrink = true,
  tdm_child = true,
  temporal_shift = true,
  tensor_unfold = true,
  thresholded_relu = true,
  tile = true,
  trace = true,
  trans_layout = true,
  transpose = true,
  triangular_solve = true,
  tril = true,
  trilinear_interp = true,
  triu = true,
  trunc = true,
  trunc_divide = true,
  truncated_gaussian_random = true,
  unbind = true,
  unfold = true,
  uniform = true,
  uniform_inplace = true,
  uniform_random_batch_size_like = true,
  unsqueeze = true,
  unstack = true,
  var = true,
  variance = true,
  view_dtype = true,
  view_shape = true,
  view_slice = true,
  weight_dequantize = true,
  weight_only_linear = true,
  weight_quantize = true,
  yolo_box = true,
  yolo_box_head = true,
  yolo_box_post = true,
  zeros = true,
  zeros_like = true,

  -- 2.3.4 full-generation annotation. Each entry read against the yaml
  -- IR and (where semantics were ambiguous) the paddle/python source:
  --
  -- distributed collectives: ring_id/rank/nranks are process-group ids,
  -- NOT data indices -> no conversion.
  all_gather = true,
  c_scatter = true,
  global_gather = true,
  global_scatter = true,
  partial_allgather = true,
  reduce_scatter = true,
  -- model diagnostics: plain pass-through
  disable_check_model_nan_inf = true,
  enable_check_model_nan_inf = true,
  -- losses whose `label` is a 0/1 binary TARGET or a continuous value
  -- (not a 1..C class index; converting would corrupt the loss):
  --   bce_loss          ops.yaml:601   label in {0,1}
  --   log_loss          ops.yaml:3336  label in {0,1}
  --   sigmoid_cross_entropy_with_logits ops.yaml:5119 label in {0,1}
  --   hinge_loss        ops.yaml:2762  labels in {-1,1}
  --   kldiv_loss        ops.yaml:3100  label is a distribution
  --   huber_loss        ops.yaml:2795  label is a continuous target
  --   auc               ops.yaml:545   label is a 0/1 binary target
  auc = true,
  bce_loss = true,
  hinge_loss = true,
  huber_loss = true,
  kldiv_loss = true,
  log_loss = true,
  sigmoid_cross_entropy_with_logits = true,
  -- masks / coordinates / probabilities: not index semantics
  masked_scatter = true,          -- mask is bool
  where = true,                   -- condition is bool
  grid_sample = true,             -- grid is normalized coords
  gumbel_softmax = true,          -- output is a distribution
  cal_aux_loss = true,            -- masks + loss stats
  flashmask_get_unique_id = true, -- unique_id is a random seed
  repeat_interleave_with_tensor_index = true, -- repeats is a count
}


-- ---------------------------------------------------------------------------
-- Query API for the generator
-- ---------------------------------------------------------------------------

-- Returns true if the operator is known to the table at all.
function M.has(op)
  return M.takes_index[op] ~= nil or M.returns_index[op] ~= nil or M.no_index[op] == true
end

-- Params of `op` that are 1-based indices (empty table if none annotated).
function M.index_params(op)
  return M.takes_index[op] or {}
end

-- 1-based output positions of `op` that carry index values.
function M.index_outputs(op)
  return M.returns_index[op] or {}
end

-- Hard guard for the generator (conventions.md sec.5.2): emit a fatal
-- error unless the operator is annotated. NEVER default to pass-through;
-- a skipped annotation is a silent off-by-one.
--   op_name: the yaml `op :` name
--   where:   caller context for the error message
function M.require_annotated(op_name, where)
  if not M.has(op_name) then
    error(string.format(
      "index_semantics: operator '%s' is not annotated (%s). "
      .. "Read its args/output in paddle/phi/ops/yaml/ops.yaml and add an "
      .. "entry to tools/gen/index_semantics.lua before generating. "
      .. "Never default an unannotated operator through.",
      tostring(op_name), tostring(where)))
  end
end

return M