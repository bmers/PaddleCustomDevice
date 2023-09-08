# from __future__ import print_function, division

# import os
# import numpy
# import unittest
# import paddle

# from .lla,a_layer_pass import llama_fuse_attention_layer, llama_fuse_attention_parallel_layer, remove_fill_constant1p3, remove_fill_constant1p4

# paddle.enable_static()

# def setUp():
#   for lib in os.listdir(os.getenv("CUSTOM_DEVICE_ROOT")):
#     if lib.endswith(".so"):
#       paddle.utils.cpp_extension.extension_utils.load_op_meta_info_and_register_op(
#         lib
#       )