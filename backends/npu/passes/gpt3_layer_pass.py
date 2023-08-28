from __future__ import print_function, division
import paddle
from paddle.incubate.passes import ir


def transpose_without_shape(x):
    op = ir.PassDesc.OP.transpose2
    op.SetAttr("axis", [0, 2, 1, 3])
    op._outputs.pop("XShape")
    return op(X=x)

def reshape_without_shape(x):
    op = ir.PassDesc.OP.reshape2
    op.SetAttr("shape", [0, 0, -1])
    op._outputs.pop("XShape")
    return op(X=x)

def reshape_without_shape_not_set_attr(x):
    op = ir.PassDesc.OP.reshape2
    op._outputs.pop("XShape")
    return op(X=x)

def layernorm_only_out(x, norm_weight, norm_bias):
    layernorm_op = ir.PassDesc.OP.layer_norm
    layernorm_op._outputs.pop("Mean")
    layernorm_op._outputs.pop("Variance")
    x = layernorm_op(X=x, Scale=norm_weight, Bias=norm_bias)
    return x

def linear_without_params(x):
    x = ir.PassDesc.OP.matmul_v2(X=x)
    return ir.PassDesc.OP.elementwise_add(X=x)

def linear_with_params(x, linear_weight, linear_bias):
    x = ir.PassDesc.OP.matmul_v2(X=x, Y=linear_weight)
    return ir.PassDesc.OP.elementwise_add(X=x, Y=linear_bias)

def linear_with_allreduce(x, linear_weight, linear_bias):
    x = ir.PassDesc.OP.matmul_v2(X=x, Y=linear_weight)
    x = ir.PassDesc.OP.c_allreduce_sum(X=x)
    return ir.PassDesc.OP.elementwise_add(X=x, Y=linear_bias)

def concat_in_axis_1(x, y):
    op = ir.PassDesc.OP.concat
    op.SetAttr("axis", 1)
    return op(X=[x, y])

def split_3(x):
    split_op = ir.PassDesc.OP.split
    split_op.SetAttr("axis", 3)
    split_op.SetAttr("num", 3)
    split_op._outputs = {}
    split_op(X=x._outputs["Out"])
    outs_name = [paddle.fluid.unique_name.generate('split') for i in range(3)]

    split_op._desc.set_output("Out", outs_name)
    block = paddle.static.default_main_program().current_block()
    results = []
    for out in outs_name:
        results.append(block.create_var(name=out))
    return results[0], results[1], results[2]

@ir.RegisterPass
def gen_fuse_attention_layer():
    def pattern(embeddings, norm_weight, norm_bias, mix_linear_weight, mix_linear_bias,
        self_out_linear_weight, self_out_linear_bias, self_out_norm_weight, self_out_norm_bias,
        ffn_linear_weight, ffn_linear_bias, ffn_out_linear_weight, ffn_out_linear_bias, attn_mask):
        # layernorm
        x = layernorm_only_out(embeddings, norm_weight, norm_bias)
        # linear
        x = linear_with_params(x, mix_linear_weight, mix_linear_bias)
        # reshape
        x = reshape_without_shape_not_set_attr(x)
        # split
        q, k, v = split_3(x)

        # # q transpose
        transed_q = transpose_without_shape(q)

        transed_k = transpose_without_shape(k)

        transed_v = transpose_without_shape(v)

        scaled_q = ir.PassDesc.OP.scale(X=transed_q)

        q_mul_k = ir.PassDesc.OP.matmul_v2(X=scaled_q, Y=transed_k)
        q_mul_k.SetAttr("trans_y", True)

        scaled_q_mul_k = ir.PassDesc.OP.scale(X=q_mul_k)

        added_attn_weight = ir.PassDesc.OP.elementwise_add(X=scaled_q_mul_k, Y=attn_mask)

        softmax_attn_weight = ir.PassDesc.OP.softmax(X=added_attn_weight)

        dropout_op = ir.PassDesc.OP.dropout
        dropout_op.SetAttr("dropout_implementation", "upscale_in_train")
        dropout_op._outputs.pop("Mask")
        softmax_attn_weight = dropout_op(X=softmax_attn_weight)

        out = ir.PassDesc.OP.matmul_v2(X=softmax_attn_weight, Y=transed_v)

        out = transpose_without_shape(out)

        out = reshape_without_shape(out)

        out = linear_with_params(out, self_out_linear_weight, self_out_linear_bias)
        
        dropout_op_2 = ir.PassDesc.OP.dropout
        dropout_op_2.SetAttr("dropout_implementation", "upscale_in_train")
        dropout_op_2._outputs.pop("Mask")
        out = dropout_op_2(X=out)

        #resadd
        res_add_out = ir.PassDesc.OP.elementwise_add(X=embeddings, Y=out)
        layer_out = layernorm_only_out(res_add_out, self_out_norm_weight, self_out_norm_bias)

        linear_out = linear_with_params(layer_out, ffn_linear_weight, ffn_linear_bias)

        gelu_out = ir.PassDesc.OP.gelu(X=linear_out)

        linear_2_out = linear_with_params(gelu_out, ffn_out_linear_weight, ffn_out_linear_bias)

        dropout_op_3 = ir.PassDesc.OP.dropout
        dropout_op_3.SetAttr("dropout_implementation", "upscale_in_train")
        dropout_op_3._outputs.pop("Mask")
        drop_3_out = dropout_op_3(X=linear_2_out)

        # # residule
        res_add_out_2 = ir.PassDesc.OP.elementwise_add(X=res_add_out, Y=drop_3_out)

        return res_add_out_2, k, v

    def gpt3_layer_adaptor(embeddings, norm_weight, norm_bias, mix_linear_weight, mix_linear_bias,
        self_out_linear_weight, self_out_linear_bias, self_out_norm_weight, self_out_norm_bias,
        ffn_linear_weight, ffn_linear_bias, ffn_out_linear_weight, ffn_out_linear_bias, attn_mask):
        gpt3_layer_without_kvcache_op = ir.PassDesc.OP.gpt3_layer_without_kvcache
        gpt3_layer_without_kvcache_op._outputs = {}
        gpt3_layer_without_kvcache_op(
            Hidden=embeddings,
            NormWeight=norm_weight,
            NormBias=norm_bias,
            MixLinearWeight=mix_linear_weight,
            MixLinearBias=mix_linear_bias,
            SelfOutLinearWeight=self_out_linear_weight,
            SelfOutLinearBias=self_out_linear_bias,
            SelfOutNormWeight=self_out_norm_weight,
            SelfOutNormBias=self_out_norm_bias,
            FfnLinearWeight=ffn_linear_weight,
            FfnLinearBias=ffn_linear_bias,
            FfnOutLinearWeight=ffn_out_linear_weight,
            FfnOutLinearBias=ffn_out_linear_bias,
            AttentionMask=attn_mask)

        outs_name = [paddle.fluid.unique_name.generate('gpt3_layer_without_kvcache') for i in range(3)] # 3 outputs
        gpt3_layer_without_kvcache_op._desc.set_output("Out", [outs_name[0]])
        gpt3_layer_without_kvcache_op._desc.set_output("PresentKey", [outs_name[1]])
        gpt3_layer_without_kvcache_op._desc.set_output("PresentValue", [outs_name[2]])

        gpt3_layer_without_kvcache_op.Attr("begin_norm_axis").MappedPattern(op="layer_norm", name="begin_norm_axis", index=0)
        gpt3_layer_without_kvcache_op.Attr("epsilon").MappedPattern(op="layer_norm", name="epsilon", index=0)
        gpt3_layer_without_kvcache_op.Attr("shape").MappedPattern(op="reshape2", name="shape", index=0)
        gpt3_layer_without_kvcache_op.Attr("scale").MappedPattern(op="scale", name="scale", index=1)

        block = paddle.static.default_main_program().current_block()
        results = []
        for out in outs_name:
            results.append(block.create_var(name=out))
        return results[0], results[1], results[2]

    def replace(embeddings, norm_weight, norm_bias, mix_linear_weight, mix_linear_bias,
        self_out_linear_weight, self_out_linear_bias, self_out_norm_weight, self_out_norm_bias,
        ffn_linear_weight, ffn_linear_bias, ffn_out_linear_weight, ffn_out_linear_bias, attn_mask):

        out = gpt3_layer_adaptor(embeddings, norm_weight, norm_bias, mix_linear_weight, mix_linear_bias,
            self_out_linear_weight, self_out_linear_bias, self_out_norm_weight, self_out_norm_bias,
            ffn_linear_weight, ffn_linear_bias, ffn_out_linear_weight, ffn_out_linear_bias, attn_mask)

        return out[0], out[1], out[2]

    return pattern, replace

@ir.RegisterPass
def gen_fuse_attention_cached_layer():
    def pattern(embeddings, norm_weight, norm_bias, mix_linear_weight, mix_linear_bias,
    self_out_linear_weight, self_out_linear_bias, self_out_norm_weight, self_out_norm_bias,
    ffn_linear_weight, ffn_linear_bias, ffn_out_linear_weight, ffn_out_linear_bias, attn_mask, past_key, past_value):
        # layernorm
        x = layernorm_only_out(embeddings, norm_weight, norm_bias)
                # linear
        x = linear_with_params(x, mix_linear_weight, mix_linear_bias)
                # reshape
        x = reshape_without_shape_not_set_attr(x)
        # split
        q, k, v = split_3(x)

        # # q transpose
        transed_q = transpose_without_shape(q)
        concated_k = concat_in_axis_1(past_key, k)
        transed_k = transpose_without_shape(concated_k)
        
        concated_v = concat_in_axis_1(past_value, v)
        transed_v = transpose_without_shape(concated_v)

        scaled_q = ir.PassDesc.OP.scale(X=transed_q)

        q_mul_k = ir.PassDesc.OP.matmul_v2(X=scaled_q, Y=transed_k)
        q_mul_k.SetAttr("trans_y", True)

        scaled_q_mul_k = ir.PassDesc.OP.scale(X=q_mul_k)

        added_attn_weight = ir.PassDesc.OP.elementwise_add(X=scaled_q_mul_k, Y=attn_mask)

        softmax_attn_weight = ir.PassDesc.OP.softmax(X=added_attn_weight)

        dropout_op = ir.PassDesc.OP.dropout
        dropout_op.SetAttr("dropout_implementation", "upscale_in_train")
        dropout_op._outputs.pop("Mask")
        softmax_attn_weight = dropout_op(X=softmax_attn_weight)

        out = ir.PassDesc.OP.matmul_v2(X=softmax_attn_weight, Y=transed_v)

        out = transpose_without_shape(out)

        out = reshape_without_shape(out)

        # linear
        out = linear_with_params(out, self_out_linear_weight, self_out_linear_bias)
        # 
        dropout_op_2 = ir.PassDesc.OP.dropout
        dropout_op_2.SetAttr("dropout_implementation", "upscale_in_train")
        dropout_op_2._outputs.pop("Mask")
        out = dropout_op_2(X=out)

        #resadd
        res_add_out = ir.PassDesc.OP.elementwise_add(X=embeddings, Y=out)

        layer_out = layernorm_only_out(res_add_out, self_out_norm_weight, self_out_norm_bias)

        linear_out = linear_with_params(layer_out, ffn_linear_weight, ffn_linear_bias)

        gelu_out = ir.PassDesc.OP.gelu(X=linear_out)

        linear_2_out = linear_with_params(gelu_out, ffn_out_linear_weight, ffn_out_linear_bias)

        dropout_op_3 = ir.PassDesc.OP.dropout
        dropout_op_3.SetAttr("dropout_implementation", "upscale_in_train")
        dropout_op_3._outputs.pop("Mask")
        drop_3_out = dropout_op_3(X=linear_2_out)

        # # residule
        res_add_out_2 = ir.PassDesc.OP.elementwise_add(X=res_add_out, Y=drop_3_out)

        assign_k = ir.PassDesc.OP.assign(X=concated_k)
        assign_v = ir.PassDesc.OP.assign(X=concated_v)
        
        return res_add_out_2, assign_k, assign_v

    def gpt3_layer_cache_adaptor(embeddings, norm_weight, norm_bias, mix_linear_weight, mix_linear_bias,
    self_out_linear_weight, self_out_linear_bias, self_out_norm_weight, self_out_norm_bias,
    ffn_linear_weight, ffn_linear_bias, ffn_out_linear_weight, ffn_out_linear_bias, attn_mask, past_key, past_value):
        gpt3_layer_cache_op = ir.PassDesc.OP.gpt3_layer
        gpt3_layer_cache_op._outputs = {}
        gpt3_layer_cache_op(
            Hidden=embeddings,
            NormWeight=norm_weight,
            NormBias=norm_bias,
            MixLinearWeight=mix_linear_weight,
            MixLinearBias=mix_linear_bias,
            SelfOutLinearWeight=self_out_linear_weight,
            SelfOutLinearBias=self_out_linear_bias,
            SelfOutNormWeight=self_out_norm_weight,
            SelfOutNormBias=self_out_norm_bias,
            FfnLinearWeight=ffn_linear_weight,
            FfnLinearBias=ffn_linear_bias,
            FfnOutLinearWeight=ffn_out_linear_weight,
            FfnOutLinearBias=ffn_out_linear_bias,
            AttentionMask=attn_mask,
            PastKey=past_key,
            PastValue=past_value)

        outs_name = [paddle.fluid.unique_name.generate('gpt3_layer') for i in range(3)] # 3 outputs
        print(outs_name)
        gpt3_layer_cache_op._desc.set_output("Out", [outs_name[0]])
        gpt3_layer_cache_op._desc.set_output("PresentKey", [outs_name[1]])
        gpt3_layer_cache_op._desc.set_output("PresentValue", [outs_name[2]])

        gpt3_layer_cache_op.Attr("begin_norm_axis").MappedPattern(op="layer_norm", name="begin_norm_axis", index=0)
        gpt3_layer_cache_op.Attr("epsilon").MappedPattern(op="layer_norm", name="epsilon", index=0)
        gpt3_layer_cache_op.Attr("shape").MappedPattern(op="reshape2", name="shape", index=0)
        gpt3_layer_cache_op.Attr("scale").MappedPattern(op="scale", name="scale", index=1)
        
        block = paddle.static.default_main_program().current_block()
        results = []
        for out in outs_name:
            results.append(block.create_var(name=out))
        return results[0], results[1], results[2]

    def replace(embeddings, norm_weight, norm_bias, mix_linear_weight, mix_linear_bias,
    self_out_linear_weight, self_out_linear_bias, self_out_norm_weight, self_out_norm_bias,
    ffn_linear_weight, ffn_linear_bias, ffn_out_linear_weight, ffn_out_linear_bias, attn_mask, past_key, past_value):

        out = gpt3_layer_cache_adaptor(embeddings, norm_weight, norm_bias, mix_linear_weight, mix_linear_bias,
            self_out_linear_weight, self_out_linear_bias, self_out_norm_weight, self_out_norm_bias,
            ffn_linear_weight, ffn_linear_bias, ffn_out_linear_weight, ffn_out_linear_bias, attn_mask, past_key, past_value)
    

        return out[0], out[1], out[2]

    return pattern, replace


@ir.RegisterPass
def gen_fuse_attention_parallel_layer():
    def pattern(embeddings, norm_weight, norm_bias, mix_linear_weight, mix_linear_bias,
        self_out_linear_weight, self_out_linear_bias, self_out_norm_weight, self_out_norm_bias,
        ffn_linear_weight, ffn_linear_bias, ffn_out_linear_weight, ffn_out_linear_bias, attn_mask):
        # layernorm
        x = layernorm_only_out(embeddings, norm_weight, norm_bias)
        x = ir.PassDesc.OP.c_identity(X=x)
        # linear
        x = linear_with_params(x, mix_linear_weight, mix_linear_bias)
        # reshape
        x = reshape_without_shape_not_set_attr(x)
        # split
        q, k, v = split_3(x)

        # q k v transpose
        transed_q = transpose_without_shape(q)
        transed_k = transpose_without_shape(k)
        transed_v = transpose_without_shape(v)

        scaled_q = ir.PassDesc.OP.scale(X=transed_q)

        q_mul_k = ir.PassDesc.OP.matmul_v2(X=scaled_q, Y=transed_k)
        q_mul_k.SetAttr("trans_y", True)
        scaled_q_mul_k = ir.PassDesc.OP.scale(X=q_mul_k)
        added_attn_weight = ir.PassDesc.OP.elementwise_add(X=scaled_q_mul_k, Y=attn_mask)

        softmax_attn_weight = ir.PassDesc.OP.softmax(X=added_attn_weight)

        dropout_op = ir.PassDesc.OP.dropout
        dropout_op.SetAttr("dropout_implementation", "upscale_in_train")
        dropout_op._outputs.pop("Mask")
        softmax_attn_weight = dropout_op(X=softmax_attn_weight)

        out = ir.PassDesc.OP.matmul_v2(X=softmax_attn_weight, Y=transed_v)
        out = transpose_without_shape(out)
        out = reshape_without_shape(out)

        out = linear_with_allreduce(out, self_out_linear_weight, self_out_linear_bias)

        dropout_op_2 = ir.PassDesc.OP.dropout
        dropout_op_2.SetAttr("dropout_implementation", "upscale_in_train")
        dropout_op_2._outputs.pop("Mask")
        out = dropout_op_2(X=out)

        #resadd
        res_add_out = ir.PassDesc.OP.elementwise_add(X=embeddings, Y=out)
        layer_out = layernorm_only_out(res_add_out, self_out_norm_weight, self_out_norm_bias)
        layer_out= ir.PassDesc.OP.c_identity(X=layer_out)

        linear_out = linear_with_params(layer_out, ffn_linear_weight, ffn_linear_bias)
        gelu_out = ir.PassDesc.OP.gelu(X=linear_out)

        linear_2_out = linear_with_allreduce(gelu_out, ffn_out_linear_weight, ffn_out_linear_bias)

        dropout_op_3 = ir.PassDesc.OP.dropout
        dropout_op_3.SetAttr("dropout_implementation", "upscale_in_train")
        dropout_op_3._outputs.pop("Mask")
        drop_3_out = dropout_op_3(X=linear_2_out)

        res_add_out_2 = ir.PassDesc.OP.elementwise_add(X=res_add_out, Y=drop_3_out)

        return res_add_out_2, k, v

    def gpt3_layer_adaptor(embeddings, norm_weight, norm_bias, mix_linear_weight, mix_linear_bias,
        self_out_linear_weight, self_out_linear_bias, self_out_norm_weight, self_out_norm_bias,
        ffn_linear_weight, ffn_linear_bias, ffn_out_linear_weight, ffn_out_linear_bias, attn_mask):
        gpt3_layer_without_kvcache_parallel_op = ir.PassDesc.OP.gpt3_layer_without_kvcache_parallel
        gpt3_layer_without_kvcache_parallel_op._outputs = {}
        gpt3_layer_without_kvcache_parallel_op(
            Hidden=embeddings,
            NormWeight=norm_weight,
            NormBias=norm_bias,
            MixLinearWeight=mix_linear_weight,
            MixLinearBias=mix_linear_bias,
            SelfOutLinearWeight=self_out_linear_weight,
            SelfOutLinearBias=self_out_linear_bias,
            SelfOutNormWeight=self_out_norm_weight,
            SelfOutNormBias=self_out_norm_bias,
            FfnLinearWeight=ffn_linear_weight,
            FfnLinearBias=ffn_linear_bias,
            FfnOutLinearWeight=ffn_out_linear_weight,
            FfnOutLinearBias=ffn_out_linear_bias,
            AttentionMask=attn_mask)

        outs_name = [paddle.fluid.unique_name.generate('gpt3_layer_without_kvcache_parallel') for i in range(3)] # 3 outputs
        gpt3_layer_without_kvcache_parallel_op._desc.set_output("Out", [outs_name[0]])
        gpt3_layer_without_kvcache_parallel_op._desc.set_output("PresentKey", [outs_name[1]])
        gpt3_layer_without_kvcache_parallel_op._desc.set_output("PresentValue", [outs_name[2]])

        gpt3_layer_without_kvcache_parallel_op.Attr("begin_norm_axis").MappedPattern(op="layer_norm", name="begin_norm_axis", index=0)
        gpt3_layer_without_kvcache_parallel_op.Attr("epsilon").MappedPattern(op="layer_norm", name="epsilon", index=0)
        gpt3_layer_without_kvcache_parallel_op.Attr("shape").MappedPattern(op="reshape2", name="shape", index=0)
        gpt3_layer_without_kvcache_parallel_op.Attr("scale").MappedPattern(op="scale", name="scale", index=1)

        block = paddle.static.default_main_program().current_block()
        results = []
        for out in outs_name:
            results.append(block.create_var(name=out))
        return results[0], results[1], results[2]

    def replace(embeddings, norm_weight, norm_bias, mix_linear_weight, mix_linear_bias,
        self_out_linear_weight, self_out_linear_bias, self_out_norm_weight, self_out_norm_bias,
        ffn_linear_weight, ffn_linear_bias, ffn_out_linear_weight, ffn_out_linear_bias, attn_mask):

        out = gpt3_layer_adaptor(embeddings, norm_weight, norm_bias, mix_linear_weight, mix_linear_bias,
            self_out_linear_weight, self_out_linear_bias, self_out_norm_weight, self_out_norm_bias,
            ffn_linear_weight, ffn_linear_bias, ffn_out_linear_weight, ffn_out_linear_bias, attn_mask)

        return out[0], out[1], out[2]

    return pattern, replace

@ir.RegisterPass
def gen_fuse_attention_cached_parallel_layer():
    def pattern(embeddings, norm_weight, norm_bias, mix_linear_weight, mix_linear_bias,
    self_out_linear_weight, self_out_linear_bias, self_out_norm_weight, self_out_norm_bias,
    ffn_linear_weight, ffn_linear_bias, ffn_out_linear_weight, ffn_out_linear_bias, attn_mask, past_key, past_value):
        # layernorm
        x = layernorm_only_out(embeddings, norm_weight, norm_bias)
        x = ir.PassDesc.OP.c_identity(X=x)
        # linear
        x = linear_with_params(x, mix_linear_weight, mix_linear_bias)
        # reshape
        x = reshape_without_shape_not_set_attr(x)
        # split
        q, k, v = split_3(x)

        # q k v transpose
        transed_q = transpose_without_shape(q)
        concated_k = concat_in_axis_1(past_key, k)
        transed_k = transpose_without_shape(concated_k)
        
        concated_v = concat_in_axis_1(past_value, v)
        transed_v = transpose_without_shape(concated_v)

        scaled_q = ir.PassDesc.OP.scale(X=transed_q)

        q_mul_k = ir.PassDesc.OP.matmul_v2(X=scaled_q, Y=transed_k)
        q_mul_k.SetAttr("trans_y", True)

        scaled_q_mul_k = ir.PassDesc.OP.scale(X=q_mul_k)

        added_attn_weight = ir.PassDesc.OP.elementwise_add(X=scaled_q_mul_k, Y=attn_mask)

        softmax_attn_weight = ir.PassDesc.OP.softmax(X=added_attn_weight)

        dropout_op = ir.PassDesc.OP.dropout
        dropout_op.SetAttr("dropout_implementation", "upscale_in_train")
        dropout_op._outputs.pop("Mask")
        softmax_attn_weight = dropout_op(X=softmax_attn_weight)

        out = ir.PassDesc.OP.matmul_v2(X=softmax_attn_weight, Y=transed_v)

        out = transpose_without_shape(out)

        out = reshape_without_shape(out)

        # linear
        out = linear_with_allreduce(out, self_out_linear_weight, self_out_linear_bias)
        dropout_op_2 = ir.PassDesc.OP.dropout
        dropout_op_2.SetAttr("dropout_implementation", "upscale_in_train")
        dropout_op_2._outputs.pop("Mask")
        out = dropout_op_2(X=out)

        #resadd
        res_add_out = ir.PassDesc.OP.elementwise_add(X=embeddings, Y=out)

        layer_out = layernorm_only_out(res_add_out, self_out_norm_weight, self_out_norm_bias)
        layer_out = ir.PassDesc.OP.c_identity(X=layer_out)

        linear_out = linear_with_params(layer_out, ffn_linear_weight, ffn_linear_bias)

        gelu_out = ir.PassDesc.OP.gelu(X=linear_out)

        linear_2_out = linear_with_allreduce(gelu_out, ffn_out_linear_weight, ffn_out_linear_bias)

        dropout_op_3 = ir.PassDesc.OP.dropout
        dropout_op_3.SetAttr("dropout_implementation", "upscale_in_train")
        dropout_op_3._outputs.pop("Mask")
        drop_3_out = dropout_op_3(X=linear_2_out)

        res_add_out_2 = ir.PassDesc.OP.elementwise_add(X=res_add_out, Y=drop_3_out)

        assign_k = ir.PassDesc.OP.assign(X=concated_k)
        assign_v = ir.PassDesc.OP.assign(X=concated_v)
        
        return res_add_out_2, assign_k, assign_v

    def gpt3_layer_cache_adaptor(embeddings, norm_weight, norm_bias, mix_linear_weight, mix_linear_bias,
    self_out_linear_weight, self_out_linear_bias, self_out_norm_weight, self_out_norm_bias,
    ffn_linear_weight, ffn_linear_bias, ffn_out_linear_weight, ffn_out_linear_bias, attn_mask, past_key, past_value):
        gpt3_layer_parallel_op = ir.PassDesc.OP.gpt3_layer_parallel_async
        gpt3_layer_parallel_op._outputs = {}
        gpt3_layer_parallel_op(
            Hidden=embeddings,
            NormWeight=norm_weight,
            NormBias=norm_bias,
            MixLinearWeight=mix_linear_weight,
            MixLinearBias=mix_linear_bias,
            SelfOutLinearWeight=self_out_linear_weight,
            SelfOutLinearBias=self_out_linear_bias,
            SelfOutNormWeight=self_out_norm_weight,
            SelfOutNormBias=self_out_norm_bias,
            FfnLinearWeight=ffn_linear_weight,
            FfnLinearBias=ffn_linear_bias,
            FfnOutLinearWeight=ffn_out_linear_weight,
            FfnOutLinearBias=ffn_out_linear_bias,
            AttentionMask=attn_mask,
            PastKey=past_key,
            PastValue=past_value)

        outs_name = [paddle.fluid.unique_name.generate('gpt3_layer_parallel_async') for i in range(3)] # 3 outputs
        print(outs_name)
        gpt3_layer_parallel_op._desc.set_output("Out", [outs_name[0]])
        gpt3_layer_parallel_op._desc.set_output("PresentKey", [outs_name[1]])
        gpt3_layer_parallel_op._desc.set_output("PresentValue", [outs_name[2]])

        gpt3_layer_parallel_op.Attr("begin_norm_axis").MappedPattern(op="layer_norm", name="begin_norm_axis", index=0)
        gpt3_layer_parallel_op.Attr("epsilon").MappedPattern(op="layer_norm", name="epsilon", index=0)
        gpt3_layer_parallel_op.Attr("shape").MappedPattern(op="reshape2", name="shape", index=0)
        gpt3_layer_parallel_op.Attr("scale").MappedPattern(op="scale", name="scale", index=1)
        
        block = paddle.static.default_main_program().current_block()
        results = []
        for out in outs_name:
            results.append(block.create_var(name=out))
        return results[0], results[1], results[2]

    def replace(embeddings, norm_weight, norm_bias, mix_linear_weight, mix_linear_bias,
      self_out_linear_weight, self_out_linear_bias, self_out_norm_weight, self_out_norm_bias,
      ffn_linear_weight, ffn_linear_bias, ffn_out_linear_weight, ffn_out_linear_bias, attn_mask, past_key, past_value):

        out = gpt3_layer_cache_adaptor(embeddings, norm_weight, norm_bias, mix_linear_weight, mix_linear_bias,
            self_out_linear_weight, self_out_linear_bias, self_out_norm_weight, self_out_norm_bias,
            ffn_linear_weight, ffn_linear_bias, ffn_out_linear_weight, ffn_out_linear_bias, attn_mask, past_key, past_value)

        return out[0], out[1], out[2]

    return pattern, replace
