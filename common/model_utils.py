"""
common/model_utils.py
模型底层工具：将外层 model.forward 路由到 model.thinker.forward 的 patch。
被 sft/model.py 和 grpo/trainer.py 共同使用。
"""


def patch_outer_forward(model):
    """
    将 Qwen3ASRForConditionalGeneration 的 forward 路由到 thinker.forward，
    以便 HuggingFace Trainer 和手写训练循环均能正确调用。

    同时为 peft 兼容性代理 get/set_input_embeddings。
    """
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return

    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError(
            "Cannot patch forward: model has no `.thinker.forward`. "
            "Your qwen3_asr model may be incompatible."
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_features=None,
        feature_attention_mask=None,
        labels=None,
        **kwargs,
    ):
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            **kwargs,
        )

    cls.forward = forward

    # peft 内部会调用 get_input_embeddings / set_input_embeddings
    def get_input_embeddings(self):
        return self.thinker.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.thinker.set_input_embeddings(value)

    cls.get_input_embeddings = get_input_embeddings
    cls.set_input_embeddings = set_input_embeddings
    cls._forward_patched = True
