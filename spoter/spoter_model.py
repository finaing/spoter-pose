
import copy
import logging
import types
import torch

import torch.nn as nn
from typing import Optional


def _get_clones(mod, n):
    return nn.ModuleList([copy.deepcopy(mod) for _ in range(n)])


class SPOTERTransformerDecoderLayer(nn.TransformerDecoderLayer):
    """
    Edited TransformerDecoderLayer implementation omitting the redundant self-attention operation as opposed to the
    standard implementation.
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation):
        super(SPOTERTransformerDecoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, activation)

        # Replace self_attn with a lightweight stub rather than deleting it.
        # Newer PyTorch (≥2.0) inspects self.layers[0].self_attn.batch_first
        # inside TransformerDecoder.forward before dispatching to our forward().
        del self.self_attn
        self.self_attn = types.SimpleNamespace(batch_first=False)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None, tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:

        tgt = tgt + self.dropout1(tgt)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class SPOTER(nn.Module):
    """
    Implementation of the SPOTER (Sign POse-based TransformER) architecture for sign language recognition from sequence
    of skeletal data.
    """

    def __init__(self, num_classes, hidden_dim=55):
        super().__init__()

        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim))
        self.pos = nn.Parameter(torch.cat([self.row_embed[0].unsqueeze(0).repeat(1, 1, 1)], dim=-1).flatten(0, 1).unsqueeze(0))
        self.class_query = nn.Parameter(torch.rand(1, hidden_dim))
        self.transformer = nn.Transformer(hidden_dim, 9, 6, 6)
        self.linear_class = nn.Linear(hidden_dim, num_classes)

        # Deactivate the initial attention decoder mechanism
        custom_decoder_layer = SPOTERTransformerDecoderLayer(self.transformer.d_model, self.transformer.nhead, 2048,
                                                             0.1, "relu")
        self.transformer.decoder.layers = _get_clones(custom_decoder_layer, self.transformer.decoder.num_layers)

    def forward(self, inputs):
        h = torch.unsqueeze(inputs.flatten(start_dim=1), 1).float()
        h = self.transformer(self.pos + h, self.class_query.unsqueeze(0)).transpose(0, 1)
        res = self.linear_class(h)

        return res

    # ── Transfer-learning helpers ─────────────────────────────────────────────

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, num_classes: int,
                        freeze_encoder: bool = False) -> "SPOTER":
        """
        Load a pretrained SPOTER checkpoint and prepare it for fine-tuning on a
        new classification task.

        All weights are transferred except the final classification head
        (``linear_class``), which is re-initialised for ``num_classes``.
        The transformer decoder and ``class_query`` are kept trainable so they
        can adapt to the new task even when the encoder is frozen.

        Parameters
        ----------
        checkpoint_path : str
            Path to a checkpoint saved by ``torch.save(model, path)``.
        num_classes : int
            Number of output classes for the new task.
        freeze_encoder : bool
            If True, freeze the transformer encoder and positional parameters
            immediately after loading.  Call :meth:`unfreeze_encoder` later
            (or set ``--freeze_epochs`` in ``train.py``) to resume end-to-end
            fine-tuning.

        Returns
        -------
        SPOTER
            Model ready for fine-tuning.
        """
        pretrained = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Infer hidden_dim from the pretrained head rather than requiring the
        # caller to know it.
        hidden_dim = pretrained.linear_class.in_features
        model = cls(num_classes=num_classes, hidden_dim=hidden_dim)

        # Transfer every parameter except the classification head.
        state = pretrained.state_dict()
        del state["linear_class.weight"]
        del state["linear_class.bias"]
        missing, unexpected = model.load_state_dict(state, strict=False)

        if missing != ["linear_class.weight", "linear_class.bias"]:
            logging.warning("Unexpected missing keys when loading pretrained weights: %s", missing)
        if unexpected:
            logging.warning("Unexpected keys in checkpoint (ignored): %s", unexpected)

        logging.info(
            "Loaded pretrained SPOTER from '%s' (hidden_dim=%d). "
            "Classification head re-initialised for %d classes.",
            checkpoint_path, hidden_dim, num_classes,
        )

        if freeze_encoder:
            model.freeze_encoder()

        return model

    def freeze_encoder(self):
        """
        Freeze the transformer encoder and positional parameters.

        The decoder, ``class_query``, and ``linear_class`` remain trainable so
        the model can adapt to a new task without touching the pretrained
        sequence representations.
        """
        for param in self.transformer.encoder.parameters():
            param.requires_grad = False
        self.row_embed.requires_grad = False
        self.pos.requires_grad = False
        logging.info("Encoder frozen — only decoder, class_query, and linear_class are trainable.")

    def unfreeze_encoder(self):
        """Unfreeze all encoder parameters for end-to-end fine-tuning."""
        for param in self.transformer.encoder.parameters():
            param.requires_grad = True
        self.row_embed.requires_grad = True
        self.pos.requires_grad = True
        logging.info("Encoder unfrozen — all parameters are now trainable.")


if __name__ == "__main__":
    pass
