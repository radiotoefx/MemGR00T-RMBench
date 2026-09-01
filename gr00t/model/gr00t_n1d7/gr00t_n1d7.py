# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Any, Tuple

import torch
from torch import nn
from torch.distributions import Beta
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
import tree

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.modules.dit import AlternateVLDiT, DiT, SelfAttentionTransformer
from gr00t.model.modules.embodiment_conditioned_mlp import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)


logger = logging.getLogger(__name__)


class Gr00tN1d7ActionHead(nn.Module):
    """Action head component for flow matching diffusion policy."""

    supports_gradient_checkpointing = True

    def __init__(self, config: Gr00tN1d7Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim
        ttt_config = {
            "enabled": config.use_ttt,
            "num_layers": config.ttt_num_layers,
            "layer_indices": config.ttt_layer_indices,
            "dim": config.ttt_dim,
            "hidden_dim": config.ttt_hidden_dim,
            "base_lr": config.ttt_base_lr,
            "gate_init": config.ttt_gate_init,
        }

        if config.use_alternate_vl_dit:
            self.model = AlternateVLDiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
                attend_text_every_n_blocks=config.attend_text_every_n_blocks,
                ttt_config=ttt_config,
            )
            logger.info("Using AlternateVLDiT for diffusion model")
        else:
            self.model = DiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
                ttt_config=ttt_config,
            )
            logger.info("Using DiT for diffusion model")
        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps
        self.register_tokens = (
            nn.Parameter(torch.empty(1, config.ttt_num_register_tokens, self.input_embedding_dim))
            if config.use_ttt and config.ttt_num_register_tokens > 0
            else None
        )
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, mean=0.0, std=0.02)

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim * config.state_history_length,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )

        vl_self_attention_cfg = getattr(config, "vl_self_attention_cfg", None)
        if vl_self_attention_cfg and vl_self_attention_cfg.get("num_layers", 0) > 0:
            self.vl_self_attention = SelfAttentionTransformer(**vl_self_attention_cfg)
        else:
            self.vl_self_attention = nn.Identity()

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # State dropout parameters
        self.state_dropout_prob = config.state_dropout_prob

        # Pin the time-sampling Beta to CPU/fp32 explicitly. The action head can
        # be instantiated under a meta / no_init_weights default-device context
        # (e.g. nested from_pretrained). A Beta built from bare Python floats
        # would then place its concentration tensors on the meta device (or in
        # the active default dtype, e.g. bf16). With validate_args enabled that
        # already fails here in __init__ (Beta's internal .item() check cannot
        # run on meta); even with validation off, sample_time would later raise
        # or return garbage. Explicit device/dtype here makes the sampler depend
        # only on the config, not on the construction-time device/dtype context,
        # so the noise schedule is identical across SDPA/FA2/FA4 and meta vs.
        # real-device loads. config is the canonical source for these values.
        self.beta_dist = Beta(
            torch.tensor(float(config.noise_beta_alpha), dtype=torch.float32, device="cpu"),
            torch.tensor(float(config.noise_beta_beta), dtype=torch.float32, device="cpu"),
        )
        self.num_timestep_buckets = config.num_timestep_buckets
        self.set_trainable_parameters(
            config.tune_projector, config.tune_diffusion_model, config.tune_vlln, config.tune_ttt
        )

    def set_trainable_parameters(
        self,
        tune_projector: bool,
        tune_diffusion_model: bool,
        tune_vlln: bool,
        tune_ttt: bool = True,
    ):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_vlln = tune_vlln
        self.tune_ttt = tune_ttt
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        # RoboTTT pretraining freezes the base DiT while training only the new
        # sequence-modeling layers. Re-enable them after the broad DiT freeze.
        if self.config.use_ttt:
            for layer in self.model.ttt_layers():
                layer.requires_grad_(tune_ttt)
            if self.register_tokens is not None:
                self.register_tokens.requires_grad_(tune_ttt)
        if not tune_vlln:
            self.vlln.requires_grad_(False)
            self.vl_self_attention.requires_grad_(False)
        logger.debug(f"Tune action head projector: {self.tune_projector}")
        logger.debug(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        logger.debug(f"Tune action head vlln: {self.tune_vlln}")
        logger.debug(f"Tune action head TTT: {self.tune_ttt}")
        # Check if any parameters are still trainable. If not, log a warning.
        if not tune_projector and not tune_diffusion_model and not tune_vlln:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    logger.debug(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            logger.warning("No action head trainable parameters found.")

    def copy_embodiment_projectors_(self, source_id: int, target_id: int) -> None:
        """Warm-start the target state/action projectors from a pretrained slot.

        The shared DiT and VLM are intentionally untouched. The copy happens only
        at training startup; checkpoints continue to address the independent
        target slot and therefore do not depend on an inference-time ID override.
        """
        if source_id == target_id:
            raise ValueError("Projector source and target embodiment IDs must differ")
        self.state_encoder.copy_category_(source_id, target_id)
        self.action_encoder.copy_category_(source_id, target_id)
        self.action_decoder.copy_category_(source_id, target_id)

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()
            if not self.tune_vlln:
                self.vlln.eval()
                self.vl_self_attention.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        sample = (1 - sample) * self.config.noise_s
        return sample

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def reset_ttt_state(self, batch_size: int | None = None) -> None:
        self.model.reset_ttt_state(batch_size)

    def initialize_ttt_parameters(self) -> None:
        """Initialize TTT additions after loading a Vanilla checkpoint."""
        for layer in self.model.ttt_layers():
            layer.reset_parameters()
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, mean=0.0, std=0.02)

    def detach_ttt_state(self) -> None:
        self.model.detach_ttt_state()

    def ttt_diagnostics(self) -> list[dict[str, float | int]]:
        return self.model.ttt_diagnostics()

    def _join_state_action_tokens(
        self, state_features: torch.Tensor, action_features: torch.Tensor
    ) -> torch.Tensor:
        tokens = [state_features, action_features]
        if self.register_tokens is not None:
            registers = self.register_tokens.expand(state_features.shape[0], -1, -1)
            tokens.insert(0, registers)
        return torch.cat(tokens, dim=1)

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """
        Forward pass through the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - action: [B, action_horizon, action_dim] (during training)
                - embodiment_id: [B] (embodiment IDs)
                - action_mask: [B, action_horizon, action_dim]

        Returns:
            BatchFeature containing:
                - loss: action prediction loss
        """
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        if self.config.use_ttt and not self.config.ttt_sequence_training:
            self.reset_ttt_state(action_input.state.shape[0])

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Handle state history
        assert action_input.state.shape[1] == self.config.state_history_length
        action_input.state = action_input.state.view(action_input.state.shape[0], 1, -1)

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Dropout state features (training only): zero out dropped states.
        if self.training and self.state_dropout_prob > 0:
            do_dropout = (
                torch.rand(state_features.shape[0], device=state_features.device)
                < self.state_dropout_prob
            )
            do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
            state_features = state_features * (1 - do_dropout)

        # Embed noised action trajectory.
        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        sa_embs = self._join_state_action_tokens(state_features, action_features)
        vl_attn_mask = backbone_output.backbone_attention_mask

        if self.config.use_alternate_vl_dit:
            image_mask = backbone_output.image_mask
            backbone_attention_mask = backbone_output.backbone_attention_mask
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
            )
        else:
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
            )

        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        action_mask = action_input.action_mask
        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = action_loss.sum() / (action_mask.sum() + 1e-6)

        return {
            "loss": loss,
            "action_loss": action_loss,
            "action_mask": action_mask,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }

    def forward_robottt_sequence_prefix_batched(
        self,
        backbone_outputs: list[BatchFeature],
        action_inputs: list[BatchFeature],
        chunk_size: int,
    ) -> list[BatchFeature]:
        """Batch frozen DiT blocks before the first TTT layer across time.

        Only the suffix beginning at the first recurrent TTT block is executed
        timestep-by-timestep. This preserves fast-state ordering while avoiding
        repeated single-item execution of the frozen DiT prefix.
        """
        if not isinstance(self.model, AlternateVLDiT):
            raise ValueError("TTT action-prefix batching currently requires AlternateVLDiT")
        if self.tune_projector or self.tune_diffusion_model or self.tune_vlln:
            raise ValueError("TTT action-prefix batching requires a fully frozen base action head")
        if self.register_tokens is not None and self.register_tokens.requires_grad:
            raise ValueError("TTT action-prefix batching does not support trainable register tokens")
        if len(backbone_outputs) != len(action_inputs) or not action_inputs:
            raise ValueError("Backbone and action sequence lengths must match and be non-empty")

        ttt_indices = [
            index
            for index, block in enumerate(self.model.transformer_blocks)
            if block.ttt is not None
        ]
        if not ttt_indices:
            raise ValueError("TTT action-prefix batching requires at least one TTT layer")
        first_ttt_index = min(ttt_indices)
        self.set_frozen_modules_to_eval_mode()

        prepared = []
        # Preserve the legacy per-timestep RNG order: state dropout, flow noise,
        # then flow time are sampled one timestep at a time.
        with torch.no_grad():
            for backbone_output, action_input in zip(backbone_outputs, action_inputs):
                backbone_output = self.process_backbone_output(backbone_output)
                vl_embeds = backbone_output.backbone_features
                embodiment_id = action_input.embodiment_id
                assert action_input.state.shape[1] == self.config.state_history_length
                state = action_input.state.view(action_input.state.shape[0], 1, -1)
                state_features = self.state_encoder(state, embodiment_id)
                if self.training and self.state_dropout_prob > 0:
                    do_dropout = (
                        torch.rand(state_features.shape[0], device=state_features.device)
                        < self.state_dropout_prob
                    )
                    state_features = state_features * (
                        1 - do_dropout[:, None, None].to(dtype=state_features.dtype)
                    )

                actions = action_input.action
                noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
                flow_time = self.sample_time(
                    actions.shape[0], device=actions.device, dtype=actions.dtype
                )[:, None, None]
                noisy_trajectory = (1 - flow_time) * noise + flow_time * actions
                velocity = actions - noise
                timestep = (flow_time[:, 0, 0] * self.num_timestep_buckets).long()
                action_features = self.action_encoder(
                    noisy_trajectory, timestep, embodiment_id
                )
                if self.config.add_pos_embed:
                    pos_ids = torch.arange(
                        action_features.shape[1], dtype=torch.long, device=vl_embeds.device
                    )
                    action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)
                prepared.append(
                    {
                        "hidden_states": self._join_state_action_tokens(
                            state_features, action_features
                        ),
                        "encoder_hidden_states": vl_embeds,
                        "image_mask": backbone_output.image_mask,
                        "backbone_attention_mask": backbone_output.backbone_attention_mask,
                        "timestep": timestep,
                        "velocity": velocity,
                        "action_mask": action_input.action_mask,
                        "actions": actions,
                        "embodiment_id": embodiment_id,
                        "backbone_features": vl_embeds,
                        "state_features": state_features,
                    }
                )

        prefix_hidden_states: list[torch.Tensor] = []
        prefix_temb: list[torch.Tensor] = []
        for start in range(0, len(prepared), chunk_size):
            chunk = prepared[start : start + chunk_size]
            batch_size = chunk[0]["hidden_states"].shape[0]
            with torch.no_grad():
                hidden_states = torch.cat(
                    [item["hidden_states"] for item in chunk], dim=0
                ).contiguous()
                encoder_hidden_states = torch.cat(
                    [item["encoder_hidden_states"] for item in chunk], dim=0
                ).contiguous()
                image_mask = torch.cat([item["image_mask"] for item in chunk], dim=0)
                backbone_attention_mask = torch.cat(
                    [item["backbone_attention_mask"] for item in chunk], dim=0
                )
                timestep = torch.cat([item["timestep"] for item in chunk], dim=0)
                temb = self.model.timestep_encoder(timestep)
                image_attention_mask = image_mask & backbone_attention_mask
                non_image_attention_mask = (~image_mask) & backbone_attention_mask

                for index, block in enumerate(
                    self.model.transformer_blocks[:first_ttt_index]
                ):
                    if index % 2 == 1:
                        hidden_states = block(
                            hidden_states,
                            attention_mask=None,
                            encoder_hidden_states=None,
                            encoder_attention_mask=None,
                            temb=temb,
                            ttt_update_enabled=False,
                        )
                    else:
                        curr_mask = (
                            non_image_attention_mask
                            if index % (2 * self.model.attend_text_every_n_blocks) == 0
                            else image_attention_mask
                        )
                        hidden_states = block(
                            hidden_states,
                            attention_mask=None,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_attention_mask=curr_mask,
                            temb=temb,
                            ttt_update_enabled=False,
                        )

            for index in range(len(chunk)):
                item_start = index * batch_size
                item_end = item_start + batch_size
                prefix_hidden_states.append(hidden_states[item_start:item_end])
                prefix_temb.append(temb[item_start:item_end])

        def run_ttt_layer_over_time(
            index: int, hidden_states_by_time: list[torch.Tensor]
        ) -> list[torch.Tensor]:
            """Preserve the recurrent update order for one TTT layer."""
            block = self.model.transformer_blocks[index]
            outputs_by_time = []
            for item, hidden_states, temb in zip(
                prepared, hidden_states_by_time, prefix_temb
            ):
                image_attention_mask = (
                    item["image_mask"] & item["backbone_attention_mask"]
                )
                non_image_attention_mask = (
                    ~item["image_mask"]
                ) & item["backbone_attention_mask"]
                if index % 2 == 1:
                    hidden_states = block(
                        hidden_states,
                        attention_mask=None,
                        encoder_hidden_states=None,
                        encoder_attention_mask=None,
                        temb=temb,
                        ttt_update_enabled=True,
                    )
                else:
                    curr_mask = (
                        non_image_attention_mask
                        if index % (2 * self.model.attend_text_every_n_blocks) == 0
                        else image_attention_mask
                    )
                    hidden_states = block(
                        hidden_states,
                        attention_mask=None,
                        encoder_hidden_states=item["encoder_hidden_states"],
                        encoder_attention_mask=curr_mask,
                        temb=temb,
                        ttt_update_enabled=True,
                    )
                outputs_by_time.append(hidden_states)
            return outputs_by_time

        def run_frozen_span_batched(
            start_index: int,
            end_index: int,
            hidden_states_by_time: list[torch.Tensor],
        ) -> list[torch.Tensor]:
            """Batch a deterministic frozen span between recurrent TTT layers.

            Autograd intentionally remains enabled: although the span's own
            parameters are frozen, gradients must cross it to earlier TTT
            layers in the same TBPTT segment.
            """
            if start_index >= end_index:
                return hidden_states_by_time
            outputs_by_time = []
            for start in range(0, len(prepared), chunk_size):
                chunk = prepared[start : start + chunk_size]
                hidden_chunk = hidden_states_by_time[start : start + chunk_size]
                batch_size = hidden_chunk[0].shape[0]
                hidden_states = torch.cat(hidden_chunk, dim=0).contiguous()
                temb = torch.cat(prefix_temb[start : start + len(chunk)], dim=0)
                encoder_hidden_states = torch.cat(
                    [item["encoder_hidden_states"] for item in chunk], dim=0
                ).contiguous()
                image_mask = torch.cat([item["image_mask"] for item in chunk], dim=0)
                backbone_attention_mask = torch.cat(
                    [item["backbone_attention_mask"] for item in chunk], dim=0
                )
                image_attention_mask = image_mask & backbone_attention_mask
                non_image_attention_mask = (~image_mask) & backbone_attention_mask
                for index in range(start_index, end_index):
                    block = self.model.transformer_blocks[index]
                    if block.ttt is not None:
                        raise RuntimeError(
                            f"TTT layer {index} cannot be executed in a frozen batched span"
                        )
                    if index % 2 == 1:
                        hidden_states = block(
                            hidden_states,
                            attention_mask=None,
                            encoder_hidden_states=None,
                            encoder_attention_mask=None,
                            temb=temb,
                            ttt_update_enabled=False,
                        )
                    else:
                        curr_mask = (
                            non_image_attention_mask
                            if index % (2 * self.model.attend_text_every_n_blocks) == 0
                            else image_attention_mask
                        )
                        hidden_states = block(
                            hidden_states,
                            attention_mask=None,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_attention_mask=curr_mask,
                            temb=temb,
                            ttt_update_enabled=False,
                        )
                for offset in range(len(chunk)):
                    item_start = offset * batch_size
                    item_end = item_start + batch_size
                    outputs_by_time.append(hidden_states[item_start:item_end])
            return outputs_by_time

        # The recurrent dependencies form one independent chain per TTT layer.
        # Process each such chain in time order, while batching deterministic
        # frozen spans between chains across time. This is algebraically the
        # same dependency graph as timestep-major execution.
        hidden_states_by_time = prefix_hidden_states
        for position, ttt_index in enumerate(ttt_indices):
            hidden_states_by_time = run_ttt_layer_over_time(
                ttt_index, hidden_states_by_time
            )
            next_ttt_index = (
                ttt_indices[position + 1]
                if position + 1 < len(ttt_indices)
                else len(self.model.transformer_blocks)
            )
            hidden_states_by_time = run_frozen_span_batched(
                ttt_index + 1, next_ttt_index, hidden_states_by_time
            )

        outputs = []
        for item, hidden_states, temb in zip(
            prepared, hidden_states_by_time, prefix_temb
        ):
            shift, scale = self.model.proj_out_1(F.silu(temb)).chunk(2, dim=1)
            model_output = self.model.norm_out(hidden_states) * (1 + scale[:, None])
            model_output = self.model.proj_out_2(model_output + shift[:, None])
            pred = self.action_decoder(model_output, item["embodiment_id"])
            pred_actions = pred[:, -item["actions"].shape[1] :]
            action_loss = (
                F.mse_loss(pred_actions, item["velocity"], reduction="none")
                * item["action_mask"]
            )
            loss = action_loss.sum() / (item["action_mask"].sum() + 1e-6)
            outputs.append(
                BatchFeature(
                    data={
                        "loss": loss,
                        "action_loss": action_loss,
                        "action_mask": item["action_mask"],
                        "backbone_features": item["backbone_features"],
                        "state_features": item["state_features"],
                    }
                )
            )
        return outputs

    def _encode_features(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        """
        Encode features for the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_history_length, max_state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - state_features: [B, 1, input_embedding_dim]
        """
        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Handle state history: if we have fewer timesteps than expected, repeat to fill
        state = action_input.state
        current_T = state.shape[1]
        assert current_T == self.config.state_history_length, "current_T != state_history_length"
        # Reshape state from [B, state_history_length, max_state_dim] to [B, 1, state_history_length * max_state_dim]
        state = state.view(state.shape[0], 1, -1)

        # Embed state.
        state_features = self.state_encoder(state, embodiment_id)

        return BatchFeature(data={"backbone_features": vl_embeds, "state_features": state_features})

    @torch.no_grad()
    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_features: [B, seq_len, backbone_embedding_dim]
            state_features: [B, state_horizon, input_embedding_dim]
            embodiment_id: [B] (embodiment IDs)
            backbone_output: Output from the backbone model
        """
        vl_embeds = backbone_features

        # Set initial actions from an explicit episode object or a policy-private
        # generator. Neither path depends on the process-global Torch RNG.
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        expected_noise_shape = (batch_size, self.config.action_horizon, self.action_dim)
        initial_noise = None if options is None else options.get("initial_noise")
        if initial_noise is not None:
            if not isinstance(initial_noise, torch.Tensor):
                raise TypeError("initial_noise must be a torch.Tensor")
            if tuple(initial_noise.shape) != expected_noise_shape:
                raise ValueError(
                    f"initial_noise shape {tuple(initial_noise.shape)} does not match "
                    f"{expected_noise_shape}"
                )
            actions = initial_noise.to(device=device, dtype=vl_embeds.dtype).clone()
        else:
            generator = None if options is None else options.get("generator")
            if generator is not None and not isinstance(generator, torch.Generator):
                raise TypeError("generator must be a torch.Generator")
            actions = torch.randn(
                size=expected_noise_shape,
                dtype=vl_embeds.dtype,
                device=device,
                generator=generator,
            )
        sampled_initial_noise = actions.detach().clone()

        dt = 1.0 / self.num_inference_timesteps
        vel_strength = torch.ones_like(actions)

        if "action" in action_input:
            # If action in input when doing get action, it means we want to use RTC.
            # action_horizon is the action horizon of the input action.
            # rtc_overlap_steps is the number of steps to overlap with the previous action chunks.
            # rtc_frozen_steps is the number of steps to freeze the action, which is the latency of the policy inference.
            # rtc_ramp_rate is the rate of the ramp of denoising the actions.
            assert options is not None, "options is not None"
            assert "action_horizon" in options, "action_horizon is not in options"
            assert "rtc_overlap_steps" in options, "rtc_overlap_steps is not in options"
            assert "rtc_frozen_steps" in options, "rtc_frozen_steps is not in options"
            assert "rtc_ramp_rate" in options, "rtc_ramp_rate is not in options"

            action_horizon_before_padding = options["action_horizon"]

            # Use previous action instead of pure noise to do inpainting
            actions[:, : options["rtc_overlap_steps"], :] = action_input["action"][
                :,
                action_horizon_before_padding
                - options["rtc_overlap_steps"] : action_horizon_before_padding,
                :,
            ]
            vel_strength[:, : options["rtc_frozen_steps"], :] = 0.0
            # NOTE: use an exponential ramp strength to set the remaining unfrozen rtc_steps
            intermediate_steps = options["rtc_overlap_steps"] - options["rtc_frozen_steps"]
            # Create exponential ramp from 0 to 1 over intermediate steps
            t = torch.linspace(0.0, 1.0, intermediate_steps + 2, device=device)
            ramp = 1 - torch.exp(-options["rtc_ramp_rate"] * t)
            ramp = ramp / ramp[-1].clamp_min(1e-8)  # normalize to [0,1]
            ramp = ramp[
                1:-1
            ]  # we will only take the middle part of the ramp, ignore the 0.0 and 1.0
            # Apply ramp to the intermediate steps [batch, intermediate_steps, action_dim]
            vel_strength[
                :,
                options["rtc_frozen_steps"] : options["rtc_overlap_steps"],
                :,
            ] = ramp[None, :, None].to(device)

        # Run denoising steps.
        ttt_state_update_only = bool(
            options is not None and options.get("ttt_state_update_only", False)
        )
        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # Add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            sa_embs = self._join_state_action_tokens(state_features, action_features)

            # Run model forward.
            if self.config.use_alternate_vl_dit:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                    ttt_update_enabled=(
                        self.config.ttt_update_during_rollout and t == 0
                    ),
                )
            else:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    ttt_update_enabled=(
                        self.config.ttt_update_during_rollout and t == 0
                    ),
                )
            if ttt_state_update_only:
                # Offline rollout-state priming only needs the first denoising
                # forward: TTT writes are enabled at t=0 and disabled for every
                # later denoising step. The predicted action is deliberately
                # unused by that probe, so the remaining deterministic steps
                # cannot affect episode-local fast weights.
                return BatchFeature(
                    data={
                        "action_pred": actions,
                        "initial_noise": sampled_initial_noise,
                        "backbone_features": vl_embeds,
                        "state_features": state_features,
                    }
                )
            pred = self.action_decoder(model_output, embodiment_id)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity * vel_strength

        return BatchFeature(
            data={
                "action_pred": actions,
                "initial_noise": sampled_initial_noise,
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )

    @torch.no_grad()
    def get_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - action_pred: [B, action_horizon, action_dim] predicted actions
        """
        features = self._encode_features(backbone_output, action_input)
        return self.get_action_with_features(
            backbone_features=features.backbone_features,
            state_features=features.state_features,
            embodiment_id=action_input.embodiment_id,
            backbone_output=backbone_output,
            action_input=action_input,
            options=options,
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def prepare_input(self, batch: dict) -> BatchFeature:
        """Prepare input batch for the action head."""
        return BatchFeature(data=batch)


def get_backbone_cls(config: Gr00tN1d7Config):
    if "nvidia/Cosmos-Reason2" in config.model_name or "Qwen/Qwen3-VL" in config.model_name:
        # We import here as Qwen3Backbone depends on newer transformers versions than the rest of the code.
        from gr00t.model.modules.qwen3_backbone import Qwen3Backbone

        return Qwen3Backbone
    else:
        raise ValueError(f"Unsupported model name: {config.model_name}")


class Gr00tN1d7(PreTrainedModel):
    """Gr00tN1d7: VLA model with Cosmos-Reason2-2B (Qwen3-VL) backbone."""

    config_class = Gr00tN1d7Config
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Gr00tN1d7Config,
        transformers_loading_kwargs: dict = {"trust_remote_code": True},
    ):
        """
        Initialize Gr00tN1d7 model.

        Args:
            config: Model configuration
            transformers_loading_kwargs: Dict with transformers loading parameters:
                - transformers_trust_remote_code: Whether to trust remote code when loading from HF Hub
                - transformers_local_files_only: Whether to only use local files
                - model_revision: Specific model revision to use
                - transformers_cache_dir: Directory to cache downloaded models
                - transformers_access_token: HuggingFace access token for gated models

        Note: During training, transformers parameters are passed from training config.
              During inference (e.g., from_pretrained), defaults are used.
        """
        super().__init__(config)
        self.config = config

        backbone_cls = get_backbone_cls(config)
        self.backbone = backbone_cls(
            model_name=config.model_name,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.select_layer,
            reproject_vision=config.reproject_vision,
            use_flash_attention=config.use_flash_attention,
            load_bf16=config.load_bf16,
            tune_top_llm_layers=config.tune_top_llm_layers,
            trainable_params_fp32=config.backbone_trainable_params_fp32,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

        # Initialize action head
        self.action_head = Gr00tN1d7ActionHead(config)
        from .processing_gr00t_n1d7 import Gr00tN1d7DataCollator

        self.collator = Gr00tN1d7DataCollator(
            model_name=config.model_name,
            model_type=config.backbone_model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

    def prepare_input(self, inputs: dict) -> Tuple[BatchFeature, BatchFeature]:
        """Prepare inputs for backbone and action head."""

        # NOTE -- currently the eval code doesn't use collator, so we need to add it here
        # this should ideally be fixed upstream
        if "vlm_content" in inputs:
            # Fix for n_envs > 1: Process all environments' VLM content, not just the first
            vlm_content_list = inputs["vlm_content"]
            # Ensure vlm_content_list is always a list for consistent processing
            if not isinstance(vlm_content_list, list):
                vlm_content_list = [vlm_content_list]

            # Process all VLM contents through the collator
            prep = self.collator([{"vlm_content": vlm} for vlm in vlm_content_list])["inputs"]
            inputs.pop("vlm_content")
            inputs.update(prep)

        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        # Move to device and dtype
        def to_device_with_dtype(x):
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.dtype)
            else:
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_dtype, action_inputs)

        return backbone_inputs, action_inputs

    def forward(self, inputs: dict) -> BatchFeature:
        """
        Forward pass through the complete model.

        Args:
            inputs: Dictionary containing:
                - Action inputs (state, action, embodiment_id, etc.)

        Returns:
            BatchFeature containing loss and other outputs
        """
        if "robottt_sequence" in inputs:
            return self.forward_robottt_sequence(
                inputs["robottt_sequence"],
                reset_state=inputs.get("robottt_reset_state", True),
                timestep_offset=inputs.get("robottt_timestep_offset", 0),
                total_sequence_length=inputs.get("robottt_total_sequence_length"),
            )

        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head(backbone_outputs, action_inputs)

        return action_outputs

    def forward_robottt_sequence(
        self,
        timestep_inputs: list[dict],
        reset_state: bool = True,
        timestep_offset: int = 0,
        total_sequence_length: int | None = None,
    ) -> BatchFeature:
        """Train recurrent TTT memory on one ordered window per batch item.

        The collator transposes a batch of episode windows into a list of
        timestep batches. Fast weights are reset once at the window boundary,
        carried between timesteps, and detached (but not reset) at TBPTT
        boundaries. Each action-head call independently samples flow time/noise,
        matching sequence action forcing.
        """
        if not self.config.use_ttt or not self.config.ttt_sequence_training:
            raise ValueError(
                "robottt_sequence input requires use_ttt=True and ttt_sequence_training=True"
            )
        if not timestep_inputs:
            raise ValueError("robottt_sequence cannot be empty")

        if reset_state:
            self.reset_ttt_state()
        else:
            # The previous segment has already been backpropagated. Preserve
            # fast-weight values while cutting the graph at this boundary.
            self.detach_ttt_state()
        prepared_inputs = [self.prepare_input(step_inputs) for step_inputs in timestep_inputs]
        backbone_chunk_size = int(getattr(self.config, "ttt_backbone_chunk_size", 1))
        if backbone_chunk_size > 1:
            if any(parameter.requires_grad for parameter in self.backbone.parameters()):
                raise ValueError("TTT backbone chunking requires a fully frozen backbone")
            backbone_outputs = []
            first_action_input = prepared_inputs[0][1]
            batch_tensor = next(
                (value for value in first_action_input.values() if isinstance(value, torch.Tensor)),
                None,
            )
            if batch_tensor is None:
                raise ValueError("Cannot infer sequence batch size from action inputs")
            batch_size = batch_tensor.shape[0]
            for start in range(0, len(prepared_inputs), backbone_chunk_size):
                chunk = prepared_inputs[start : start + backbone_chunk_size]
                merged_backbone_inputs = BatchFeature(
                    data={
                        key: torch.cat([item[0][key] for item in chunk], dim=0)
                        for key in chunk[0][0]
                    }
                )
                with torch.no_grad():
                    merged_output = self.backbone(merged_backbone_inputs)
                for index in range(len(chunk)):
                    item_start = index * batch_size
                    item_end = item_start + batch_size
                    backbone_outputs.append(
                        BatchFeature(
                            data={
                                key: value[item_start:item_end]
                                for key, value in merged_output.items()
                            }
                        )
                    )
        else:
            backbone_outputs = [self.backbone(item[0]) for item in prepared_inputs]

        timestep_losses = []
        action_losses = []
        action_masks = []
        segment_length = self.config.ttt_tbptt_segment_length

        action_prefix_chunk_size = int(
            getattr(self.config, "ttt_action_prefix_chunk_size", 1)
        )
        timestep_outputs = []
        if action_prefix_chunk_size > 1:
            for start in range(0, len(prepared_inputs), segment_length):
                if start > 0:
                    self.detach_ttt_state()
                timestep_outputs.extend(
                    self.action_head.forward_robottt_sequence_prefix_batched(
                        backbone_outputs[start : start + segment_length],
                        [item[1] for item in prepared_inputs[start : start + segment_length]],
                        chunk_size=action_prefix_chunk_size,
                    )
                )
        else:
            for timestep, ((_, action_inputs), backbone_output) in enumerate(
                zip(prepared_inputs, backbone_outputs)
            ):
                if timestep > 0 and timestep % segment_length == 0:
                    self.detach_ttt_state()
                timestep_outputs.append(self.action_head(backbone_output, action_inputs))

        for outputs in timestep_outputs:
            timestep_losses.append(outputs["loss"])
            action_losses.append(outputs["action_loss"])
            action_masks.append(outputs["action_mask"])

        total_sequence_length = total_sequence_length or len(timestep_inputs)
        decision_start = int(
            total_sequence_length
            * float(getattr(self.config, "ttt_decision_loss_start_fraction", 1.0))
        )
        decision_weight = float(getattr(self.config, "ttt_decision_loss_weight", 1.0))
        decision_action_indices = getattr(
            self.config, "ttt_decision_action_indices", None
        )
        effective_timestep_losses = torch.stack(timestep_losses)
        if decision_action_indices is not None:
            if not decision_action_indices:
                raise ValueError("ttt_decision_action_indices must be non-empty when provided")
            action_dim = action_losses[0].shape[-1]
            if max(decision_action_indices) >= action_dim:
                raise ValueError(
                    "ttt_decision_action_indices contains an index outside the action dimension "
                    f"{action_dim}: {decision_action_indices}"
                )
            selected_indices = torch.tensor(
                decision_action_indices,
                device=action_losses[0].device,
                dtype=torch.long,
            )
            selected_losses = []
            for action_loss, action_mask in zip(action_losses, action_masks):
                selected_loss = action_loss.index_select(-1, selected_indices)
                selected_mask = action_mask.index_select(-1, selected_indices)
                selected_losses.append(
                    selected_loss.sum() / (selected_mask.sum() + 1e-6)
                )
            selected_timestep_losses = torch.stack(selected_losses)
        else:
            selected_timestep_losses = effective_timestep_losses
        loss_weights = torch.ones(
            len(timestep_losses),
            device=timestep_losses[0].device,
            dtype=timestep_losses[0].dtype,
        )
        local_decision_start = max(0, decision_start - timestep_offset)
        if local_decision_start < len(loss_weights):
            loss_weights[local_decision_start:] = decision_weight
            effective_timestep_losses = effective_timestep_losses.clone()
            effective_timestep_losses[local_decision_start:] = selected_timestep_losses[
                local_decision_start:
            ]

        return BatchFeature(
            data={
                "loss": (effective_timestep_losses * loss_weights).sum()
                / loss_weights.sum(),
                "timestep_loss": effective_timestep_losses,
                "full_timestep_loss": torch.stack(timestep_losses),
                "action_loss": torch.stack(action_losses, dim=1),
                "action_mask": torch.stack(action_masks, dim=1),
                "loss_weights": loss_weights,
            }
        )

    def get_action(self, inputs: dict, options: dict[str, Any] | None = None) -> BatchFeature:
        """
        Generate actions using the complete model.
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)

        # Forward through backbone
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head.get_action(backbone_outputs, action_inputs, options)

        return action_outputs

    def reset_ttt_state(self, batch_size: int | None = None) -> None:
        """Reset episode-local RoboTTT fast weights to their learned initialization."""
        self.action_head.reset_ttt_state(batch_size)

    def detach_ttt_state(self) -> None:
        """Detach fast weights at a TBPTT segment boundary without clearing memory."""
        self.action_head.detach_ttt_state()

    def ttt_diagnostics(self) -> list[dict[str, float | int]]:
        return self.action_head.ttt_diagnostics()

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


# Register the model with HuggingFace
AutoConfig.register("Gr00tN1d7", Gr00tN1d7Config)
AutoModel.register(Gr00tN1d7Config, Gr00tN1d7)
