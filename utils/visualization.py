"""Visualization utilities for model interpretability."""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from typing import Any


class GradCAM:
    """Gradient-weighted Class Activation Mapping for CNN models."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model: Target model
            target_layer: Layer to visualize (e.g., model.layer4, model.features[-1])
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_full_backward_hook(self.save_gradients)

    def save_activations(self, module: nn.Module, input: tuple, output: torch.Tensor):
        """Forward hook to save activations."""
        self.activations = output.detach()

    def save_gradients(self, module: nn.Module, grad_input: tuple, grad_output: tuple):
        """Backward hook to save gradients."""
        self.gradients = grad_output[0].detach()

    def __call__(self, images: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.

        Args:
            images: Input tensor [B, 3, H, W]
            class_idx: Target class index. If None, uses predicted class.

        Returns:
            Grad-CAM heatmap [H, W] in range [0, 1]
        """
        batch_size = images.shape[0]
        
        # Forward pass
        self.model.eval()
        logits = self.model(images)
        
        if class_idx is None:
            class_idx = logits.argmax(dim=1)[0].item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for target class
        target_score = logits[:, class_idx]
        target_score.sum().backward(retain_graph=True)

        # Compute Grad-CAM
        gradients = self.gradients[0].cpu().numpy()  # [C, H, W]
        activations = self.activations[0].cpu().numpy()  # [C, H, W]

        weights = gradients.mean(axis=(1, 2), keepdims=True)  # [C, 1, 1]
        cam = np.sum(weights * activations, axis=0)  # [H, W]
        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)

        return cam


class AttentionRollout:
    """Attention Rollout for ViT models."""

    def __init__(self, model: nn.Module, layers: list[nn.Module] | None = None):
        """
        Args:
            model: ViT model
            layers: List of attention layers. If None, auto-detect from model.
        """
        self.model = model
        self.attention_maps = []
        self.hooks = []

        if layers is None:
            layers = self._get_attention_layers(model)

        for layer in layers:
            h = layer.register_forward_hook(self.save_attention)
            self.hooks.append(h)

    def _get_attention_layers(self, model: nn.Module) -> list[nn.Module]:
        """Auto-detect attention layers from ViT model."""
        layers = []
        for module in model.modules():
            if hasattr(module, "attention"):
                layers.append(module.attention)
        return layers

    def save_attention(self, module: nn.Module, input: tuple, output: tuple):
        """Hook to save attention weights."""
        if isinstance(output, tuple):
            attention = output[1] if len(output) > 1 else None
        else:
            attention = None
        if attention is not None:
            self.attention_maps.append(attention.detach().cpu())

    def __call__(self, images: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """
        Generate attention rollout heatmap.

        Args:
            images: Input tensor [1, 3, H, W]
            class_idx: Target class index. If None, uses predicted class.

        Returns:
            Attention rollout heatmap [H, W] in range [0, 1]
        """
        self.attention_maps = []
        self.model.eval()

        with torch.no_grad():
            logits = self.model(images)

        if class_idx is None:
            class_idx = logits.argmax(dim=1)[0].item()

        # If no attention maps captured, return uniform attention
        if not self.attention_maps:
            return np.ones((images.shape[2] // 16, images.shape[3] // 16)) / (images.shape[2] // 16)

        # Stack attention maps [num_layers, num_heads, seq_len, seq_len]
        attention = torch.stack(self.attention_maps)
        attention = attention.mean(dim=1)  # Average over heads [num_layers, seq_len, seq_len]

        # Rollout: recursively identity attention from all layers
        rollout = torch.eye(attention.shape[1]).to(attention.device)
        for attn in attention:
            rollout = torch.matmul(attn, rollout)

        # Extract class token attention (last patch is usually class token)
        mask = rollout[0, 0, 1:].cpu().numpy()  # [num_patches-1]
        mask = np.maximum(mask, 0)
        mask = mask / (mask.max() + 1e-8)

        # Reshape to spatial dimensions (assumes square patches)
        num_patches = int(np.sqrt(len(mask)))
        img_h = images.shape[2]
        patch_h = img_h // num_patches

        heatmap = mask.reshape(num_patches, num_patches)
        heatmap = cv2.resize(heatmap, (img_h, img_h))

        return heatmap


def visualize_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay heatmap on image.

    Args:
        image: Input image [H, W, 3] in range [0, 255]
        heatmap: Heatmap [H, W] in range [0, 1]
        alpha: Blending factor
        colormap: OpenCV colormap

    Returns:
        Blended image [H, W, 3]
    """
    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)

    # Convert heatmap to uint8
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)

    # Apply colormap
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)

    # Convert BGR to RGB
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Blend
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)

    return overlay
