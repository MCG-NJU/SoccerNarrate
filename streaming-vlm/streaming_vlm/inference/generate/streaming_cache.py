from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from transformers import DynamicCache
from transformers.cache_utils import DynamicLayer


class _CacheView:
    """Lightweight view that keeps `key_cache`/`value_cache` in sync with DynamicCache layers."""

    def __init__(self, parent: DynamicCache, attr: str):
        self.parent = parent
        self.attr = attr

    def _ensure_layer(self, layer_idx: int) -> DynamicLayer:
        # Lazily create missing layers so callers can assign directly.
        while len(self.parent.layers) <= layer_idx:
            layer_cls = self.parent.layer_class_to_replicate
            layer = layer_cls() if layer_cls is not None else DynamicLayer()
            # Default empty tensors for consistency with older key_cache/value_cache code paths.
            layer.keys = torch.tensor([])
            layer.values = torch.tensor([])
            self.parent.layers.append(layer)
        layer = self.parent.layers[layer_idx]
        # Backfill missing attributes when an empty layer is accessed directly.
        if not hasattr(layer, self.attr):
            setattr(layer, self.attr, torch.tensor([]))
        # Keep `is_initialized` untouched so DynamicLayer can lazy-init with the right device/dtype.
        return layer

    def __len__(self) -> int:
        return len(self.parent.layers)

    def __getitem__(self, layer_idx: int) -> torch.Tensor:
        layer = self._ensure_layer(layer_idx)
        return getattr(layer, self.attr)

    def __setitem__(self, layer_idx: int, value: torch.Tensor) -> None:
        layer = self._ensure_layer(layer_idx)
        setattr(layer, self.attr, value)
        # Keep metadata aligned so get_seq_length() and dtype/device dependent paths work.
        layer.is_initialized = True
        if hasattr(layer, "dtype"):
            layer.dtype = value.dtype
        if hasattr(layer, "device"):
            layer.device = value.device


# Inherits from DynamicCache but adds position cache and legacy key/value accessors
class StreamingCache(DynamicCache):
    def __init__(self, _distributed_cache_data: Optional[Iterable] = None, **kwargs) -> None:
        super().__init__(_distributed_cache_data, **kwargs)
        self.position_ids_cache: List[torch.Tensor] = []
        # Provide legacy-style access for callers that mutate key/value tensors directly.
        self.key_cache = _CacheView(self, "keys")
        self.value_cache = _CacheView(self, "values")

    def _ensure_layer(self, layer_idx: int) -> DynamicLayer:
        # Helper so views and updates can grow the cache lazily.
        return self.key_cache._ensure_layer(layer_idx)

    def update_position_ids(self, position_ids: torch.Tensor, layer_idx: int = 0):
        """
        Parameters:
            position_ids (`torch.Tensor`):
                The new position ids to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
        """
        if position_ids is not None:
            if len(self.position_ids_cache) <= layer_idx:
                # There may be skipped layers, fill them with empty lists
                for _ in range(len(self.position_ids_cache), layer_idx):
                    self.position_ids_cache.append(torch.tensor([]))
                self.position_ids_cache.append(position_ids)
            else:  # fills previously skipped layers; checking for tensor causes errors
                self.position_ids_cache[layer_idx] = position_ids

        return self.position_ids_cache[layer_idx]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *update_args,
        cache_kwargs: Optional[Dict[str, Any]] = None,
        **update_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Supports both the current signature (`update(key, value, layer_idx, cache_kwargs)`)
        and the older variant (`update(key, value, layer_idx, position_ids, cache_kwargs)`).
        """
        position_ids = update_kwargs.pop("position_ids", None)

        # Backwards-compatible positional handling.
        if len(update_args) == 1:
            if isinstance(update_args[0], dict) or update_args[0] is None:
                cache_kwargs = update_args[0]
            else:
                position_ids = update_args[0]
        elif len(update_args) >= 2:
            position_ids = update_args[0]
            cache_kwargs = update_args[1]

        if "cache_kwargs" in update_kwargs:
            cache_kwargs = update_kwargs["cache_kwargs"]

        # Grow cache lazily and optionally stash position ids for shrink mode.
        self._ensure_layer(layer_idx)
        if position_ids is not None:
            self.update_position_ids(position_ids, layer_idx)

        keys, values = super().update(key_states, value_states, layer_idx, cache_kwargs=cache_kwargs)
        return keys, values
