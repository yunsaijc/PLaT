# -*- coding: utf-8 -*-
"""
PLaT: Policy-guided Latent Unfolding for Generation

A latent space reasoning framework.

Module structure:
- config: Configuration management
- models: Model definitions
- datasets: Data processing
- engine: Training engine
- utils: Utility functions
- scripts: Training and testing scripts
"""

__version__ = "0.1.0"
__author__ = "JC"

from plat.config import PLaTConfig, CoTSFTConfig
from plat.models import PLaTModel, BaseReasoningModel

