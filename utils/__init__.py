# -*- coding: utf-8 -*-
from plat.utils.registry import Registry, PLANNER_REGISTRY
from plat.utils.helpers import (
    validate_equation,
    extract_answer_from_text,
    extract_steps_from_text,
    compare_answers,
    batch_to_device,
    compute_metrics_from_states,
    format_reasoning_chain,
    create_context_and_target,
    safe_slice,
    compute_ema_cumulative,
    truncate_cache,
    deepcopy_cache,
)
