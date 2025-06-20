"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Exceptions for the MLGym framework.
"""
from __future__ import annotations


class FormatError(Exception):
    pass

class ContextWindowExceededError(Exception):
    pass


class CostLimitExceededError(Exception):
    pass

class APIError(Exception):
    pass

class RateLimitExceededError(Exception):
    pass

class NoOutputTimeoutError(TimeoutError):
    pass
