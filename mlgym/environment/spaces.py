"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Spaces for Gymnasium environments.

This module provides custom Gymnasium spaces that extend the base Gymnasium spaces
with additional functionality and properties.

"""

from __future__ import annotations

from typing import Any

from gymnasium.spaces import Dict, Text

MAX_UNICODE_CODEPOINT = 0x10FFFF


class Unicode(Text):
    """A space representing a unicode string.

    Unicode is a replacement for the Text space in Gymnasium, with the
    following differences:

    - Each character can be an arbitrary unicode character.
    - The sample method samples from the specified character set.
    """

    def contains(self, x: Any) -> bool:  # noqa: ANN401
        """Return boolean specifying if x is a valid member of this space."""
        # Do not check the character set.
        return isinstance(x, str) and self.min_length <= len(x) <= self.max_length

    def __repr__(self) -> str:
        """Gives a string representation of this space."""
        return f"Unicode({self.min_length}, {self.max_length})"

    def __eq__(self, other: object) -> bool:
        """Check whether ``other`` is equivalent to this instance."""
        return (
            isinstance(other, Unicode) and self.min_length == other.min_length and self.max_length == other.max_length
        )


class AnyDict(Dict):
    def contains(self, x: Any) -> bool:  # noqa: ANN401
        return isinstance(x, dict)

    def __repr__(self) -> str:
        # TODO: define representation for the observation space
        return "ObservationSchema"

    def __eq__(self, other: object) -> bool:
        instance = isinstance(other, AnyDict)
        check_keys = self.keys() == other.keys()  # type: ignore
        check_values = all(self[key] == other[key] for key in self.keys())  # type: ignore
        return instance and check_keys and check_values
