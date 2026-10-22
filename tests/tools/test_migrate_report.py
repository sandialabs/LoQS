"""Tester for loqs.tools.migrate.report"""

import pytest

from loqs.tools.migrate.report import (
    ManualReviewItem,
    RewriteItem,
    annotate_manual_review,
    remap_manual_review,
    remap_rewrites,
)


class TestRemap:
    @pytest.mark.parametrize(
        "remap, item_type",
        [
            pytest.param(
                remap_manual_review, ManualReviewItem, id="manual_review"
            ),
            pytest.param(remap_rewrites, RewriteItem, id="rewrites"),
        ],
    )
    def test_line_shifts_when_earlier_lines_are_removed(
        self, remap, item_type
    ):
        old = "a\nb\nc\nd\n"
        new = "a\nc\nd\n"  # "b" removed
        items = [item_type(line=4, message="about d")]
        remapped = remap(old, new, items)
        assert remapped == [item_type(line=3, message="about d")]


class TestAnnotateManualReview:
    def test_multiple_items_each_shift_the_next_ones_further_down(self):
        source = "a = 1\nb = 2\nc = 3\n"
        items = [
            ManualReviewItem(line=1, message="about a"),
            ManualReviewItem(line=3, message="about c"),
        ]
        annotated, remapped = annotate_manual_review(source, items)
        lines = annotated.splitlines()
        by_message = {item.message: item.line for item in remapped}
        assert lines[by_message["about a"] - 1] == "a = 1"
        assert lines[by_message["about c"] - 1] == "c = 3"
