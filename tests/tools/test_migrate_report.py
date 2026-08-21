"""Tester for loqs.tools.migrate.report"""

from loqs.tools.migrate.report import (
    ManualReviewItem,
    annotate_manual_review,
    remap_manual_review,
)


class TestRemapManualReview:
    def test_identity_when_source_is_unchanged(self):
        items = [ManualReviewItem(line=3, message="x")]
        assert remap_manual_review("a\nb\n", "a\nb\n", items) == items

    def test_empty_review_list_is_a_no_op(self):
        assert remap_manual_review("a\n", "b\n", []) == []

    def test_line_shifts_forward_when_earlier_lines_are_removed(self):
        old = "a\nb\nc\nd\n"
        new = "a\nc\nd\n"  # "b" removed
        items = [ManualReviewItem(line=4, message="about d")]
        remapped = remap_manual_review(old, new, items)
        assert remapped == [ManualReviewItem(line=3, message="about d")]

    def test_line_shifts_backward_when_earlier_lines_are_added(self):
        old = "a\nb\n"
        new = "a\nx\ny\nb\n"  # two lines inserted before "b"
        items = [ManualReviewItem(line=2, message="about b")]
        remapped = remap_manual_review(old, new, items)
        assert remapped == [ManualReviewItem(line=4, message="about b")]

    def test_unaffected_line_is_untouched(self):
        old = "a\nb\nc\n"
        new = "a\nb\nc\nd\n"  # appended after everything
        items = [ManualReviewItem(line=1, message="about a")]
        assert remap_manual_review(old, new, items) == items


class TestAnnotateManualReview:
    def test_inserts_a_comment_above_the_flagged_line(self):
        source = "x = 1\ny = 2\n"
        items = [ManualReviewItem(line=2, message="short message")]
        annotated, remapped = annotate_manual_review(source, items)
        lines = annotated.splitlines()
        assert lines[0] == "x = 1"
        assert lines[1].startswith("# LOQS-MIGRATE")
        assert "short message" in lines[1]
        assert lines[2] == "y = 2"

    def test_mentions_v1_2_as_the_transition_point(self):
        annotated, _ = annotate_manual_review("y = 2\n", [ManualReviewItem(line=1, message="m")])
        assert "1.2" in annotated

    def test_comment_matches_the_flagged_line_indentation(self):
        source = "if True:\n    y = 2\n"
        annotated, _ = annotate_manual_review(source, [ManualReviewItem(line=2, message="m")])
        comment_line = annotated.splitlines()[1]
        assert comment_line.startswith("    #")

    def test_wraps_long_messages_to_at_most_two_lines(self):
        message = " ".join(["word"] * 60)  # much longer than the wrap width
        annotated, _ = annotate_manual_review("y = 2\n", [ManualReviewItem(line=1, message=message)])
        comment_lines = [l for l in annotated.splitlines() if l.startswith("#")]
        assert len(comment_lines) == 2

    def test_returned_manual_review_points_at_the_shifted_code_line(self):
        source = "x = 1\ny = 2\n"
        items = [ManualReviewItem(line=2, message="short message")]
        annotated, remapped = annotate_manual_review(source, items)
        lines = annotated.splitlines()
        assert len(remapped) == 1
        assert lines[remapped[0].line - 1] == "y = 2"

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

    def test_out_of_range_line_is_skipped_without_error(self):
        source = "x = 1\n"
        items = [ManualReviewItem(line=99, message="out of range")]
        annotated, remapped = annotate_manual_review(source, items)
        assert annotated == source
        assert remapped[0].line == 100  # still shifted by its own would-be comment count

    def test_empty_review_list_is_a_no_op(self):
        annotated, remapped = annotate_manual_review("x = 1\n", [])
        assert annotated == "x = 1\n"
        assert remapped == []
