"""Tests for the attribute checker plugin."""
from src.attribute_checker import AttributeCheckerPlugin


def _check(code: str):
    """Helper: run AttributeCheckerPlugin on a code string."""
    plugin = AttributeCheckerPlugin()
    return plugin.check(code=code)


class TestPhantomAttributes:
    """Test phantom attribute detection on built-in types."""

    def test_list_push_reported_as_hallucination(self):
        """list.push() is valid in JS/Java but not on Python's list."""
        # Current behaviour: reported as a HALLUCINATION. Under Condition 2 of
        # the paper this is a grounded error (a real method applied to the
        # wrong type); the same limitation affects the labelling rule
        # (Section 5.1).
        result = _check("lst = [1, 2]\nlst.push(3)")
        assert result is not None
        assert not result.is_grounded_error
        assert result.hallucination_category is not None

    def test_list_add_reported_as_hallucination(self):
        """list.add() exists on set but not on list."""
        # Current behaviour: reported as a HALLUCINATION. Under Condition 2 of
        # the paper this is a grounded error (a real method applied to the
        # wrong type); the same limitation affects the labelling rule
        # (Section 5.1).
        result = _check("lst = [1, 2]\nlst.add(3)")
        assert result is not None
        assert not result.is_grounded_error
        assert result.hallucination_category is not None

    def test_str_append_reported_as_hallucination(self):
        """str.append() exists on list but not on str."""
        # Current behaviour: reported as a HALLUCINATION. Under Condition 2 of
        # the paper this is a grounded error (a real method applied to the
        # wrong type); the same limitation affects the labelling rule
        # (Section 5.1).
        result = _check('text = "hello"\ntext.append("!")')
        assert result is not None
        assert not result.is_grounded_error
        assert result.hallucination_category is not None

    def test_list_first_is_hallucination(self):
        """list.first doesn't exist anywhere — pure hallucination."""
        result = _check("lst = [1, 2, 3]\nx = lst.first")
        assert result is not None
        assert not result.is_grounded_error
        assert result.hallucination_category is not None

    def test_valid_list_method_passes(self):
        """list.append() is valid and should pass."""
        result = _check("lst = [1, 2]\nlst.append(3)")
        assert result is None

    def test_valid_str_method_passes(self):
        """str.upper() is valid and should pass."""
        result = _check('s = "hello"\nresult = s.upper()')
        assert result is None

    def test_dict_keys_passes(self):
        """dict.keys() is valid."""
        result = _check('d = {"a": 1}\nk = d.keys()')
        assert result is None


class TestBoundMethodNoCall:
    """Test detection of bound method access without parentheses."""

    def test_upper_without_parens(self):
        """s.upper without () should be caught as grounded error."""
        result = _check('text = "hello"\nresult = text.upper')
        assert result is not None
        assert result.is_grounded_error
        assert "without calling" in result.explanation

    def test_upper_with_parens_passes(self):
        """s.upper() with () should pass."""
        result = _check('text = "hello"\nresult = text.upper()')
        assert result is None


class TestNoFalsePositives:
    """Regression tests for previously identified false positive patterns."""

    def test_unknown_var_not_flagged(self):
        """Variables with no type info should not be flagged."""
        result = _check("x.method()")
        assert result is None  # x is unknown, skip

    def test_function_param_dict_keys(self):
        """A dict parameter calling .keys() should pass."""
        code = "def process(config):\n    return list(config.keys())"
        result = _check(code)
        assert result is None
