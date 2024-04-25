"""Tester for loqs.utils.roclassproperty
"""

import pytest

from loqs.utils import roclassproperty

class TestROClassProperty:

    def test_roclassproperty(self):
        class A:
            @roclassproperty
            def a(self):
                return "A"
            
            @roclassproperty
            def b(self):
                raise NotImplementedError("TODO")
            
        class B(A):
            @roclassproperty
            def b(self):
                return "B"
            
        # Should work on instantiated class
        assert A.a == "A"
        with pytest.raises(NotImplementedError):
            A.b

        # Should also still work on an instance
        a = A()
        assert a.a == "A"
        with pytest.raises(NotImplementedError):
            a.b

        # And then B should have both
        assert B.a == "A"
        assert B.b == "B"

        b = B()
        assert B.a == "A"
        assert B.b == "B"

        # Setter/deleter should error
        with pytest.raises(AttributeError):
            A.a = "Should not work"
        with pytest.raises(AttributeError):
            del A.a
        
        