"""Tester for loqs.utils.roclassproperty
"""

from abc import abstractmethod
import pytest

from loqs.internal.classproperty import (
    roclassproperty,
    abstractroclassproperty,
    HasROClassProperties,
    ABCWithROClassProperties
)

class TestROClassProperty:

    def _check(self, obj):
        # Getter should work
        assert obj.a == "A"

        # Setter should error
        with pytest.raises(AttributeError):
            obj.a = "Should not work"
    
    def _check_abstract(self, obj):
        # Getter should warn
        with pytest.warns(UserWarning):
            obj.a

        # Setter should error
        with pytest.raises(AttributeError):
            obj.a = "Should not work"

    def test_roclassproperty(self):
        class A(HasROClassProperties):
            @roclassproperty
            def a(self):
                return "A"
        
        self._check(A)
        # Should also still work on an instance
        self._check(A())
    
    def test_abstractroclassproperty(self):
        class A(HasROClassProperties):
            @abstractroclassproperty
            def a(self):
                pass
        
        class B(A):
            @roclassproperty
            def a(self):
                return "A"
        
        self._check_abstract(A)
        self._check_abstract(A())

        self._check(B)
        self._check(B())
    
    def test_abcwithroclassproperties(self):
        class A(ABCWithROClassProperties):            
            @abstractroclassproperty
            def a(self):
                pass
            
            @abstractmethod
            def b(self):
                pass
        
        class B(A):
            @roclassproperty
            def a(self):
                return "A"
            
            def b(self):
                return "B"
        
        self._check_abstract(A)

        # Making instance should fail because of ABC
        with pytest.raises(TypeError):
            A()
        
        # But B should work as intended
        self._check(B)
        self._check(B())
        assert B().b() == 'B'

        
        
        