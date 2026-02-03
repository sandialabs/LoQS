"""Tester for loqs.tools.reptools"""

import numpy as np
import pytest
from loqs.backends.reps import GateRep, RepTuple
from loqs.tools import reptools

try:
    # So long as we do not use get_kraus_rep_from_ptm,
    # there are no cyclical calls, i.e. these tests are still independent
    from loqs.tools.pygstitools import kraus_to_ptm
    
    NO_PYGSTI = False
except ImportError:
    NO_PYGSTI = True

class TestRepTools:

    def _check_kraus_reptuple(self, ex: RepTuple, test: RepTuple):
        assert all([np.allclose(e[0], t[0]) for e, t in zip(ex.rep, test.rep)]) #type: ignore
        assert all([np.allclose(e[1], t[1]) for e, t in zip(ex.rep, test.rep)]) #type: ignore
        assert ex.qubits == test.qubits
        assert test.reptype == GateRep.KRAUS_OPERATORS

    def test_pauli_sym_prod_phase(self):
        assert reptools.pauli_sym_prod_phase('IX', 'XX') == 1
        assert reptools.pauli_sym_prod_phase('IY', 'XX') == -1
        assert reptools.pauli_sym_prod_phase('IYY', 'XXY') == -1
        assert reptools.pauli_sym_prod_phase('IYX', 'XXY') == 1
    
    def test_pauli_rates_eigvals(self):
        # Random 1Q depolarizing channel
        p = 0.1*np.random.rand()
        depol_rates = [1-3*p/4, p/4, p/4, p/4]
        depol_eigvals = [1, 1-p, 1-p, 1-p]
        
        test_depol_eigvals = reptools.pauli_rates_to_eigvals(depol_rates)
        test_depol_rates = reptools.pauli_eigvals_to_rates(depol_eigvals)
    
        assert np.allclose(depol_eigvals, np.array(test_depol_eigvals))
        assert np.allclose(depol_rates, np.array(test_depol_rates))

        # Random 1Q stochastic channel
        rand_nonI_rates = 0.05 * np.random.rand(3)
        rand_rates = [1 - sum(rand_nonI_rates)] + list(rand_nonI_rates)
        # We can compute the expected eigvals
        rand_eigvals = [
            1,
            1 - 2*rand_rates[2] - 2*rand_rates[3],
            1 - 2*rand_rates[1] - 2*rand_rates[3],
            1 - 2*rand_rates[1] - 2*rand_rates[2]
        ]

        test_rand_eigvals = reptools.pauli_rates_to_eigvals(rand_rates)
        assert np.allclose(rand_eigvals, np.array(test_rand_eigvals))

        # Test also forward/backwards
        test_rand_rates = reptools.pauli_eigvals_to_rates(test_rand_eigvals)
        assert np.allclose(rand_rates, np.array(test_rand_rates))

        with pytest.raises(AssertionError):
            reptools.pauli_rates_to_eigvals([0, 1, 2])
        
        with pytest.raises(AssertionError):
            reptools.pauli_eigvals_to_rates([0, 1, 2])
    
    def test_dedup_kraus(self):
        rep1 = RepTuple([(np.sqrt(0.6)*np.eye(2), 0.6), (np.sqrt(0.4)*np.eye(2), 0.4)],
            [0], GateRep.KRAUS_OPERATORS)
        ex1 = RepTuple([(np.eye(2),1.0)], [0], GateRep.KRAUS_OPERATORS)

        test1 = reptools.dedup_kraus_reptuple(rep1)
        self._check_kraus_reptuple(ex1, test1)

        # Try a slightly more complicated variant
        rep2 = RepTuple([
            (np.sqrt(0.4)*np.eye(2), 0.4),
            (np.sqrt(0.3)*np.eye(2), 0.3),
            (np.sqrt(0.1)*np.eye(2), 0.1),
            (np.sqrt(0.1)*np.array([[0, 1],[1,0]]), 0.1),
            (np.sqrt(0.05)*np.array([[0, 1],[1,0]]), 0.05),
            (np.sqrt(0.05)*np.array([[1, 0],[0,-1]]), 0.05),
        ], [0], GateRep.KRAUS_OPERATORS)
        ex2 = RepTuple([
            (np.sqrt(0.8)*np.eye(2), 0.8),
            (np.sqrt(0.15)*np.array([[0, 1],[1,0]]), 0.15),
            (np.sqrt(0.05)*np.array([[1, 0],[0,-1]]), 0.05),
        ], [0], GateRep.KRAUS_OPERATORS)

        test2 = reptools.dedup_kraus_reptuple(rep2)
        self._check_kraus_reptuple(ex2, test2)

        with pytest.raises(ValueError):
            reptools.dedup_kraus_reptuple(RepTuple([(np.eye(2), None)],[0], GateRep.KRAUS_OPERATORS))
        
    def test_compose_kraus(self):
        X = np.array([[0, 1], [1, 0]])
        Z = np.array([[1, 0], [0, -1]])
        rep1 = RepTuple(Z, [0], GateRep.UNITARY)
        rep2 = RepTuple([
            (np.sqrt(0.6)*np.eye(2), 0.6),
            (np.sqrt(0.4)*X, 0.4)
        ], [0], GateRep.KRAUS_OPERATORS)

        # Z o ([I,X]) = [Z, ZX]
        ex1 = RepTuple([
            (np.sqrt(0.6)*Z, 0.6),
            (np.sqrt(0.4)*np.array([[0,-1],[1,0]]), 0.4),
        ], [0], GateRep.KRAUS_OPERATORS)

        test1 = reptools.compose_kraus_reptuples(rep1, rep2)
        self._check_kraus_reptuple(ex1, test1)

        # Test [I, X] o [X, I] = [X, I, I, X] = [X, I] deduped
        rep3 = RepTuple([
            (np.sqrt(0.3)*X, 0.3),
            (np.sqrt(0.7)*np.eye(2), 0.7)
        ], [0], GateRep.KRAUS_OPERATORS)

        ex2 = RepTuple([
            (np.sqrt(0.6*0.3)*X, 0.6*0.3),
            (np.sqrt(0.6*0.7)*np.eye(2), 0.6*0.7),
            (np.sqrt(0.4*0.3)*np.eye(2), 0.4*0.3),
            (np.sqrt(0.4*0.7)*X, 0.4*0.7),
        ], [0], GateRep.KRAUS_OPERATORS)

        test2 = reptools.compose_kraus_reptuples(rep2, rep3, dedup=False)
        print(ex2.rep)
        print(test2.rep)
        self._check_kraus_reptuple(ex2, test2)

        # Deduped now
        ex3 = RepTuple([
            (np.sqrt(0.6*0.3+0.4*0.7)*X, 0.6*0.3+0.4*0.7),
            (np.sqrt(0.6*0.7+0.4*0.3)*np.eye(2), 0.6*0.7+0.4*0.3),
        ], [0], GateRep.KRAUS_OPERATORS)
        test3 = reptools.compose_kraus_reptuples(rep2, rep3, dedup=True)
        self._check_kraus_reptuple(ex3, test3)
    
    @pytest.mark.skipif(
        NO_PYGSTI,
        reason="Skipping as pyGSTi is not available for Kraus conversion"
    )
    def test_create_stochastic_kraus(self):
        # Depolarizing test
        p = 0.1
        rep_depol_1Q = reptools.create_pauli_stochastic_kraus_rep(
            [1-3*p/4, p/4, p/4, p/4], ["Q0"]
        )

        ptm_depol = np.diag([1] + [1-p]*3)
        
        depol_Ks = [r[0] for r in rep_depol_1Q.rep] # type: ignore
        test_ptm_depol = kraus_to_ptm(depol_Ks) # type: ignore
        assert np.allclose(ptm_depol, test_ptm_depol)

        # Random 2Q Pauli stochastic
        nonI_rates = np.random.random(15)*0.05
        rates = [1-sum(nonI_rates)] + list(nonI_rates)
        
        ptm_rand_sto = np.diag(reptools.pauli_rates_to_eigvals(rates))

        rep_rand_sto = reptools.create_pauli_stochastic_kraus_rep(rates, ["Q0", "Q1"])
        
        test_Ks = [r[0] for r in rep_rand_sto.rep] # type: ignore
        test_ptm_rand_sto = kraus_to_ptm(test_Ks) # type: ignore
        assert np.allclose(ptm_rand_sto, test_ptm_rand_sto)

        test_rates = [r[1] for r in rep_rand_sto.rep] # type: ignore
        assert np.allclose(rates, test_rates)
    
    def test_create_depol_kraus(self):
        # Test against general function
        p = 0.1
        rep_depol_sto_1Q = reptools.create_pauli_stochastic_kraus_rep(
            [1-3*p/4, p/4, p/4, p/4], ["Q0"]
        )
        rep_depol_1Q = reptools.create_depolarizing_kraus_rep(p, ["Q0"])
        self._check_kraus_reptuple(rep_depol_1Q, rep_depol_sto_1Q)

        