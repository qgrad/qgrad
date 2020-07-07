"""Tests for qgrad implementation of qutip functions"""
from numpy.testing import assert_almost_equal, assert_array_equal, assert_equal
from jax import grad
import jax.numpy as jnp
import pytest
from qutip import rand_ket, rand_dm, rand_herm
import numpy as np

from qgrad.qutip import (
    basis,
    coherent,
    create,
    destroy,
    expect,
    fidelity,
    isket,
    isbra,
    rot,
    to_dm,
    sigmax,
    sigmay,
    sigmaz,
    squeeze,
    Displace,
)


def test_fidelity():
    """
    Tests the fidelity function and computation of its gradient
    """
    ket0 = jnp.array([[1.0], [0]])  # represents |0>
    ket1 = jnp.array([[0.0], [1]])  # represents |1>
    ket_plus = 1 / jnp.sqrt(2) * (ket0 + ket1)  # represents |+>
    ket_minus = 1 / jnp.sqrt(2) * (ket0 - ket1)  # represents |->
    ket_complx = rand_ket(2).full()

    assert fidelity(ket0, ket1) == 0.0
    assert fidelity(ket0, ket0) == 1.0
    assert fidelity(ket1, ket1) == 1.0
    assert fidelity(ket_plus, ket_minus) == 0.0
    assert fidelity(rand_ket(4).full(), rand_ket(4).full()) <= 1.0
    assert fidelity(rand_ket(10).full(), rand_ket(10).full()) >= 0.0
    assert fidelity(ket_complx, ket_complx) == 1.0
    assert_almost_equal(fidelity(ket_plus, ket0), 1.0 / 2.0)
    assert_almost_equal(fidelity(ket_plus, ket1), 1.0 / 2.0)
    assert_almost_equal(fidelity(ket0, ket_minus), 1.0 / 2.0)
    assert_almost_equal(fidelity(ket1, ket_minus), 1.0 / 2.0)


def test_fidelity_max_dm():
    """Tests for density matrices with respect to themselves to be equal to 1 (max)"""
    for _ in range(10):
        rho1 = jnp.asarray(rand_dm(25))
        rho2 = jnp.asarray(rand_dm(25))
        assert_almost_equal(fidelity(rho1, rho1), 1.0)
        assert_almost_equal(fidelity(rho2, rho2), 1.0)


def test_fidelity_max_ket():
    """Tests for ket states with respect to themselves to be equal to 1 (max)"""
    for _ in range(10):
        ket1 = jnp.asarray(rand_ket(25))
        ket2 = jnp.asarray(rand_ket(25))
        assert_almost_equal(fidelity(ket1, ket2), 1.0)
        assert_almost_equal(fidelity(rho2, rho2), 1.0)


def test_fidelity_bounded_mixedmixed(tol=1e-7):
    """Tests for boundedness of fidelity among mixed states to be between [0, 1]"""
    for _ in range(10):
        rho1 = jnp.asarray(rand_dm(25))
        rho2 = jnp.asarray(rand_dm(25))
        F = fidelity(rho1, rho2)
        assert -tol <= F <= 1 + tol


def test_fidelity_bounded_puremixed(tol=1e-7):
    for _ in range(10):
        rho1 = jnp.asarray(rand_dm(25))
        ket1 = jnp.asarray(rand_ket(25))
        F = fidelity(rho1, ket1)
        assert -tol <= F <= 1 + tol


def test_fidelity_bounded_purepure(tol=1e-7):
    """Tests for boundedness of fidelity among kets to be between [0, 1]"""
    for _ in range(10):
        ket1 = jnp.asarray(rand_ket(25))
        ket2 = jnp.asarray(rand_ket(25))
        F = fidelity(ket1, ket22)
        assert -tol <= F <= 1 + tol


def test_rot():
    """Tests the rot function and computation of its gradient"""
    ket0 = jnp.asarray([1, 0], dtype="complex64")
    evo = jnp.dot(rot[0.5, 0.7, 0.8], ket0)
    back_evo = jnp.dot(rot[0.5, 0.7, 0.8], evo)

    assert jnp.all(rot([0, 0, 0]) == jnp.identity(2, dtype="complex64"))
    assert not jnp.all(jnp.equal(evo, back_evo))


def test_basis():
    """Tests the `basis` method"""
    np_arr = np.zeros((4, 1), dtype=np.complex64)
    np_arr[2, 0] = 1.0
    assert np.array_equal(basis(4, 2), jnp.asarray(np_arr))


def test_destroy():
    """Tests the annihilation/destroy/lowering operator"""
    # Destruction operator annihilates the bosonic number state
    b9 = basis(10, 9)  # Fock/number state with 1 at 9th index
    d10 = destroy(10)  # 10-dimensional destroy operator
    lowered = jnp.dot(d10, b9)
    assert_equal(
        np.allclose(lowered, 3.0 * basis(10, 8)), True
    )  # Multiply the eigen value
    d3 = destroy(3)
    matrix3 = jnp.asarray(
        [
            [0.00000000 + 0.0j, 1.00000000 + 0.0j, 0.00000000 + 0.0j],
            [0.00000000 + 0.0j, 0.00000000 + 0.0j, 1.41421356 + 0.0j],
            [0.00000000 + 0.0j, 0.00000000 + 0.0j, 0.00000000 + 0.0j],
        ]
    )
    assert_equal(np.allclose(matrix3, d3), True)

    assert_equal(np.allclose(dag(destroy(3)), create(3)), True)


def test_create():
    """Tests for the creation operator"""
    b3 = basis(5, 3)
    c5 = create(8)
    raised = jnp.dot(c5, b3)
    assert_equal(np.allclose(raised, 2.0 * basis(5, 4)), True)
    c3 = create(3)
    matrix3 = jnp.asarray(
        [
            [0.00000000 + 0.0j, 0.00000000 + 0.0j, 0.00000000 + 0.0j],
            [1.00000000 + 0.0j, 0.00000000 + 0.0j, 0.00000000 + 0.0j],
            [0.00000000 + 0.0j, 1.41421356 + 0.0j, 0.00000000 + 0.0j],
        ]
    )
    assert_equal(np.allclose(matrix3, c3), True)


def test_sigmax():
    assert_array_equal(sigmax(), jnp.array([[0.0, 1.0], [1.0, 0.0]]))


def test_sigmay():
    assert_array_equal(
        sigmay(), jnp.array([[0.0 + 0.0j, 0.0 - 1.0j], [0.0 + 1.0j, 0.0 + 0.0j]])
    )


def test_sigmaz():
    assert_array_equal(sigmaz(), jnp.array([[1.0, 0.0], [0.0, -1.0]]))


@pytest.mark.paramterize("oper", [sigmax(), sigmay(), sigmaz()])
@pytest.mark.paramterize("state", [basis(2, 0), basis(2, 1)])
def test_expect_sigmaxyz(oper, state):
    """Tests the `expect` function on Pauli-X, Pauli-Y and Pauli-Z."""
    # The stacked pytest decorators check all the argument combinations like a Cartesian product
    if oper != sigmaz():
        assert expect(oper, state) == 0.0
    elif state == basis(2, 0):
        assert expect(oper, state) == 1.0
    else:
        assert expect(oper, state) == -1.0


displace = Displace(4)  # initializing displace object for the test


@pytest.mark.paramterize(
    "oper, state",
    [
        (rand_herm(2), basis(2, 0)),
        (displace(1.0), basis(2, 1)),
        (squeeze(4, 1.5), basis(2, 1)),
    ],
)
def test_expect_herm(oper, state):
    """Tests that the expectation value of a hermitian operator is real and that of 
       the non-hermitian operator is complex"""
    if oper == squeeze(4, 1.5) or oper == displace(1.0):
        assert jnp.iscomplex(expect(oper, state)) == True
    else:
        assert jnp.iscomplex(expect(oper, state)) == True


@pytest.mark.parameterize(
    "oper, state", [(rand_herm(5), rand_ket(5)), (rand_dm(5), rand_ket(5))]
)
def test_expect_dag(oper, state):
    r"""Reconciles the expectation value of a random operator with the analytic calculation
       
      .. math:: <A> = <\psi|A|\psi>
    """
    expected = jnp.dot(jnp.dot(dag(state), oper), state)
    assert abs(expect(oper, state) - expected) < 1e-7


def test_coherent():
    """Tests the coherent state method"""
    assert abs(expect(destroy(10), coherent(10, 0.5))) < 1e-4


def test_dag_ket():
    r"""Tests the dagger operation :math:`A^{\dagger}` on operator :math:`A`"""
    # test with all real entries
    assert_array_equal(dag(basis(2, 0).full()), [[1.0, 0.0]])
    assert_array_equal(dag(basis(2, 1).full()), [[0.0, 1.0]])
    # test with all complex entries
    ket1 = jnp.array(
        [
            [0.04896761 + 0.18014458j],
            [0.6698803 + 0.13728367j],
            [-0.07598839 + 0.38113445j],
            [-0.00505985 + 0.10700243j],
            [-0.18735261 + 0.5476768j],
        ],
        dtype=jnp.complex64,
    )
    ket1_dag = jnp.array(
        [
            [
                0.04896761 - 0.18014458j,
                0.6698803 - 0.13728367j,
                -0.07598839 - 0.38113445j,
                -0.00505985 - 0.10700243j,
                -0.18735261 - 0.5476768j,
            ]
        ],
        dtype=jnp.complex64,
    )
    assert_array_equal(dag(ket1), ket1_dag)


@pytest.mark.repeat(10)
def test_dag_dot():
    """Tests the dagger operation with dot product"""
    i = np.random.randint(3, 10)
    ket = rand_ket(i)
    assert_almost_equal(jnp.dot(dag(ket), ket), 1.0)


def test_isket():
    """Tests the `isket` method to see whether a state is a ket based on its shape"""
    for i in range(2, 6):
        assert isket(rand_ket(i).full()) == True  # tests kets

    for j in range(2, 6):
        assert isket(dag(rand_ket(j).full())) == False  # tests bras

    for k in range(2, 6):
        assert isket(rand_dm(k).full()) == False  # tests density matrices


def test_isbra():
    """Tests the `isbra` method to see whether a state is a bra based on its shape"""
    for i in range(2, 6):
        assert isbra(rand_ket(i).full()) == False  # tests kets

    for j in range(2, 6):
        assert isket(dag(rand_ket(j).full())) == True  # tests bras

    for k in range(2, 6):
        assert isket(rand_dm(k).full()) == False  # tests density matrices


def test_to_dm():
    """Tests the `to_dm` method that converts kets and bras to density matrices"""
    dm0 = jnp.array(
        [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]], dtype=jnp.complex64
    )
    dm1 = jnp.array(
        [[0.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]], dtype=jnp.complex64
    )
    # testing kets
    assert_array_equal(to_dm(basis(2, 0)), dm0)
    assert_array_equal(to_dm(basis(2, 1)), dm1)
    # testing bras
    assert_array_equal(to_dm(dag(basis(2, 0))), dm0)
    assert_array_equal(to_dm(dag(basis(2, 1))), dm1)


def test_squeeze():
    """Tests the squeeze operator"""
    sq = squeeze(4, 0.1 + 0.1j)
    sqmatrix = jnp.array(
        [
            [
                0.99500417 + 0.0j,
                0.00000000 + 0.0j,
                0.07059289 - 0.07059289j,
                0.00000000 + 0.0j,
            ],
            [
                0.00000000 + 0.0j,
                0.98503746 + 0.0j,
                0.00000000 + 0.0j,
                0.12186303 - 0.12186303j,
            ],
            [
                -0.07059289 - 0.07059289j,
                0.00000000 + 0.0j,
                0.99500417 + 0.0j,
                0.00000000 + 0.0j,
            ],
            [
                0.00000000 + 0.0j,
                -0.12186303 - 0.12186303j,
                0.00000000 + 0.0j,
                0.98503746 + 0.0j,
            ],
        ],
        dtype=jnp.complex64,
    )

    assert_equal(np.allclose(sq, sqmatrix), True)


class TestDisplace:
    """A test class for the displace operator"""

    def test_displace():
        dp = Displace(4)
        dpmatrix = jnp.array(
            [
                [
                    0.96923323 + 0.0j,
                    -0.24230859 + 0.0j,
                    0.04282883 + 0.0j,
                    -0.00626025 + 0.0j,
                ],
                [
                    0.24230859 + 0.0j,
                    0.90866411 + 0.0j,
                    -0.33183303 + 0.0j,
                    0.07418172 + 0.0j,
                ],
                [
                    0.04282883 + 0.0j,
                    0.33183303 + 0.0j,
                    0.84809499 + 0.0j,
                    -0.41083747 + 0.0j,
                ],
                [
                    0.00626025 + 0.0j,
                    0.07418172 + 0.0j,
                    0.41083747 + 0.0j,
                    0.90866411 + 0.0j,
                ],
            ],
            dtype=jnp.complex64,
        )

        assert_equal(np.allclose(dp(0.25), dpmatrix), True)
