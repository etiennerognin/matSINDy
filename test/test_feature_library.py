import numpy as np
import matsindy.feature_library as fl
import pytest


def test_edge_create_mul_tuples_zero_terms():
    """Expected behaviour for zero terms: raise ValueError"""
    n_terms = 0
    variable_names = {}
    transpose_map = {}
    remove_skew = False
    remove_symmetric = False
    unique_symmetric = False
    unique_circ = False
    with pytest.raises(ValueError):
        fl._create_mul_tuples(n_terms,
                              variable_names,
                              transpose_map,
                              remove_skew,
                              remove_symmetric,
                              unique_symmetric,
                              unique_circ
                              )


def test_create_mul_tuples_one_term():
    """Expected behaviour for one term: return sorted tuples"""
    n_terms = 1
    variable_names = {'A', 'B', 'AA'}
    transpose_map = {'A': 'A', 'B': 'B', 'AA': 'AA'}
    remove_skew = False
    remove_symmetric = False
    unique_symmetric = False
    unique_circ = False
    assert fl._create_mul_tuples(n_terms,
                                 variable_names,
                                 transpose_map,
                                 remove_skew,
                                 remove_symmetric,
                                 unique_symmetric,
                                 unique_circ
                                 ) == [('A',), ('AA',), ('B',)]


def test_create_mul_tuples_two_terms():
    """Expected behaviour for two terms: left and right multiplications."""
    n_terms = 2
    variable_names = {'A', 'B'}
    transpose_map = {'A': 'A', 'B': 'B'}
    remove_skew = False
    remove_symmetric = False
    unique_symmetric = False
    unique_circ = False
    assert fl._create_mul_tuples(n_terms,
                                 variable_names,
                                 transpose_map,
                                 remove_skew,
                                 remove_symmetric,
                                 unique_symmetric,
                                 unique_circ
                                 ) == [('A',), ('B',), ('A', 'A'), ('A', 'B'),
                                       ('B', 'A'), ('B', 'B')]


def test_create_mul_tuples_remove_skew():
    """Remove skew based on transpose map"""
    n_terms = 1
    variable_names = {'A', 'B', 'C'}
    transpose_map = {'A': 'A', 'B': '-B', 'C': None}
    remove_skew = True
    remove_symmetric = False
    unique_symmetric = False
    unique_circ = False
    assert fl._create_mul_tuples(n_terms,
                                 variable_names,
                                 transpose_map,
                                 remove_skew,
                                 remove_symmetric,
                                 unique_symmetric,
                                 unique_circ
                                 ) == [('A',), ('C',)]


def test_create_mul_tuples_remove_symmetric():
    """Remove symmetric based on transpose map"""
    n_terms = 1
    variable_names = {'A', 'B', 'C'}
    transpose_map = {'A': 'A', 'B': '-B', 'C': None}
    remove_skew = False
    remove_symmetric = True
    unique_symmetric = False
    unique_circ = False
    assert fl._create_mul_tuples(n_terms,
                                 variable_names,
                                 transpose_map,
                                 remove_skew,
                                 remove_symmetric,
                                 unique_symmetric,
                                 unique_circ
                                 ) == [('B',), ('C',)]


def test_create_mul_tuples_unique_symmetric():
    """Remove tuple leading to identical symmetrised features"""
    n_terms = 2
    variable_names = {'A', 'B'}
    transpose_map = {'A': 'A', 'B': 'B'}
    remove_skew = False
    remove_symmetric = False
    unique_symmetric = True
    unique_circ = False
    assert fl._create_mul_tuples(n_terms,
                                 variable_names,
                                 transpose_map,
                                 remove_skew,
                                 remove_symmetric,
                                 unique_symmetric,
                                 unique_circ
                                 ) == [('A',), ('B',), ('A', 'A'), ('A', 'B'),
                                       ('B', 'B')]


def test_create_mul_tuples_unique_cric():
    """Remove tuple leading to identical traces"""
    n_terms = 2
    variable_names = {'A', 'B'}
    transpose_map = {'A': 'A', 'B': 'B'}
    remove_skew = False
    remove_symmetric = False
    unique_symmetric = False
    unique_circ = True
    assert fl._create_mul_tuples(n_terms,
                                 variable_names,
                                 transpose_map,
                                 remove_skew,
                                 remove_symmetric,
                                 unique_symmetric,
                                 unique_circ
                                 ) == [('A',), ('B',), ('A', 'A'), ('A', 'B'),
                                       ('B', 'B')]


def test_matmul_by_name_one_term():
    """Should just return the value."""
    variables = {'A': np.ones((10, 3, 3))}
    multiplication_tuple = ('A')
    assert np.allclose(fl.matmul_by_name(variables, multiplication_tuple),
                       np.ones((10, 3, 3)))


def test_matmul_by_name_two_terms():
    """Matrix multiplication along the last two axes."""
    variables = {'A': np.ones((10, 3, 3)),
                 'B': np.full((10, 3, 3), np.diag([0, 1, 2]))}
    result = np.full((10, 3, 3), np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]))
    multiplication_tuple = ('A', 'B')
    assert np.allclose(fl.matmul_by_name(variables, multiplication_tuple),
                       result)


def test_edge_matmul_by_name_scalar():
    """Illegal use of matmul, should raise exception."""
    variables = {'A': np.ones(10),
                 'B': np.ones((10, 3, 3))}
    multiplication_tuple = ('A', 'B')
    with pytest.raises(ValueError):
        fl.matmul_by_name(variables, multiplication_tuple)
