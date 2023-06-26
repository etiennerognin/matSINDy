import numpy as np


class FeatureLibrary:
    """FeatureLibrary of features.

    Parameters
    ----------
    output_kind : {'scalar', 'matrix'}
        Type of data returned by the FeatureLibrary functions.
    feature_functions : list of function <lambda>
        List of callable feature functions. They should take a dictionary as
        argument and return an ndarray with shape according to `output_kind`.
    feature_names : list of str
        List of corresponding feature names
    variable_names : set of str
        Set of variable names which will be called by the feature functions.
    transpose_map : dict, default None
        Transpose map with key->value dictionary where key is a variable name
        string and value is:
        1. Same as key if the variable is a symmetric matrix;
        2. Same as key with a leading `-` if the variable is skew-symmetric;
        3. The name of another variable if the transpose is in variable_names.
        4. `None` if none of the above applies.
        Examples:
        - variable_names = {'S', 'D', 'W'} with S, D symmetric and W skew, then
          transpose_map = {'S':'S', 'D':'D', 'W':'-W'}
        - variable_names = {'S', '∇U', '∇Uᵀ'} with S symmetric, then
          transpose_map = {'S':'S', '∇U':'∇Uᵀ', '∇Uᵀ':'∇U'}
    symmetry : {'symmetric', 'skew', None}, default None.
        In case of matrix output, if there is any type of symmetry by
        construction. This entry is filled by FeatureLibrary generators.
    degrees : dict of dict
        Dictionary with {feature_name: {variable_name: degree}}. Indicate the
        degree of each variable in a feature. This entry is filled by
        polynomial FeatureLibrary generators.
    """
    MATMUL_SYMBOL = '∘'
    T_SYMBOL = 'ᵀ'
    MUL_SYMBOL = ''

    def __init__(self, output_kind=None, feature_functions=None,
                 feature_names=None, variable_names=None, transpose_map=None,
                 symmetry=None, degrees=None):
        self.output_kind = output_kind
        self.feature_functions = feature_functions
        self.feature_names = feature_names
        if variable_names is not None:
            self.variable_names = set(variable_names)
        else:
            self.variable_names = variable_names
        self.transpose_map = transpose_map
        self.symmetry = symmetry
        self.degrees = degrees

    def __repr__(self):
        """String representation of a FeatureLibrary."""
        r = []
        r.append(f"Type of feature output: {self.output_kind}")
        if self.output_kind == 'matrix':
            r.append(f"Symmetry of features: {self.symmetry}")
        r.append(f"Variables names: {self.variable_names}")
        r.append(f"Variables transpose map: {self.transpose_map}")
        if self.feature_functions:
            r.append(f"Number of feature_functions: "
                     f"{len(self.feature_functions)}")
            for i, name in enumerate(self.feature_names):
                r.append(f"({i})\t{name}")
        else:
            r.append("FeatureLibrary is empty (no feature functions).")
        return '\n'.join(r)

    def __add__(self, other):
        """Concatenate two libraries. Create unique set of variable names but
        functions may be duplicated if present in both libraries.
        """
        if self.output_kind != other.output_kind:
            raise ValueError('The two libraries should be of the same kind '
                             '(scalar of matrix).')
        if self.output_kind == 'matrix' and self.symmetry != other.symmetry:
            print('Warning: adding libraries of different symmetry properties:'
                  ' result will be general.')
            symmetry = None
        else:
            symmetry = self.symmetry

        feature_functions = self.feature_functions + other.feature_functions
        feature_names = self.feature_names + other.feature_names
        variable_names = self.variable_names | other.variable_names
        transpose_map = {**self.transpose_map, **other.transpose_map}
        degrees = {**self.degrees, **other.degrees}

        return FeatureLibrary(self.output_kind,
                              feature_functions=feature_functions,
                              feature_names=feature_names,
                              variable_names=variable_names,
                              transpose_map=transpose_map,
                              symmetry=symmetry,
                              degrees=degrees
                              )

    def __iadd__(self, other):
        """See __add__."""
        if self.output_kind != other.output_kind:
            raise ValueError('The two libraries should be of the same kind '
                             '(scalar of matrix).')
        if self.output_kind == 'matrix' and self.symmetry != other.symmetry:
            print('Warning: adding libraries of different symmetry properties:'
                  'result will be general.')
            self.symmetry = None

        self.feature_functions += other.feature_functions
        self.feature_names += other.feature_names
        self.variable_names |= other.variable_names
        self.transpose_map = {**self.transpose_map, **other.transpose_map}
        self.degrees = {**self.degrees, **other.degrees}
        return self

    def __mul__(self, other):
        """Tensor product of two libraries. Behaviour:
        - Both libraries are matrix features: not implemented yet.
        - First is scalar, the other matrix: return a matrix FeatureLibrary.
        - Both scalar: return scalar FeatureLibrary
        """
        if self.output_kind == 'matrix' and other.output_kind == 'matrix':
            raise NotImplementedError('Tensor multiplication not implemented '
                                      'for matrix libraries.')
        if self.output_kind == 'matrix' and other.output_kind == 'scalar':
            raise NotImplementedError('Swap around, use: scalar*matrix')
        if self.output_kind == 'scalar' and other.output_kind == 'matrix':
            # Tensor dot scalar and matrix FeatureLibrary:
            output_kind = 'matrix'
            symmetry = other.symmetry
        else:
            output_kind = 'scalar'
            symmetry = None
        variable_names = self.variable_names | other.variable_names
        transpose_map = {**self.transpose_map, **other.transpose_map}

        feature_functions = []
        feature_names = []
        degrees = {}

        for f1, name1 in zip(self.feature_functions, self.feature_names):
            for f2, name2 in zip(other.feature_functions, other.feature_names):
                if name1 == name2:
                    # Small cosmetics
                    name12 = f"({name1})²"
                else:
                    if '+' in name2:
                        name2m = f"({name2})"
                    else:
                        name2m = name2
                    name12 = self.MUL_SYMBOL.join([name1, name2m])

                # Check is the function is already there by commutation
                # Note: this will not cover all cases!
                name21 = self.MUL_SYMBOL.join([name2, name1])
                if name21 not in feature_names:
                    if other.output_kind == 'matrix':
                        # We need to arrange broadcasting
                        feature_functions.append(lambda variables,
                                                 f1=f1, f2=f2:
                                                 f1(variables)[:, None, None]
                                                 * f2(variables))
                    else:
                        feature_functions.append(lambda variables,
                                                 f1=f1, f2=f2:
                                                 f1(variables)*f2(variables))

                    feature_names.append(name12)
                    degrees[name12] = _sum_degrees(self.degrees[name1],
                                                   other.degrees[name2])

        return FeatureLibrary(output_kind,
                              feature_functions=feature_functions,
                              feature_names=feature_names,
                              variable_names=variable_names,
                              transpose_map=transpose_map,
                              symmetry=symmetry,
                              degrees=degrees
                              )

    def __len__(self):
        """Return the number of feature functions."""
        return len(self.feature_functions)

    @classmethod
    def from_variable_names(cls, output_kind, variable_names, intercept=False):
        """Simple constructor which create a FeatureLibrary just with the
        variable names.

        Parameters
        ----------
        output_kind : {'scalar', 'matrix'}
            Type of data returned by the FeatureLibrary functions.
        variable_names : list of str
            list of variable names which will be called by the feature
            functions.
        intercept : bool, default False
            If True, add a feature returning 1 (scalar library) or the identity
            matrix (matrix library).

        Returns
        -------
        FeatureLibrary
        """
        names = list(variable_names)
        names.sort()

        feature_functions = []
        feature_names = []
        degrees = {}

        # Number of samples, only evaluated at runtime from data.
        N = 'len(next(iter(variables.values())))'

        if intercept and output_kind == 'scalar':
            feature_functions += [lambda variables, N=N: np.ones(eval(N))]
            feature_names += ['1']
            degrees['1'] = {}

        if intercept and output_kind == 'matrix':
            feature_functions += [lambda variables, N=N:
                                  np.full((eval(N), 3, 3), np.eye(3))]
            feature_names += ['I']
            degrees['I'] = {}

        for name in names:
            feature_functions += [lambda variables, name=name: variables[name]]
            feature_names += [name]
            degrees[name] = {name: 1}

        return cls(output_kind=output_kind,
                   feature_functions=feature_functions,
                   feature_names=feature_names,
                   variable_names=variable_names,
                   degrees=degrees
                   )

    @classmethod
    def from_polynomial_matrices(cls, variable_names, transpose_map, n_terms=2,
                                 intercept=True, symmetry=None):
        """Alternative constructor for FeatureLibrary. Make a list of callable
        feature functions based on matrix multiplication of all the variable
        names, along with list of function names. Variables names must label
        matrix variables.

        Parameters
        ----------
        variable_names : set of str
            List of variable names wihch will be called by the feature
            functions.
        transpose_map : dict, default None
            Transpose map with key->value dictionary where key is a variable
            name string and value is:
            1. Same as key if the variable is a symmetric matrix;
            2. Same as key with a leading `-` if the variable is skew-
            symmetric;
            3. The name of another variable if the transpose is in
            variable_names.
            4. `None` if none of the above applies.
        n_terms : int, default 1
            Maximum number of terms in multiplication.
        intercept : bool, default True
            If intercept=True, the first feature will be the identity matrix.
        symmetry : {'symmetric', 'skew', None}, default None.
            Symmetry of the generated features.
        """
        if intercept and symmetry == 'skew':
            raise ValueError('Cannot produce intercept while enforcing skew-'
                             'symmetry.')

        feature_functions = []
        feature_names = []
        degrees = {}

        if intercept:
            # Number of samples, only evaluated at runtime from data.
            N = 'len(next(iter(variables.values())))'
            feature_functions += [lambda variables, N=N:
                                  np.full((eval(N), 3, 3), np.eye(3))]
            feature_names += ['I']
            degrees['I'] = {}

        # Generate a list of tuples encoding the multiplication:
        if symmetry == 'symmetric':
            unique_symmetric = True
            remove_skew = True
            remove_symmetric = False
        elif symmetry == 'skew':
            raise NotImplementedError('Skew polynomial FeatureLibrary not'
                                      'implemented yet.')
            unique_symmetric = True  # Check condition
            remove_skew = False
            remove_symmetric = True
        elif symmetry is None:
            unique_symmetric = False
            remove_skew = False
            remove_symmetric = False
        else:
            raise ValueError("symmetry should be in"
                             "{'symmetric', 'skew', None}")
        unique_circ = False
        multiplication_tuples = _create_mul_tuples(n_terms, variable_names,
                                                   transpose_map,
                                                   remove_skew,
                                                   remove_symmetric,
                                                   unique_symmetric,
                                                   unique_circ
                                                   )

        for tup in multiplication_tuples:
            # Note we need to store the value of tup in the function definition
            if (symmetry == 'symmetric'
                    and not _is_symmetric(tup, transpose_map)):
                # We must return symmetric features. Apply the Prod+Prodᵀ rule
                feature_functions += [lambda variables, tup=tup:
                                      two_symm(matmul_by_name(variables, tup))]
                name = (f"{cls.MATMUL_SYMBOL.join(tup)} + "
                        f"({cls.MATMUL_SYMBOL.join(tup)}){cls.T_SYMBOL}")
                feature_names += [name]
            elif symmetry == 'skew' and not _is_skew(tup, transpose_map):
                # We must return skew features. Apply the Prod-Prodᵀ rule
                feature_functions += [lambda variables, tup=tup:
                                      two_skew(matmul_by_name(variables, tup))]
                name = (f"{cls.MATMUL_SYMBOL.join(tup)} - "
                        f"({cls.MATMUL_SYMBOL.join(tup)}){cls.T_SYMBOL}")
                feature_names += [name]
            else:
                # Normal case
                feature_functions += [lambda variables, tup=tup:
                                      matmul_by_name(variables, tup)]
                name = cls.MATMUL_SYMBOL.join(tup)
                feature_names += [name]

            degrees[name] = _tuple_to_degree(tup)

        return cls(output_kind='matrix',
                   feature_functions=feature_functions,
                   feature_names=feature_names,
                   variable_names=variable_names,
                   transpose_map=transpose_map,
                   symmetry=symmetry,
                   degrees=degrees
                   )

    @classmethod
    def from_polynomial_traces(cls, variable_names, transpose_map, n_terms=2,
                               intercept=True):
        """Make a library of feature functions based on the unique
        trace of matrix multiplications of all the variable names.

        Parameters
        ----------
        variable_names : set of str
            List of variable names wihch will be called by the feature
            functions.
        transpose_map : dict, default None
            Transpose map with key->value dictionary where key is a variable
            name string and value is:
            1. Same as key if the variable is a symmetric matrix;
            2. Same as key with a leading `-` if the variable is skew-
            symmetric;
            3. The name of another variable if the transpose is in
            variable_names.
            4. `None` if none of the above applies.
        n_terms : int, default 1
            Maximum number of terms in multiplication.
        intercept : bool, default False
             If intercept=True, the first element will be a function
             returning the float `1.`.
        """
        feature_functions = []
        feature_names = []
        degrees = {}

        # Number of samples, only evaluated at runtime from data.
        N = 'len(next(iter(variables.values())))'

        if intercept:
            feature_functions += [lambda variables, N=N: np.ones(eval(N))]
            feature_names += ['1']
            degrees['1'] = {}

        # Generate a list of tuples encoding the multiplication:
        # - Skew matrices have a null trace
        remove_skew = True
        remove_symmetric = False
        # - Transposed porducts have the same trace
        unique_symmetric = True
        # - Circulations have the same trace
        unique_circ = True
        multiplication_tuples = _create_mul_tuples(n_terms, variable_names,
                                                   transpose_map, remove_skew,
                                                   remove_symmetric,
                                                   unique_symmetric,
                                                   unique_circ
                                                   )

        for tup in multiplication_tuples:
            # Note we need to store the value of tup in the function definition
            feature_functions += [lambda variables, tup=tup:
                                  np.trace(matmul_by_name(variables, tup),
                                           axis1=1, axis2=2)]
            name = f"tr({cls.MATMUL_SYMBOL.join(tup)})"
            feature_names += [name]
            degrees[name] = _tuple_to_degree(tup)

        return cls(output_kind='scalar',
                   feature_functions=feature_functions,
                   feature_names=feature_names,
                   variable_names=variable_names,
                   transpose_map=transpose_map,
                   symmetry=None,
                   degrees=degrees
                   )

    def remove_by_name(self, feature_name):
        """Remove a feature from the FeatureLibrary with the corresponding name.
        Parameters
        ----------
        feature_name : str
            String of the feature name

        """
        try:
            i = self.feature_names.index(feature_name)
            self.feature_functions.pop(i)
            self.feature_names.remove(feature_name)
        except ValueError:
            raise ValueError(f"{feature_name} not in this FeatureLibrary.")
        return self

    def trim(self, max_degrees):
        """Trim a FeatureLibrary to leave only terms up to a certain degree.
        This is relying on the `degrees` attribute dictionary. Features which
        name is not present in this dictionary will be ignored.

        Parameters
        ----------
        max_degrees : dict
            Dictionary where for each variable of interest a maximum degree
            (int) is specified.
        """
        remove = []
        for feature_name, degrees in self.degrees.items():
            for max_var, max_degree in max_degrees.items():
                effective_degree = degrees.get(max_var, 0)
                if self.transpose_map[max_var] != max_var:
                    effective_degree += degrees.get(self.transpose_map[max_var], 0)
                #print(f'Feature: {feature_name}, Var: {max_var}, effective degree: {effective_degree}, max:{max_degree}')
                if effective_degree > max_degree:
                    remove.append(feature_name)
                    break

        for feature_name in remove:
            try:
                self.remove_by_name(feature_name)
                del self.degrees[feature_name]
            except ValueError:
                pass
        return self

    def compose(self, function, function_name):
        """Compose a scalar FeatureLibrary with a function. Information about
        `degrees` are passed to the new function.

        Parameters
        ----------
        function : function
            Function to use for the composition. Should take a scalar and
            return a scalar.
        function_name : str
            String to use for the function name. If `function_name` contains
            `*`, then the feature is inserted at this location
            (example: `|*|`), otherwise parentheses are used.
        """
        if self.output_kind == 'matrix':
            raise NotImplementedError('Composition should be applied to'
                                      'a scalar FeatureLibrary.')

        feature_functions = []
        feature_names = []
        degrees = {}
        for f, name in zip(self.feature_functions, self.feature_names):
            feature_functions.append(lambda variables, f=f, function=function:
                                     function(f(variables)))
            if '*' in function_name:
                s = function_name.split('*')
                name12 = ''.join([s[0], name, s[1]])
            else:
                name12 = f"{function_name}({name})"
            feature_names.append(name12)
            degrees[name12] = self.degrees[name]
        self.feature_functions = feature_functions
        self.feature_names = feature_names
        self.degrees = degrees
        return self


def _create_mul_tuples(n_terms, variable_names, transpose_map, remove_skew,
                       remove_symmetric, unique_symmetric, unique_circ):
    """List of encoding tuples for matrix multiplication. Individual names
    are sorted according to Python `sort()` method.

    Parameters
    ----------
    n_terms : int
        Number of terms in the multiplication. All possible combinations up
        to this number of terms are created.
    variable_names : set of str
        Names of the variables
    transpose_map : dict
        Dictionary of 'variable':'transposed variable' items
    remove_skew : bool
        Remove all tuples that are skew-symmetric.
    remove_symmetric : bool
        Remove all tuples that are symmetric.
    unique_symmetric : bool
        Of two tuples producing the same symmetric part, keep only one.
    unique_circ : bool
        Of n tuples being circular permuations of one another, keep only one.

    Returns
    -------
    tuple_list : list of tuple
        List of tuples with variable names encoding multiplications.
    """
    if n_terms <= 0:
        raise ValueError(f"Incorrect value for n_terms = {n_terms}. "
                         "Provide n_terms > 0")

    import itertools as it
    import more_itertools as mit

    # First we would like to sort again `variable_names` to get
    # consistent results (since the set can be sorted differently)
    names = list(variable_names)
    names.sort()

    tuple_list = []
    for i in range(n_terms):
        tuple_list += list(it.product(names, repeat=i+1))

    for tup in tuple_list:
        if remove_symmetric and _is_symmetric(tup, transpose_map):
            tuple_list.remove(tup)
        if remove_skew and _is_skew(tup, transpose_map):
            tuple_list.remove(tup)

        # Transpose tuple
        tup_T = _transpose_mul_tuple(tup, transpose_map)
        sign, tup_T_parsed = _parse_sign(tup_T)
        if unique_symmetric and not _is_symmetric(tup, transpose_map):
            try:
                tuple_list.remove(tup_T_parsed)
            except ValueError:
                pass
        if unique_circ:
            circs = mit.circular_shifts(tup)
            # Keep only different from self
            circs = [value for value in circs if value != tup]
            for circ in circs:
                try:
                    tuple_list.remove(circ)
                    # Also remove transpose of this
                    __, circ_T = _parse_sign(_transpose_mul_tuple(
                                            circ, transpose_map)
                                            )
                    if circ_T != tup:
                        tuple_list.remove(circ_T)
                except ValueError:
                    pass
    return tuple_list


def _transpose_mul_tuple(tup, transpose_map):
    """Transpose a multiplication tuple given a transpose map.

    Parameters
    ----------
    tup : tuple of str
        Tuple of variable names.
    transpose_map : dict
        Transpose map for variables.
    Returns
    -------
    tuple
        Tuple of variable names after transposition.
    """
    return tuple(transpose_map[variable] for variable in reversed(tup))


def _tuple_to_degree(tup):
    """Take a tuple of variable names to make it a dictionary of degrees.
    Parameters
    ----------
    tup : tuple of str
        Tuple of variable names.

    Returns
    -------
    dict
        Dictionary with variable names as keys and values as count in tuple.
    """
    from collections import Counter
    return dict(Counter(tup))


def _sum_degrees(degrees1, degrees2):
    """Combine and sum two degrees dictionaries. Uses collections.Counter
    """
    from collections import Counter
    return dict(Counter(degrees1)+Counter(degrees2))


def _parse_sign(variables_tuple):
    """Parse the signes in a tuple of variables.

    Parameters
    ----------
    variables_tuple : tuple of str
        Tuple of variable names

    Returns
    -------
    sign : {-1, 1}
        Overall sign of the product, if any `-` has been detected. Return `1`
        by default.
    parsed_tuple : tuple of str
        `variables_tuple` without `-` signs.
    """
    sign = 1
    out = []
    for name in variables_tuple:
        if name is not None and name.startswith('-'):
            sign *= -1
            out.append(name[1:])
        else:
            out.append(name)
    return sign, tuple(out)


def _is_symmetric(variables_tuple, transpose_map):
    """Check whether a product will be symmetric or not.

    Parameters
    ----------
    variables_tuple : tuple of str
        Tuple of variable names
    transpose_map : dict
        Transpose map dictionary

    Returns
    -------
    Bool
        `True` if product symmetric, `False` otherwise
    """
    tup_T = tuple(transpose_map[variable]
                  for variable in reversed(variables_tuple))
    sign, tup_T_parsed = _parse_sign(tup_T)
    return (variables_tuple == tup_T_parsed and sign == 1)


def _is_skew(variables_tuple, transpose_map):
    """Check whether a product will be skew or not.

    Parameters
    ----------
    variables_tuple : tuple of str
        Tuple of variable names
    transpose_map : dict
        Transpose map dictionary
    Returns
    -------
    Bool
        `True` if product skew-symmetric, `False` otherwise
    """
    tup_T = tuple(transpose_map[variable]
                  for variable in reversed(variables_tuple))
    sign, tup_T_parsed = _parse_sign(tup_T)
    return (variables_tuple == tup_T_parsed and sign == -1)


def matmul_by_name(variables, multiplication_tuple):
    """Front-end function to call np.matmul with a multiplication tuple
    containing variable names.

    Parameters
    ----------
    variables : dict
        Data in the form of dictionary, where keys are variable names
        and values are ndarrays of size (N,D,D) (usually D=3)

    multiplication_tuple : tuple of str
        Tuple of variable names in the order which the multiplication
        should be done.

    returns
    -------
    ndarray
        The result of the chained matrix multiplication.
    """
    # Using @ to call np.matmul (Python 3.5)
    variable_values_list = [f"variables[\'{name}\']"
                            for name in multiplication_tuple]
    result = eval(' @ '.join(variable_values_list))

    return result


def two_symm(matrix):
    """Computes M+M.T over axes 1 and 2 (axis 0 is the time).

    Parameters
    ----------
    matrix : ndarray (N,D,D)
        Stack of (D,D) matrices.

    Returns
    -------
    ndarray (N,D,D)
        Stack of symmetric matrices (M+M.T).
    """
    return matrix + np.transpose(matrix, axes=(0, 2, 1))


def two_skew(matrix):
    """Computes M-M.T over axes 1 and 2 (axis 0 is the time).

    Parameters
    ----------
    matrix : ndarray (N,D,D)
        Stack of (D,D) matrices.

    Returns
    -------
    ndarray (N,D,D)
        Stack of skew matrices (M-M.T).
    """
    return matrix - np.transpose(matrix, axes=(0, 2, 1))


if __name__ == '__main__':
    A = np.ones(10)
    B = np.ones((10, 3, 3))
    print(B @ A)
