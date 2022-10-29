import numpy as np

class model:
    """Model describing input data given a certain library.
    
    Parameters
    ----------
    coefficients : np.ndarray
        Values of coefficients. Features not in the model have np.nan.
    error : float
        Average error from the optimisation algorithm.
    cv : float
        Average cross-validation error on a testing data set.
    score : float
        More general cross-validation score, for example based on expensive
        simulations of the model.
    window_width : str
        Effective width of the test function used for the weak formulation, 
        if applicable.
    label : str
        Label for the model. For example, for plotting purposes.
    coefstd : np.ndarray
        Standard deviation of coefficients. 
    num_terms : int
        Number of active (non-zero or non-NaN) terms in the model.
        
    """

    def __init__(self, coefficients=None, error=None, cv=None, score=None, window_width=None, label=None, coefstd=None):
        self.coefficients = coefficients
        self.error = error
        self.cv = cv
        self.score = score
        self.window_width = window_width
        self.label = label
        self.coefstd = coefstd
        if coefficients.any():
            self.num_terms = np.sum(~np.isnan(coefficients))
        else:
            self.num_terms = None
        
        
    def print(self, library):
        """Print model to shell.
        
        Parameters
        ----------
        library: library
            A library object used to fit the model.
            
        Raises
        ------
        RuntimeError
            If feature names and model coefficients don't have the same
            length   
        """
        if self.coefficients.size != len(library.feature_names):
            raise RuntimeError('Model and library sizes not matching.')
        if self.label:
            print(f"Regression label: {self.label}")
        if self.num_terms:
            print(f"Number of active terms: {self.num_terms}")
        for i,(coef,name) in enumerate(zip(self.coefficients,library.feature_names)):
            if not np.isnan(coef):
                if self.coefstd is None: 
                    print("({})\t{:+.4f} {}".format(i,coef,name))
                else:
                    print("({})\t{:+.4f} ±{:.4f} {}".format(i,coef,self.coefstd[i],name))
            
    
    def compute_cv(self, X, y):
        """Compute an average cross validation error based on coefs table.
        
        Parameters
        ----------
        X : np.ndarray
            The array of features evaluated on the test set.
        y : np.ndarray
            Test data set.
            
        Returns
        -------
        float
            The average cross-validation error."""
        cv = np.average(np.square(y-np.nansum(X*self.coefficients, axis=1)))
        self.cv = cv
        return cv
    
    
    def simulate(self, data, U, dt, library, method='Euler', return_trajectory=True):
        """Simulate the evolution of `data` under control `U`, and compute
        a simulation score.
        
        Parameters
        ----------
        data, U : array_like
            Data and control to simulate.
        dt : float
            Value of time step. Uniform time step is assumed in the data.
        library : library
            The library of feature functions
        method : str, default 'Euler'
            Integration scheme. Forward Euler by default (Itô compatible).
            If not 'Euler', calls `solve_ivp` from `scipy.integrate` and 
            tries to use it with the named method. For example, 'RK45' for
            Explicit Runge-Kutta method of order 5(4).
        return_trajectory : bool, default True
            Return trajectory (the simulation) if True.
            
        Returns
        -------
        data_sim : array_like
            Simulated data, same shape as `data`, only if return_trajectory==True
        score : float
            Simulation score defined as average square error. 
            Also stored in self.score.
        """
        
        # Extract feature functions
        coefs_features = list([coef, feature] for coef, feature in zip(self.coefficients, library.feature_functions) if not np.isnan(coef))
        
        time_base = np.arange(U.shape[0])*dt
        
        # Create a callable which returns the control at time t
        from scipy.interpolate import interp1d
        Ui = interp1d(time_base, U, kind='linear', axis=0, 
                      bounds_error=False, fill_value=(U[0],U[-1]),
                      assume_sorted=True)
        
        # Define callable rate of change
        def ddt(t, y):
            # Note the heavy use of transpose .T to comply with solve_ivp use.
            # First evaluate U:
            Ut = Ui(t)
            if Ut.ndim == 1:
                Ut = Ut.reshape((1,9))       
            out = np.zeros_like(y.T)
            for coef, feature in coefs_features:
                    out += coef*feature(y.T,Ut)
            return out.T
        
        if method!='Euler':
            # Solve using scipy integrator
            from scipy.integrate import solve_ivp
            sol = solve_ivp(ddt, [0, time_base[-1]], data[0], 
                                 method=method, t_eval=time_base,
                                 vectorized=True, max_step=dt)
            data_sim = sol.y.T
            
        if method=='Euler':
            # Use explicit first-order Euler scheme. Fast, and also makes sense if dt is the time step used in Ito MD
            Nt = U.shape[0]
            # Note: using solve_ivp conventions
            data_sim = np.empty((6,Nt))
            data_sim[:,0] = data[0]
            for i in range(Nt-1):
                data_sim[:,i+1] = data_sim[:,i] + dt*np.reshape(ddt(i*dt, np.reshape(data_sim[:,i],(6,1))),(6,))
            data_sim = data_sim.T
        
        # Compute score
        # Reweight according to implicit lower-triangular coefficients
        # of symmetric matrices
        err = (data[:,0]-data_sim[:,0])**2
        err += 2*(data[:,1]-data_sim[:,1])**2
        err += 2*(data[:,2]-data_sim[:,2])**2
        err += (data[:,3]-data_sim[:,3])**2
        err += 2*(data[:,4]-data_sim[:,4])**2
        err += (data[:,5]-data_sim[:,5])**2
        score = np.average(err)
        self.score = score
        
        if return_trajectory:
            return data_sim, score
        else:
            return score
        
        
    def is_similar(self, other_model):
        """Check is two model are similar. Two models are similar if they
        use exactly the same features.
        
        Parameters
        ----------
        other_model : model
            The model to compare with.
        
        Returns
        -------
        bool
            True if models are similar.
        """
        self_active_terms = list(~np.isnan(coef) for coef in self.coefficients)
        other_active_terms = list(~np.isnan(coef) for coef in other_model.coefficients)
        return np.array_equal(self_active_terms, other_active_terms)
