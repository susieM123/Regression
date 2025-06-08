'''linear_regression.py
Subclass of Analysis that performs linear regression on data
Susie Mueller
CS251 Data Analysis Visualization
Spring 2024
'''
import numpy as np
import scipy.linalg as scipy
import matplotlib.pyplot as plt

import analysis


class LinearRegression(analysis.Analysis):
    '''
    Perform and store linear regression and related analyses
    '''

    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        super().__init__(data)

        # ind_vars: Python list of strings.
        #   1+ Independent variables (predictors) entered in the regression.
        self.ind_vars = None
        # dep_var: string. Dependent variable predicted by the regression.
        self.dep_var = None

        # A: ndarray. shape=(num_data_samps, num_ind_vars)
        #   Matrix for independent (predictor) variables in linear regression
        self.A = None

        # y: ndarray. shape=(num_data_samps, 1)
        #   Vector for dependent variable (true values) being predicted by linear regression
        self.y = None

        # R2: float. R^2 statistic
        self.R2 = None

        # Mean SEE. float. Measure of quality of fit
        self.mse = None

        # slope: ndarray. shape=(num_ind_vars, 1)
        #   Regression slope(s)
        self.slope = None
        # intercept: float. Regression intercept
        self.intercept = None
        # residuals: ndarray. shape=(num_data_samps, 1)
        #   Residuals from regression fit
        self.residuals = None

        # p: int. Polynomial degree of regression model (Week 2)
        self.p = 1

    def linear_regression(self, ind_vars, dep_var, p = 1):
        '''Performs a linear regression on the independent (predictor) variable(s) `ind_vars`
        and dependent variable `dep_var.

        Parameters:
        -----------
        ind_vars: Python list of strings. 1+ independent variables (predictors) entered in the regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. 1 dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.

        TODO:
        - Use your data object to select the variable columns associated with the independent and
        dependent variable strings.
        - Perform linear regression by using Scipy to solve the least squares problem y = Ac
        for the vector c of regression fit coefficients. Don't forget to add the coefficient column
        for the intercept!
        - Compute R^2 on the fit and the residuals.
        - By the end of this method, all instance variables should be set (see constructor).

        NOTE: Use other methods in this class where ever possible (do not write the same code twice!)
        '''
        self.p = p
        self.ind_vars = np.array(ind_vars) # initialize IV 
        self.dep_var = np.array(dep_var) # initialize DV 
        self.A = self.data.select_data(ind_vars) # select IV columns using data obj
        self.y = self.data.select_data(dep_var) # select DV columns using data obj
       
        N = len(self.A)
        ones_column = np.ones((N, 1))
        if p > 1: 
            A_hat = np.hstack((ones_column, self.make_polynomial_matrix(self.A, self.p)))
        else: 
            A_hat = np.hstack((ones_column, self.A)) # create IV matrix
        c, _, _, _ = scipy.lstsq(A_hat, self.y)
        self.intercept = float(c[0]) # make intercept a float
        self.slope = c[1:] # access all slope values
        y_pred = self.predict()
        self.R2 = self.r_squared(y_pred) # computes r^2
        self.residuals = self.compute_residuals(y_pred) # computes residuals
        self.mse = self.compute_mse() # computes mean squared error 


    def predict(self, X=None):
        '''Use fitted linear regression model to predict the values of data matrix self.A.
        Generates the predictions y_pred = mA + b, where (m, b) are the model fit slope and intercept,
        A is the data matrix.

        Parameters:
        -----------
        X: ndarray. shape=(num_data_samps, num_ind_vars).
            If None, use self.A for the "x values" when making predictions.
            If not None, use X as independent var data as "x values" used in making predictions.

        Returns
        -----------
        y_pred: ndarray. shape=(num_data_samps, 1)
            Predicted y (dependent variable) values

        NOTE: You can write this method without any loops!
        '''
        c = self.slope 
        b = self.intercept
        if X is None: 
            X = np.array(self.A) 
        else: 
            X = np.array(X)
        if self.p > 1: # check if polynomial degree is greater than 1
            X = self.make_polynomial_matrix(X, self.p)
        y_pred =  X @ c + b
        return y_pred
    
    def r_squared(self, y_pred):
        '''Computes the R^2 quality of fit statistic

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps,).
            Dependent variable values predicted by the linear regression model

        Returns:
        -----------
        R2: float.
            The R^2 statistic
        '''
        mean_y = np.mean(self.y)
        SSE = np.sum((self.y - y_pred)**2) # sum of squared errors
        SST = np.sum((self.y - mean_y)**2) 
        R2 = 1 - (SSE / SST)
        # print(type(R2))
        return R2

    def compute_residuals(self, y_pred):
        '''Determines the residual values from the linear regression model

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps, 1).
            Data column for model predicted dependent variable values.

        Returns
        -----------
        residuals: ndarray. shape=(num_data_samps, 1)
            Difference between the y values and the ones predicted by the regression model at the
            data samples
        '''
        return self.y - y_pred

    def compute_mse(self):
        '''Computes the mean squared error in the predicted y compared the actual y values.
        See notebook for equation.

        Returns:
        -----------
        float. Mean squared error

        Hint: Make use of self.compute_residuals
        '''
        mse = np.square(self.compute_residuals(self.predict())).mean()
        return mse

    def scatter(self, ind_var, dep_var, title):
        '''Creates a scatter plot with a regression line to visualize the model fit.
        Assumes linear regression has been already run.

        Parameters:
        -----------
        ind_var: string. Independent variable name
        dep_var: string. Dependent variable name
        title: string. Title for the plot

        TODO:
        - Use your scatter() in Analysis to handle the plotting of points. Note that it returns
        the (x, y) coordinates of the points.
        - Sample evenly spaced x values for the regression line between the min and max x data values
        - Use your regression slope, intercept, and x sample points to solve for the y values on the
        regression line.
        - Plot the line on top of the scatterplot.
        - Make sure that your plot has a title (with R^2 value in it)
        '''
        x, y = super().scatter(ind_var, dep_var, title) # use scatter method from super class
        if self.p > 1: # check for polynomial regression
            poly_x = np.linspace(np.min(x), np.max(x), 100).reshape(-1, 1) # add trailing dimension to polynomial x values
            poly_X = self.make_polynomial_matrix(poly_x, self.p)
            poly_y = poly_X @ self.slope + self.intercept # get polynomial y values using matrix multiplication
            plt.plot(poly_x, poly_y, color = 'r', label = 'Polynomial Regression')
        else: 
            plt.plot(x, self.predict(), color = 'r', label = 'Linear Regressoin')
        plt.title(f"{title} (R^2: {self.R2:.2f})")
        plt.xlabel(ind_var)
        plt.ylabel(dep_var)
        plt.show()
        return x,y

    def pair_plot(self, data_vars, fig_sz=(12, 12), hists_on_diag=True):
        '''Mak es a pair plot with regression lines in each panel.
        There should be a len(data_vars) x len(data_vars) grid of plots, show all variable pairs
        on x and y axes.

        Parameters:
        -----------
        data_vars: Python list of strings. Variable names in self.data to include in the pair plot.
        fig_sz: tuple. len(fig_sz)=2. Width and height of the whole pair plot figure.
            This is useful to change if your pair plot looks enormous or tiny in your notebook!
        hists_on_diag: bool. If true, draw a histogram of the variable along main diagonal of
            pairplot.

        TODO:
        - Use your pair_plot() in Analysis to take care of making the grid of scatter plots.
        Note that this method returns the figure and axes array that you will need to superimpose
        the regression lines on each subplot panel.
        - In each subpanel, plot a regression line of the ind and dep variable. Follow the approach
        that you used for self.scatter. Note that here you will need to fit a new regression for
        every ind and dep variable pair.
        - Make sure that each plot has a title (with R^2 value in it)
        '''
        fig, axes = super().pair_plot(data_vars, fig_sz = fig_sz) # inherit pair plot method from parent class
        numVars = len(data_vars)
        for row in range(numVars):
            for col in range(numVars): 
                if hists_on_diag and row == col: # plot histogram on diagnol 
                    axes[row, col].remove() # removes scatterplot  
                    axes[row, col] = fig.add_subplot(numVars, numVars, row*numVars+col+1)
                    axes[row, col].hist(self.data.select_data(data_vars[row])) # graph histogram
                    if col < numVars-1:
                        axes[row, col].set_xticks([])
                    else:
                        axes[row, col].set_xlabel(data_vars[row])
                    if row > 0:
                        axes[row, col].set_yticks([])
                    else:
                        axes[row, col].set_ylabel(data_vars[col]) 
                self.linear_regression([data_vars[row]], data_vars[col]) # calculate linear regression
                x = self.data.select_data(data_vars[row])
                xLine = np.linspace(np.min(x), np.max(x), 100) 
                yLine = self.intercept + xLine[:, np.newaxis] @ self.slope # must reshape xLine
                axes[row,col].plot(xLine, yLine)
                axes[row,col].set_title(f"R^2: {self.R2:.2f}") # include r2 value in title
                if row == len(data_vars) - 1: # when row is zero
                    axes[row,col].set_xlabel(data_vars[col]) # set x label
                if col == 0: # when col is zero
                    axes[row,col].set_ylabel(data_vars[row]) # set y label


    def make_polynomial_matrix(self, A, p):
        '''Takes an independent variable data column vector `A and transforms it into a matrix appropriate
        for a polynomial regression model of degree `p`.

        (Week 2)

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, 1)
            Independent variable data column vector x
        p: int. Degree of polynomial regression model.

        Returns:
        -----------
        ndarray. shape=(num_data_samps, p)
            Independent variable data transformed for polynomial model.
            Example: if p=10, then the model should have terms in your regression model for
            x^1, x^2, ..., x^9, x^10.

        NOTE: There should not be a intercept term ("x^0"), the linear regression solver method
        should take care of that.
        '''
        matrix = np.ones((A.shape[0], p))
        for col in range(1, p + 1): # iterate over number of columns
            matrix[:, col-1] = np.squeeze(A ** col) # get rid of singleton dimension from A
        matrix = matrix.astype(float) # convert matrix to float
        return matrix 


    def poly_regression(self, ind_var, dep_var, p):
        '''Perform polynomial regression — generalizes self.linear_regression to polynomial curves
        (Week 2)
        NOTE: For single linear regression only (one independent variable only)

        Parameters:
        -----------
        ind_var: str. Independent variable entered in the single regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. Dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        p: int. Degree of polynomial regression model.
             Example: if p=10, then the model should have terms in your regression model for
             x^1, x^2, ..., x^9, x^10, and an added column of 1s for the intercept.

        TODO:
        - This method should mirror the structure of self.linear_regression (compute all the same things)
        - Differences are:
            - You create a matrix based on the independent variable data matrix (self.A) with columns
            appropriate for polynomial regresssion. Do this with self.make_polynomial_matrix.
            - You set the instance variable for the polynomial regression degree (self.p)
        '''
        pass



    def get_fitted_slope(self):
        '''Returns the fitted regression slope.
        (Week 2)

        Returns:
        -----------
        ndarray. shape=(num_ind_vars, 1). The fitted regression slope(s).
        '''
        return self.slope

    def get_fitted_intercept(self):
        '''Returns the fitted regression intercept.
        (Week 2)

        Returns:
        -----------
        float. The fitted regression intercept(s).
        '''
        return self.intercept

    def initialize(self, ind_vars, dep_var, slope, intercept, p):
        '''Sets fields based on parameter values.
        (Week 2)

        Parameters:
        -----------
        ind_vars: Python list of strings. 1+ independent variables (predictors) entered in the regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. Dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        slope: ndarray. shape=(num_ind_vars, 1)
            Slope coefficients for the linear regression fits for each independent var
        intercept: float.
            Intercept for the linear regression fit
        p: int. Degree of polynomial regression model.

        TODO:
        - Use parameters and call methods to set all instance variables defined in constructor. 
        '''
        self.ind_vars = ind_vars
        self.dep_vars = dep_var
        self.slope = slope
        self.intercept = intercept 
        self.p = p

        self.A = self.data.select_data(ind_vars)
        self.y = self.data.select_data(dep_var)

        self.R2 = self.r_squared(self.predict())
        self.residuals = self.compute_residuals(self.predict())
        self.mse = self.compute_mse()