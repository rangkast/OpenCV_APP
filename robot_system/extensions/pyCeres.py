from robot_system.resource.definition import *
import PyCeres as ceres


##
# 1D SLAM with range measurements
##

# "Between Factor" for 1D transform measurement
class Transform1DFactor(ceres.CostFunction):
    # factor initialized with a measurement and an associated variance
    def __init__(self, z, var):
        super().__init__()
        # set size of residuals and parameters
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([1, 1])
        # set internal factor variables
        self.transform = z
        self.var = var

    # computes the residual and jacobian from the factor and connected state edges
    def Evaluate(self, parameters, residuals, jacobians):
        # measurement residual compares estimated vs measured transform, scaled by
        # measurement belief
        xi = parameters[0][0]
        xj = parameters[1][0]
        residuals[0] = (self.transform - (xj - xi)) / self.var

        # jacobian of the residual w.r.t. the parameters
        if jacobians != None:
            if jacobians[0] != None:
                jacobians[0][0] = 1.0 / self.var
            if jacobians[1] != None:
                jacobians[1][0] = -1.0 / self.var

        return True


class Range1DFactor(ceres.CostFunction):
    def __init__(self, z, var):
        super().__init__()
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([1, 1])
        self.range = z
        self.var = var

    def Evaluate(self, parameters, residuals, jacobians):
        # measurement residual compares estimated vs measured distance to a
        # specific landmark, scaled by measurement belief
        l = parameters[0][0]
        x = parameters[1][0]
        residuals[0] = (self.range - (l - x)) / self.var

        if jacobians != None:
            if jacobians[0] != None:
                jacobians[0][0] = -1.0 / self.var
            if jacobians[1] != None:
                jacobians[1][0] = 1.0 / self.var

        return True


# optimization problem
problem = ceres.Problem()

# true state positions
x = np.array([0., 1., 2., 3., 4., 5., 6., 7.])
# true landmark positions
l = np.array([10., 15., 13.])

# faulty landmark position beliefs
lhat = np.array([11., 12., 15.])

# simulate noisy 1D state estimates and landmark measurements that will
# be added to the problem as factors
xhat = np.array([0., 0., 0., 0., 0., 0., 0., 0.])
mu, sigma = 0.0, 1.0  # for normal distribution scalable by variance
Tvar = 1.0e-3  # variance of transform measurement noise
rvar = 1.0e-5  # variance of range measurement noise
for i in range(x.size):
    if i > 0:
        # measured relative transform in 1D
        That = x[i] - x[i - 1] + np.random.normal(mu, sigma, 1).item() * np.sqrt(Tvar)
        # propagate frontend state estimate
        xhat[i] = xhat[i - 1] + That
        # add between factor to problem
        problem.AddResidualBlock(Transform1DFactor(That, Tvar), None, xhat[i - 1:i], xhat[i:i + 1])

    for j in range(l.size):
        # measured range from robot pose i to landmark j
        zbar = l[j] - x[i] + np.random.normal(mu, sigma, 1).item() * np.sqrt(rvar)
        # add range factor to problem
        problem.AddResidualBlock(Range1DFactor(zbar, rvar), None, lhat[j:j + 1], xhat[i:i + 1])

# initial error, for reference
init_error = np.linalg.norm(x - xhat) + np.linalg.norm(l - lhat)

# set solver options
options = ceres.SolverOptions()
options.max_num_iterations = 25
options.linear_solver_type = ceres.LinearSolverType.DENSE_QR
options.minimizer_progress_to_stdout = True

# solve!
summary = ceres.Summary()
ceres.Solve(options, problem, summary)

# report results
# print(summary.FullReport())
final_error = np.linalg.norm(x - xhat) + np.linalg.norm(l - lhat)
print('Total error of optimized states and landmarks: %f -> %f' % (init_error, final_error))
