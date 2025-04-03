import subprocess, sys, os

whoami = subprocess.Popen("whoami", shell=True, stdout=subprocess.PIPE).stdout.read().decode()[:-1]
home = "/home/{0}/sources/".format(whoami) 
sys.path.append(home + "fetricksx")
import fetricksx as ft

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from mpi4py import MPI
from petsc4py.PETSc import ScalarType  
from petsc4py import PETSc
import ufl
from dolfinx import io, fem
from dolfinx.cpp.nls.petsc import NewtonSolver
from dolfinx.common import list_timings, TimingType
from dolfinx_materials.quadrature_map import QuadratureMap
from dolfinx_materials.solvers import NonlinearMaterialProblem
from dolfinx_materials.jax_materials import (
    vonMisesIsotropicHardening,
    LinearElasticIsotropic,
)




E = 70e3
sig0 = 350.0
sigu = 500.0
b = 1e3
elastic_model = LinearElasticIsotropic(E=70e3, nu=0.3)


def yield_stress(p):
    return sig0 + (sigu - sig0) * (1 - jnp.exp(-b * p))


material = vonMisesIsotropicHardening(elastic_model, yield_stress)

mesh = ft.Mesh("meshes/mesh.geo")

# We define the function space for the displacement $\boldsymbol{u}$, interpolated here with quadratic Lagrange elements. We apply prescribed Dirichlet boundary conditions on the bottom and top dofs. For the consitutive equation, we use a quadrature degree equal to twice the degree of the associated strain i.e. `2*(order-1)=2` here.

# +
order = 2
deg_quad = 2 * (order - 1)
shape = (2,)

V = fem.functionspace(mesh, ("CG", order, shape))

uD_b = fem.Constant(mesh.domain, np.zeros(2, dtype=ScalarType))
uD_t = fem.Constant(mesh.domain, 0.2*np.ones(2, dtype=ScalarType))

bcs = [ft.dirichletbc(uD_t, 2, V), ft.dirichletbc(uD_b, 1, V)]


def strain(u):
    return ufl.as_vector(
        [
            u[0].dx(0),
            u[1].dx(1),
            0.0,
            1 / np.sqrt(2) * (u[1].dx(0) + u[0].dx(1)),
            0.0,
            0.0,
        ]
    )

u = fem.Function(V)

qmap = QuadratureMap(mesh, deg_quad, material)
print("Gradients", material.gradient_names)
print("Fluxes", material.flux_names)
qmap.register_gradient(material.gradient_names[0], strain(u))


du = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

du = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

sig = qmap.fluxes["Stress"]
C = qmap.jacobians[('Stress','Strain')]
Res = ufl.dot(sig, strain(v)) * mesh.dx
# Jac = qmap.derivative(Res, u, du)
Jac = ufl.inner(ufl.dot(C,strain(du)), strain(v))*mesh.dx

problem = NonlinearMaterialProblem(qmap, Res, Jac, u, bcs)
newton = NewtonSolver(MPI.COMM_WORLD)
newton.rtol = 1e-6
newton.atol = 1e-6
newton.convergence_criterion = "residual"
newton.max_it = 20


from dolfinx.fem.petsc import (
    assemble_vector,
    assemble_matrix,
    create_matrix,
    create_vector,
    NonlinearProblem ) 

newton.setF(Res, create_vector(problem.L))
newton.setJ(Jac, create_matrix(problem.a))


def form(x):
    x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    problem._constitutive_update()

newton.set_form(form)

it, converged = newton.solve(u.x.petsc_vec)
