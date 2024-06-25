import argparse
import numpy as np
from kernel_tuner import tune_kernel
from kernel_tuner.utils.directives import (
    Code,
    OpenACC,
    Cxx,
    extract_directive_signature,
    extract_directive_code,
    generate_directive_function,
    extract_directive_data,
    extract_preprocessor,
)


def command_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", help="length", type=int, default=8445)
    parser.add_argument("--elems", help="numElem", type=int, default=27000)
    parser.add_argument("--nodes", help="numNode", type=int, default=29791)
    parser.add_argument("--float", help="Use single precision", action="store_true")
    return parser.parse_args()


arguments = command_line()

# load source code
with open("lulesh.cc") as file:
    source = file.read()
# LULESH compiler options
compiler_options = [
    "-acc=gpu",
    "-gpu=cc86",
    "-I.",
    "-mp",
    "-lm",
    "-O2",
    "-DUSE_MPI=0",
    "-DSEDOV_SYNC_POS_VEL_LATE=1",
]
# data types
real_type = np.float64
real_bytes = 8
if arguments.float:
    real_type = np.float32
    real_bytes = 4
# preprocessor
user_preprocessor = ["#define emin Real_t(-1.0e+15)\n"]

# extracting tunable code
app = Code(OpenACC(), Cxx())
preprocessor = extract_preprocessor(source)
signatures = extract_directive_signature(source, app)
functions = extract_directive_code(source, app)
data = extract_directive_data(source, app)

tuning_results = dict()

# CalcEnergyForElems_0
print("Tuning CalcEnergyForElems_0")
user_preprocessor += [f"#define length {arguments.length}\n"]
code = generate_directive_function(
    preprocessor + user_preprocessor,
    signatures["CalcEnergyForElems_0"],
    functions["CalcEnergyForElems_0"],
    app,
    data=data["CalcEnergyForElems_0"],
)
e_new = np.zeros(arguments.length).astype(real_type)
e_old = np.random.rand(arguments.length).astype(real_type)
p_old = np.random.rand(arguments.length).astype(real_type)
q_old = np.random.rand(arguments.length).astype(real_type)
delvc = np.random.rand(arguments.length).astype(real_type)
work = np.random.rand(arguments.length).astype(real_type)
args = [e_new, e_old, p_old, q_old, delvc, work]

tune_params = dict()
tune_params["vlength"] = [32 * i for i in range(1, 33)]
metrics = dict()
metrics["GB/s"] = lambda p: (6 * real_bytes * arguments.length / 10**9) / (
    p["time"] / 10**3
)
metrics["GFLOPS/s"] = lambda p: (6 * arguments.length / 10**9) / (p["time"] / 10**3)

tuning_results["CalcEnergyForElems_0"] = tune_kernel(
    "CalcEnergyForElems_0",
    code,
    0,
    args,
    tune_params,
    compiler_options=compiler_options,
    compiler="nvc++",
    metrics=metrics,
)

# InitStressTermsForElems
print("Tuning InitStressTermsForElems")
user_preprocessor += [
    f"#define numElem {arguments.elems}\n",
]
code = generate_directive_function(
    preprocessor + user_preprocessor,
    signatures["InitStressTermsForElems"],
    functions["InitStressTermsForElems"],
    app,
    data=data["InitStressTermsForElems"],
)
p = np.random.rand(arguments.elems).astype(real_type)
q = np.random.rand(arguments.elems).astype(real_type)
sigxx = np.zeros(arguments.elems).astype(real_type)
sigyy = np.zeros(arguments.elems).astype(real_type)
sigzz = np.zeros(arguments.elems).astype(real_type)
args = [p, q, sigxx, sigyy, sigzz]

metrics["GB/s"] = lambda p: (5 * real_bytes * arguments.elems / 10**9) / (
    p["time"] / 10**3
)
metrics["GFLOPS/s"] = lambda p: (2 * arguments.elems / 10**9) / (p["time"] / 10**3)

tuning_results["InitStressTermsForElems"] = tune_kernel(
    "InitStressTermsForElems",
    code,
    0,
    args,
    tune_params,
    compiler_options=compiler_options,
    compiler="nvc++",
    metrics=metrics,
)

# CalcForceForNodes
print("Tuning CalcForceForNodes")
user_preprocessor += [
    f"#define numNode {arguments.nodes}\n",
]
code = generate_directive_function(
    preprocessor + user_preprocessor,
    signatures["CalcForceForNodes"],
    functions["CalcForceForNodes"],
    app,
    data=data["CalcForceForNodes"],
)
fx = np.random.rand(arguments.nodes).astype(real_type)
fy = np.random.rand(arguments.nodes).astype(real_type)
fz = np.random.rand(arguments.nodes).astype(real_type)
args = [fx, fy, fz]

metrics.clear()
metrics["GB/s"] = lambda p: (3 * real_bytes * arguments.nodes / 10**9) / (
    p["time"] / 10**3
)

tuning_results["CalcForceForNodes"] = tune_kernel(
    "CalcForceForNodes",
    code,
    0,
    args,
    tune_params,
    compiler_options=compiler_options,
    compiler="nvc++",
    metrics=metrics,
)

# CalcAccelerationForNodes
print("Tuning CalcAccelerationForNodes")
code = generate_directive_function(
    preprocessor + user_preprocessor,
    signatures["CalcAccelerationForNodes"],
    functions["CalcAccelerationForNodes"],
    app,
    data=data["CalcAccelerationForNodes"],
)
fx = np.random.rand(arguments.nodes).astype(real_type)
fy = np.random.rand(arguments.nodes).astype(real_type)
fz = np.random.rand(arguments.nodes).astype(real_type)
xdd = np.zeros(arguments.nodes).astype(real_type)
ydd = np.zeros(arguments.nodes).astype(real_type)
zdd = np.zeros(arguments.nodes).astype(real_type)
nodalMass = np.random.rand(arguments.nodes).astype(real_type)
args = [fx, fy, fz, xdd, ydd, zdd, nodalMass]

tune_params.clear()
tune_params["vlength_CalcAccelerationForNodes"] = [32 * i for i in range(1, 33)]
tune_params["tile_CalcAccelerationForNodes"] = [2**i for i in range(0, 8)]
metrics["GB/s"] = lambda p: (9 * real_bytes * arguments.nodes / 10**9) / (
    p["time"] / 10**3
)
metrics["GFLOPS/s"] = lambda p: (3 * arguments.nodes / 10**9) / (p["time"] / 10**3)

tuning_results["CalcAccelerationForNodes"] = tune_kernel(
    "CalcAccelerationForNodes",
    code,
    0,
    args,
    tune_params,
    compiler_options=compiler_options,
    compiler="nvc++",
    metrics=metrics,
)


# CalcPositionForNodes
print("Tuning CalcPositionForNodes")
user_preprocessor += ["#define dt Real_t(0.4325)\n"]
code = generate_directive_function(
    preprocessor + user_preprocessor,
    signatures["CalcPositionForNodes"],
    functions["CalcPositionForNodes"],
    app,
    data=data["CalcPositionForNodes"],
)
x = np.random.rand(arguments.nodes).astype(real_type)
y = np.random.rand(arguments.nodes).astype(real_type)
z = np.random.rand(arguments.nodes).astype(real_type)
xd = np.random.rand(arguments.nodes).astype(real_type)
yd = np.random.rand(arguments.nodes).astype(real_type)
zd = np.random.rand(arguments.nodes).astype(real_type)
args = [x, y, z, xd, yd, zd]

tune_params.clear()
tune_params["vlength_CalcPositionForNodes"] = [32 * i for i in range(1, 33)]
tune_params["tile_CalcPositionForNodes"] = [2**i for i in range(0, 8)]
metrics["GB/s"] = lambda p: (9 * real_bytes * arguments.nodes / 10**9) / (
    p["time"] / 10**3
)
metrics["GFLOPS/s"] = lambda p: (6 * arguments.nodes / 10**9) / (p["time"] / 10**3)

tuning_results["CalcPositionForNodes"] = tune_kernel(
    "CalcPositionForNodes",
    code,
    0,
    args,
    tune_params,
    compiler_options=compiler_options,
    compiler="nvc++",
    metrics=metrics,
)

# CalcLagrangeElements
print("Tuning CalcLagrangeElements")
code = generate_directive_function(
    preprocessor + user_preprocessor,
    signatures["CalcLagrangeElements"],
    functions["CalcLagrangeElements"],
    app,
    data=data["CalcLagrangeElements"],
)
vdov = np.zeros(arguments.elems).astype(real_type)
dxx = np.random.rand(arguments.nodes).astype(real_type)
dyy = np.random.rand(arguments.nodes).astype(real_type)
dzz = np.random.rand(arguments.nodes).astype(real_type)
vnew = np.random.rand(arguments.nodes).astype(real_type)
args = [vdov, dxx, dyy, dzz, vnew]

tune_params.clear()
tune_params["vlength_CalcLagrangeElements"] = [32 * i for i in range(1, 33)]
tune_params["tile_CalcLagrangeElements"] = [2**i for i in range(0, 8)]
metrics["GB/s"] = lambda p: (10 * real_bytes * arguments.nodes / 10**9) / (
    p["time"] / 10**3
)
metrics["GFLOPS/s"] = lambda p: (6 * arguments.nodes / 10**9) / (p["time"] / 10**3)

tuning_results["CalcLagrangeElements"] = tune_kernel(
    "CalcLagrangeElements",
    code,
    0,
    args,
    tune_params,
    compiler_options=compiler_options,
    compiler="nvc++",
    metrics=metrics,
)
