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
    allocate_signature_memory
)


def command_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", help="Length", type=int, default=4096)
    parser.add_argument("--float", help="Use single precision", action="store_true")
    return parser.parse_args()

arguments = command_line()

# load source code
with open("lulesh.cc") as file:
    source = file.read()
# LULESH compiler options
compiler_options = [
    "-acc=gpu",
    "-I.",
    "-mp",
    "-O2",
    "-DUSE_MPI=0"
]
# data types
real_type = np.float64
real_bytes = 8
if arguments.float:
    real_type = np.float32
    real_bytes = 4
# preprocessor
user_preprocessor = [f"#define length {arguments.length}\n", "#define emin Real_t(-1.0e+15)\n"]

# extracting tunable code
app = Code(OpenACC(), Cxx())
preprocessor = extract_preprocessor(source)
preprocessor += user_preprocessor
signatures = extract_directive_signature(source, app)
functions = extract_directive_code(source, app)
data = extract_directive_data(source, app)

# CalcEnergyForElems_0
print(f"Tuning CalcEnergyForElems_0")
code = generate_directive_function(
    preprocessor,
    signatures["CalcEnergyForElems_0"],
    functions["CalcEnergyForElems_0"],
    app,
    data=data["CalcEnergyForElems_0"]
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
metrics["GB/s"] = lambda p: (6 * real_bytes * arguments.length / 10**9) / (p["time"] / 10**3)
metrics["GFLOPS/s"] = lambda p: (6 * arguments.length / 10**9) / (p["time"] / 10**3)

tune_kernel(
    "CalcEnergyForElems_0",
    code,
    0,
    args,
    tune_params,
    compiler_options=compiler_options,
    compiler="nvc++",
    metrics=metrics
)



