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
    parser.add_argument("--length", help="Length", type=int, default=4096)
    parser.add_argument("--float", help="Use single precision", action="store_true")
    return parser.parse_args()


with open("lulesh.cc") as file:
    source = file.read()
compiler_options = [
    "-acc=gpu",
    "-I.",
    "-mp",
    "-O2"
]

arguments = command_line()
user_preprocessor = [f"#define length {arguments.length}\n", "#define emin -1.0e+15\n"]

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
print(code)



