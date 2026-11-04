# SPDX-License-Identifier: GPL-3.0-only
"""Build optional native road and game-state extensions."""
from pathlib import Path
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class NativeBuild(build_ext):
    def build_extensions(self):
        for extension in self.extensions:
            if extension.name == "hexset._native_core":
                extension.extra_compile_args = (["/std:c++17", "/fp:strict"]
                    if self.compiler.compiler_type == "msvc" else
                    ["-std=c++17", "-fno-fast-math", "-ffp-contract=off"])
        super().build_extensions()


setup(cmdclass={"build_ext": NativeBuild}, ext_modules=[
    Extension("hexset._road_native", ["hexset/_road_native.cpp"], language="c++", optional=True),
    Extension("hexset._native_core", ["hexset/native/core.cpp"], language="c++", optional=True,
              depends=[str(path) for path in sorted(Path("hexset/native").glob("*.hpp"))]),
])
