from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


setup(
    name="turnpike-api",
    version="0.1.0",
    description="Turnpike MM, sorting-network MILP, and triangle certificates",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=["turnpike"],
    python_requires=">=3.9",
    install_requires=["numpy>=1.22", "scipy>=1.9"],
    extras_require={"test": ["pytest>=7"]},
    ext_modules=[
        Pybind11Extension(
            "turnpike._core",
            [
                "cpp/bindings.cpp",
                "cpp/mm.cpp",
                "cpp/network.cpp",
                "cpp/triangle.cpp",
            ],
            cxx_std=17,
        )
    ],
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
