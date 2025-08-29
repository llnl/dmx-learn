from setuptools import setup, Extension, find_packages

setup(
    name="dmx-learn",
    version="1.0.0.0",
    description="A package for estimating heterogeneous probability density functions.",
    author="Adam Walder",
    python_requires=">=3.10",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
    ],
    keywords="machine learning density estimation statistics heterogeneous data",
    license="BSD",
    install_requires=[
        "mpmath",
        "numba",
        "numpy",
        "pandas",
        "pyspark",
        "pytest",
        "scipy"
    ],
    extras_require={
        "optional": [
            "mpi4py",
            "umap-learn"
        ]
    },
    include_package_data=True,
    zip_safe=False,
)
