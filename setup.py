from setuptools import find_packages, setup

setup(
    name="sample_flow",
    packages=find_packages(where="src"),
    package_dir={"" : "src"},
    requires=[
        "numpy",
        "matplotlib",
        "jax[cpu]",
        "flax",
        "optax",
        "scikit-learn",
        "blackjax",
        "tqdm",
        "powerbox",
        "pandas",
        "chainconsumer",
        "tensorflow_probability"
    ]
)