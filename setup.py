from setuptools import setup

setup(
    name="pytorch_classification",
    version="0.0.1",
    anthor="tanganke",
    packages=["pytorch_classification"],
    requires=["hydra-core", "matplotlib", "pytorch_lightning"],
)
