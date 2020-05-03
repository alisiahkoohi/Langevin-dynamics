import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

reqs = ['torch', 'torchvision', 
    "rosenbrock @ git+https://github.com/alisiahkoohi/rosenbrock@master"]
setuptools.setup(
    name="langevin_sampling",
    version="0.1",
    author="Ali Siahkoohi",
    author_email="alisk@gatech.edu",
    description="Sampling with gradient-based Markov Chain Monte Carlo approaches",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alisiahkoohi/Langevin-dynamics",
    license='MIT',
    install_requires=reqs,
    packages=setuptools.find_packages()
)
