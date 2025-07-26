from setuptools import find_packages, setup

setup(
    name="heracles_evaluation",
    version="0.0.1",
    url="",
    author="Aaron Ray",
    author_email="aray.york@gmail.com",
    description="Experimental evaluation framework for investigating interfaces between 3D scene graphs and LLMs.",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"": ["*.yaml", "*.pddl"]},
    install_requires=[
        "openai",
    ],
)
