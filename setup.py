from setuptools import find_packages, setup

setup(
    name="heracles_agents",
    version="0.0.1",
    url="",
    author="Aaron Ray",
    author_email="aray.york@gmail.com",
    description="Experimental evaluation framework for investigating interfaces between 3D scene graphs and LLMs.",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"": ["*.yaml", "*.pddl", "*.lark"]},
    install_requires=[
        "pydantic-settings",
        "plum-dispatch",
        "lark",
        "tiktoken",
        "spark-dsg",
        "heracles @ git+https://github.com/GoldenZephyr/heracles.git#subdirectory=heracles",
    ],
    extras_require={
        "openai": ["openai"],
        "anthropic": ["anthropic"],
        "ollama": ["ollama"],
        "bedrock": ["boto3"],
        "all": [
            "openai",
            "anthropic",
            "ollama",
            "boto3",
        ],
    },
)
