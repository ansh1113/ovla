from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ovla",
    version="0.1.0",
    author="Ansh Bhansali",
    author_email="anshbhansali5@gmail.com",
    description="Universal Vision-Language-Action Transfer through Semantic Primitive Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anshb3/ovla",
    project_urls={
        "Bug Tracker": "https://github.com/anshb3/ovla/issues",
        "Documentation": "https://github.com/anshb3/ovla",
        "Source Code": "https://github.com/anshb3/ovla",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Robotics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
    },
    include_package_data=True,
    package_data={
        "ovla": [
            "models/pretrained/*.pt",
            "examples/robots/**/*.urdf",
        ],
    },
)
