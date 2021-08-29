import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="settree",
    version="0.1.7",
    author="Roy Hirsch",
    author_email='royhirsch@mail.tau.ac.il',
    url='https://github.com/TAU-MLwell/Set-Tree',
    description="A framework for learning tree-based models over sets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "settree"},
    packages=setuptools.find_packages(where="settree"),
    python_requires=">=3.6",
    install_requires=['numpy>=1.19.2', 'scikit-learn>= 0.23.1', 'scipy>=pi1.5.2']

)
