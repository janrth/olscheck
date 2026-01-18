from setuptools import setup, find_packages

setup(
    name="olscheck",
    version="0.1.8",
    author="Jan Rathfelder",
    author_email="jan_rathfelder@yahoo.de",
    description="A library to check OLS assumptions.",
    long_description=open("readme.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/janrth/olscheck", 
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", # or another license
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "statsmodels",
        "scienceplots",
        "scikit-learn"
    ],
)
