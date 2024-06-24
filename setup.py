from setuptools import setup, find_packages

setup(
    name='FunCode',
    version='0.2-alpha',
    packages=find_packages(),
    install_requires=["pandas", "matplotlib",
                      "numpy", "pykeepass", "openpyxl",
                      "msoffcrypto-tool", "pytest",],
    author="Ana Cristina Gonzalez-Sanchez",
    author_email="ana.cristina.gonzalez.sanchez@ki.se",
    description="Functions for excel file manipulation and descriptive stats plots",
    url="https://github.com/anacristina0914/FunCode", #URL github
)

