# Original Author       : Ghasem Abdi, ghasem.abdi@yahoo.com
# File Last Update Date : April 15, 2022

import setuptools

setuptools.setup(
    name="changeDetector",
    version="0.1.0",
    description="Python library for change detection based on PyTorch",
    long_description=open(file="README.md", mode="r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Abdi-Ghasem/changeDetector",
    author="Ghasem Abdi",
    author_email="ghasem.abdi@yahoo.com",
    license="MIT License",
    classifiers=[
        "Programming Language :: Python :: 3.8", 
        "License :: MIT License", 
        "Operating System :: OS Independent"
    ],
    keywords="change detection",
    project_urls={
        "code": "https://github.com/Abdi-Ghasem/changeDetector",
        "issues": "https://github.com/Abdi-Ghasem/changeDetector/issues"
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=[
        'albumentations>=1.1.0', 
        'image_slicer>=2.1.1', 
        'matplotlib>=3.5.0', 
        'natsort>=8.0.0', 
        'numpy>=1.21.4', 
        'opencv_python>=4.5.4.60', 
        'pandas>=1.3.4',
        'Pillow>=7.2.0',
        'scikit_learn>=1.0.1', 
        'setuptools>=59.2.0', 
        'timm>=0.4.12', 
        'torch>=1.10.0', 
        'torchinfo>=1.5.3', 
        'tqdm>=4.62.3'
    ],
    python_requires=">=3.7"
)