#!/usr/bin/env python3

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "AI-Driven Predictive Maintenance System for Industrial Equipment"

# Read requirements
def read_requirements():
    try:
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return [
            "pandas>=1.5.0",
            "numpy>=1.21.0",
            "scikit-learn>=1.1.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "imbalanced-learn>=0.9.0",
            "joblib>=1.2.0",
        ]

# Get long description and requirements
long_description = read_readme()
requirements = read_requirements()

setup(
    name="predictive-maintenance-ai",
    version="1.0.0",
    author="AI Predictive Maintenance Team",
    author_email="your.email@example.com",
    description="AI-Driven Predictive Maintenance System for Industrial Equipment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/predictive-maintenance-ai",
    packages=find_packages(),
    py_modules=["predictive_maintenance", "config", "utils", "api_service"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "isort>=5.0",
            "jupyter>=1.0.0",
        ],
        "api": [
            "flask>=2.0.0",
            "flask-cors>=3.0.0",
            "gunicorn>=20.0.0",
        ],
        "viz": [
            "plotly>=5.0.0",
            "dash>=2.0.0",
        ],
        "all": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "isort>=5.0",
            "jupyter>=1.0.0",
            "flask>=2.0.0",
            "flask-cors>=3.0.0",
            "gunicorn>=20.0.0",
            "plotly>=5.0.0",
            "dash>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "predictive-maintenance=predictive_maintenance:main",
            "pm-api=api_service:main",
            "pm-train=predictive_maintenance:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords="predictive maintenance, machine learning, industrial iot, failure prediction, ai",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/predictive-maintenance-ai/issues",
        "Source": "https://github.com/yourusername/predictive-maintenance-ai",
        "Documentation": "https://github.com/yourusername/predictive-maintenance-ai/blob/main/README.md",
    },
)