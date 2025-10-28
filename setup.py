from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="rag-trading-system",
    version="1.0.0",
    author="RAG Trading System",
    author_email="trading@example.com",
    description="RAG-based trading system for XAUUSD analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/rag-trading-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
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
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "windows": [
            "MetaTrader5>=5.0.37",
        ],
    },
    entry_points={
        "console_scripts": [
            "rag-trading=trading_rag:main",
            "mt5-export=scripts.mt5_remote_client:main",
            "rag-import=scripts.batch_import:main",
            "rag-convert=scripts.rag_converter:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.json"],
    },
)