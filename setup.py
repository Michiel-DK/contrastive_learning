from setuptools import setup, find_packages

# Read the requirements from requirements.txt
with open("requirements.txt", "r") as fh:
    requirements = fh.read().splitlines()

setup(
    name="contrastive_learning",
    version="0.1.0",
   # author="Your Name",
   # author_email="your_email@example.com",
    description="package to train contrastive model",
   # long_description=open("README.md").read(),
    #long_description_content_type="text/markdown",
    #url="https://github.com/yourusername/yourrepository",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.6",
)