from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='igann-it',
    version='0.1.1',
    author='Mathias Kraus, Maximilian Veitl',
    author_email='mathias.sebastian.kraus@gmail.com, max.veitl@gmail.com',
    description='Extended implementation of Interpretable Generalized Additive Neural Networks with Interaction Terms (IGANN-IT)',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/MaximilianVeitl/igann-it',
    license='MIT',
    packages=['igann'],
    zip_safe=False,
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'torch>=1.9.0',
        'matplotlib',
        'seaborn',
        'abess==0.4.5'
    ]
)

