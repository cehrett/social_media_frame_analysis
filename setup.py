from setuptools import setup, find_packages

setup(
    name='frame_extraction',
    version='0.2.0',
    packages=find_packages(),
    description='Extract and analyze frames from social media or other corpus of short text documents.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Carl Ehrett',
    author_email='cehrett@clemson.edu',
    url='https://github.com/cehrett/social_media_frame_analysis',
    install_requires=[
        'pandas',
        'numpy',
        'openai',  
        'markdown',  # Python-Markdown library
        'IPython',  # For IPython.display functionalities
        'torch',  # PyTorch
        'pyro-ppl',  # Pyro probabilistic programming language
        'matplotlib',
        'seaborn',  # For statistical data visualization
        'plotly',  # For interactive plots
        'scikit-learn',  # Scikit-learn for machine learning
        'hdbscan',  # For clustering
        'tqdm',  # For progress bars
        'langchain'  
    ],
    python_requires='>=3.9.18',
)
