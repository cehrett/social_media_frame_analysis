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
        'hdbscan==0.8.33',
        'ipython==8.16.1',
        'langchain==0.3.4',
        'langchain_community==0.3.3',
        'langchain_core==0.3.12',
        'langchain-openai==0.2.3',
        'Markdown==3.6',
        'matplotlib==3.8.3',
        'openai==1.52',
        'pandas==2.1.1',
        'plotly==5.19.0',
        'pyro-ppl==1.8.6',
        'scikit-learn==1.4.1.post1',
        'seaborn==0.13.0',
        'torch==2.4.0',
        'tqdm==4.66.2',
        'tiktoken',
        'umap-learn',  # Uniform Manifold Approximation and Projection for Dimension Reduction
    ],
    python_requires='>=3.11',
    include_package_data=True,
    package_data={
        'frame_extraction': ['utils/oai_system_message_template.txt']
    },
)
