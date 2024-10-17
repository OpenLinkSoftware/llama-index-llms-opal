from setuptools import setup, find_packages
import sys
import os

version = '0.3.1'

readme = os.path.join(os.path.dirname(__file__), "README.md")

setup(name="llama-index-opal",
    version=version,
    description="OpenLink Virtuoso OPAL Integration",
    long_description=open(readme).read(),
    long_description_content_type="text/markdown",
    url="https://github.com/OpenLinkSoftware/llama-index-llms-opal",
#    url='http://packages.python.org/llama-index-llms-opal',
    author='OpenLink Software',
    author_email='support@openlinksw.com',
    license='MIT',
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "Topic :: Software Development :: Libraries :: Python Modules ",
        "Operating System :: OS Independent",
    ],
    keywords="LLM,data,devtools,index,retrieval",
    project_urls={
#        "Documentation": "https://github.com/OpenLinkSoftware/llama-index-llms-opal/docs",
        "Source": "https://github.com/OpenLinkSoftware/llama-index-llms-opal",
        "Tracker": "https://github.com/OpenLinkSoftware/llama-index-llms-opal/issues",
    },
    packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
    include_package_data=True,
    install_requires=["llama-index-core>=0.10.55", "llama-index>=0.10.55", "httpx"],
    zip_safe=False,
    tests_require=["nose"],
    entry_points={
    },
)
