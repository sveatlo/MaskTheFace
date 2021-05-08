import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mask-the-face",
    version="0.1.0",
    author="Svätopluk Hanzel",
    author_email="svatoplukhanzel@pm.me",
    description="Masks faces",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sveatlo/MaskTheFace",
    project_urls={
        "Bug Tracker": "https://github.com/sveatlo/MaskTheFace",
    },
    classifiers=[],
    package_dir={"": "mask_the_face"},
    package_data={
        "": ["masks/*"],
    },
    exclude_package_data={"": ["dlib_models/*"]},
    include_package_data=True,
    packages=setuptools.find_packages(where="mask_the_face"),
    python_requires=">=3.7",
)


