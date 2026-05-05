import sys
is_ros = not any(arg in sys.argv for arg in ['egg_info', 'bdist_wheel', 'sdist', 'develop', 'install', "dist_info", "editable_wheel"])
if is_ros:
    from distutils.core import setup
    from catkin_pkg.python_setup import generate_distutils_setup

    d = generate_distutils_setup(
      packages=['pybullet_grasper'],
      package_dir={'': 'src'}
    )

    setup(**d)
else:

    from setuptools import setup, find_packages
    import os
    import urllib.request


    def get_data_files(directories):
        data_files = []
        for directory in directories:
            for root, _, files in os.walk(directory):
                if files:  # Only add directories that contain files
                    # Destination path inside the virtual environment
                    # e.g., venv/share/pybullet_grasper/configs/...
                    dest_dir = os.path.join('share', 'pybullet_grasper', root)

                    # List of actual file paths to copy
                    src_files = [os.path.join(root, f) for f in files]
                    data_files.append((dest_dir, src_files))
        return data_files

    app_name = "pybullet_grasper"

    with open("requirements.txt") as f:
        install_requires = f.read().splitlines()

    # URL of the README.md file (for example, from GitHub or any other location)
    url = "https://raw.githubusercontent.com/rustlluk/pybullet_grasper/refs/heads/main/README.md"

    # Fetch the README content from the URL
    try:
        with urllib.request.urlopen(url) as response:
            long_description = response.read().decode('utf-8')
    except:
        long_description = ""

    folders_to_include = ['configs', 'grippers', 'objects', 'urdf']

    setup(
        name=app_name,
        version="0.0.2",
        description="Grasping using PyBullet",

        # Tell setuptools where the Python code lives
        package_dir={"": "src"},
        packages=find_packages(where="src"),

        # Use data_files instead of package_data to grab root folders
        data_files=get_data_files(folders_to_include),

        install_requires=install_requires,
        author="Lukas Rustler",
        author_email="lukas.rustler@fel.cvut.cz",
        url="https://www.lukasrustler.cz/ShapeGrasp",
        license="Creative Commons Attribution 4.0 International (CC BY 4.0)",
        long_description=long_description,
        long_description_content_type="text/markdown",
        platforms=["any"],
        python_requires=">=3.8, <3.13"
    )