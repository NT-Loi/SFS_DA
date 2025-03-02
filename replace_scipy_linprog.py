import os
import shutil
import sys
import site
import argparse
import glob

def get_scipy_path(env_type):
    """Finds the SciPy installation directory based on the selected environment type."""
    if env_type == "anaconda":
        base_dir = os.environ.get("CONDA_PREFIX")
        if not base_dir:
            print("Error: Anaconda environment not detected. Exiting.")
            sys.exit(1)

        # Search for site-packages inside Anaconda
        site_packages_dirs = glob.glob(os.path.join(base_dir, "lib", "site-packages")) + \
                             glob.glob(os.path.join(base_dir, "envs", "*", "Lib", "site-packages"))  # Windows envs

    else:  # Standard Python environment (ignore Anaconda)
        site_packages_dirs = site.getsitepackages()

        # Ensure it is NOT inside Anaconda
        site_packages_dirs = [p for p in site_packages_dirs if "anaconda3" not in p.lower() and "conda" not in p.lower()]


    # Find SciPy's optimize module
    for site_packages in site_packages_dirs:
        scipy_path = os.path.join(site_packages, "scipy", "optimize")
        if os.path.exists(scipy_path):
            return scipy_path

    print("Error: SciPy optimize module not found in expected locations.")
    sys.exit(1)

def replace_files(scipy_path, new_files_dir):
    """Replaces _linprog.py and _linprog_simplex.py with new versions."""
    files_to_replace = ["_linprog.py", "_linprog_simplex.py"]

    for file_name in files_to_replace:
        old_file = os.path.join(scipy_path, file_name)
        new_file = os.path.join(new_files_dir, file_name)
        print(old_file)
        if os.path.exists(new_file):
            shutil.copy(new_file, old_file)
            print(f"Replaced {file_name} successfully.")
        else:
            print(f"Error: {new_file} not found. Skipping replacement.")

def main():
    parser = argparse.ArgumentParser(description="Replace _linprog.py and _linprog_simplex.py in SciPy.")
    parser.add_argument("--env", choices=["anaconda", "python"], required=True, help="Select the environment: anaconda or python.")
    parser.add_argument("--dir", required=True, help="Path to the directory containing the new files.")

    args = parser.parse_args()

    if not os.path.exists(args.dir):
        print("Error: Specified directory does not exist.")
        sys.exit(1)

    scipy_path = get_scipy_path(args.env)
    replace_files(scipy_path, args.dir)
    print("Replacement process completed successfully.")

if __name__ == "__main__":
    main()