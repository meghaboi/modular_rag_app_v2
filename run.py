import os
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Run the Streamlit app
if __name__ == "__main__":
    import subprocess
    app_path = os.path.join(project_root, "src", "core", "app.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path]) 