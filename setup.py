#!/usr/bin/env python3
"""
AI Image Generator - Setup Script
"""
import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(e.stderr)
        return False

def main():
    """Main setup function."""
    print("🚀 AI Image Generator Setup")
    print("=" * 40)

    # Check if we're in a virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("❌ Please run this script from within a virtual environment!")
        print("   Run: python -m venv venv")
        print("   Then: venv\\Scripts\\activate")
        print("   Then: python setup.py")
        return False

    # Install PyTorch first (CUDA version)
    if not run_command(
        "pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118",
        "Installing PyTorch with CUDA support"
    ):
        return False

    # Install other requirements
    if not run_command(
        "pip install -r requirements.txt",
        "Installing remaining requirements"
    ):
        return False

    print("\n🎉 Setup completed successfully!")
    print("You can now run the application with: run.bat")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
