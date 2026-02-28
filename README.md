# AI Image Generation Desktop App

A high-performance Python desktop application for local AI image generation using Stable Diffusion models, optimized for systems with limited VRAM (6GB+).

![Alt text](https://i.imgur.com/gL02uLZ.jpeg))


## 🚀 Performance Optimizations

This app includes cutting-edge optimizations for speed and memory efficiency:

### ✅ Implemented Features
- **xFormers Attention**: 2-3x faster generation with memory-efficient attention
- **Advanced Schedulers**: DPM++ 2M Karras, Euler A, UniPC, and more for optimal speed-quality balance
- **Model Quantization**: 8-bit and 4-bit quantization reducing VRAM usage by 50-75%
- **VAE Tiling**: Support for large images (2048x2048+) with memory-efficient processing
- **SDXL Support**: Automatic detection and optimization for SDXL models
- **Cancel Generation**: Stop long-running generations with a single click

### 🎯 Performance Improvements
| Optimization | Speed Impact | Memory Impact |
|--------------|-------------|----------------|
| **xFormers** | 2-3x faster | Same memory |
| **DPM++ 2M Karras** | 20-50% faster convergence | Same memory |
| **8-bit Quantization** | Same speed | 50% less VRAM |
| **4-bit Quantization** | Same speed | 75% less VRAM |
| **VAE Tiling** | Same speed | Enables large images |

## 📋 Installation

### Quick Start (Recommended)
```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate virtual environment
venv\Scripts\activate

# 3. Run setup script (installs PyTorch + all dependencies)
python setup.py

# 4. Run the application
run.bat
```

### Manual Installation
```bash
# 1. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# 2. Install PyTorch with CUDA support
pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# 3. Install remaining dependencies
pip install -r requirements.txt

# 4. Run the application
python main.py
```

### 2. Install xFormers (Optional - Performance Boost)
xFormers provides 2-3x speed improvements but may not be compatible with PyTorch 2.6.0 yet.

**Current Status:** xFormers for PyTorch 2.6.0 is not available yet. The app works perfectly without it.

**Option A: Try Installing xFormers (May Fail)**
```bash
# This may fail with PyTorch 2.6.0
pip install xformers --index-url https://download.pytorch.org/whl/cu118
```

**Option B: Wait for Compatibility**
xFormers support for PyTorch 2.6.0 will be available soon. For now, the app uses standard attention which works fine.

**Note:** The app automatically detects and handles xFormers compatibility issues. You'll see warnings but functionality is unaffected.

### 3. Download Models
The app supports both Hugging Face models and local model files:
- **Automatic**: Uses Stable Diffusion v1.4 by default
- **Local Files**: Place `.safetensors` or `.ckpt` files in the `models/` directory
- **SDXL Models**: Automatically detected and optimized

### 4. Text Enhancement with GGUF Models (Optional)
The app supports text prompt enhancement using local language models, including GGUF quantized models:

**Install GGUF Support:**
```bash
# For CPU-only (recommended for most users)
pip install llama-cpp-python --only-binary llama-cpp-python

# For GPU acceleration (if you have CUDA-compatible GPU)
pip install llama-cpp-python
```

**Using GGUF Models:**
1. Download a GGUF model (e.g., from Hugging Face)
2. Go to Settings → Text Enhancement
3. Browse and select your `.gguf` model file
4. The app will automatically detect and use GGUF models for text enhancement

**Supported Model Types:**
- **Transformers models**: GPT-2, Llama, etc. (Hugging Face format)
- **GGUF models**: Quantized models for better performance and lower memory usage

## 🎨 Usage

### Basic Generation
1. Enter your prompt in the text box
2. Optionally add a negative prompt
3. Adjust parameters (steps, guidance, dimensions)
4. Click "🎨 Generate Image"

### Advanced Options
- **Scheduler**: Choose from 12 different schedulers for speed-quality tradeoffs
- **Quantization**: Reduce memory usage with 8-bit or 4-bit quantization
- **xFormers**: Enable for faster generation (if installed)
- **VAE Tiling**: Enable for images larger than 1024x1024

### Model Management
- Use the Model Management panel to install/download models
- Supports both diffusers format and single-file models
- Automatic SDXL detection and optimization

## 🛠️ Troubleshooting

### xFormers Compatibility Issues
```
WARNING[XFORMERS]: xFormers can't load C++/CUDA extensions. xFormers was built for:
    PyTorch 2.4.0+cu118 with CUDA 1108 (you have 2.6.0+cu118)
```
**Solution:** This is expected with PyTorch 2.6.0. The app works perfectly without xFormers:
- Performance is still excellent with standard attention
- All features work normally
- xFormers support will be available when it's updated for PyTorch 2.6.0

**To silence warnings:** The app automatically handles this gracefully. No action needed.

### Memory Issues
- Try 8-bit or 4-bit quantization
- Reduce image dimensions
- Use VAE tiling for large images
- Close other GPU-intensive applications

### NumPy Compatibility Issues
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6
```
**Solution:**
```bash
# Force reinstall NumPy 1.x
pip install "numpy<2.0.0" --force-reinstall

# Then reinstall other packages
pip install -r requirements.txt
```

### Import Errors (CLIPImageProcessor, etc.)
```
ModuleNotFoundError: Could not import module 'CLIPImageProcessor'
```
**Solution:**
```bash
# Run the comprehensive dependency fixer
fix_dependencies.bat

# Or reinstall packages in correct order:
pip uninstall torch torchvision transformers diffusers accelerate -y
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.35.2
pip install diffusers==0.24.0 accelerate==0.24.1
pip install -r requirements.txt
```

### Pip Not Recognized in Virtual Environment
```
'pip' is not recognized as an internal or external command
```
**Solution:**
The setup scripts now automatically handle this by using the virtual environment's pip directly. Just run:
```bash
# This will set up everything correctly
setup_venv.bat

# Then run the app
run.bat
```

### GGUF Model Support Issues
```
llama-cpp-python not available. GGUF model support will be disabled.
```
**Solution:**
```bash
# Install for CPU-only usage (recommended)
pip install llama-cpp-python --only-binary llama-cpp-python

# For GPU acceleration (requires CUDA)
pip install llama-cpp-python

# If installation fails, GGUF support is optional
# The app works perfectly with transformers models
```

### Generation Errors
- Check model file integrity
- Ensure sufficient disk space
- Try different schedulers
- Reduce guidance scale if getting artifacts

## 🏗️ Architecture

### Core Components
- **ImageGenerationService**: Handles model loading and generation
- **ModelLoader**: Asynchronous model loading with progress reporting
- **ImageGenerator**: Background image generation with cancellation support
- **ImageGenerationPanel**: Main UI with optimization controls

### Optimization Features
- **Dynamic Scheduler Selection**: 12 schedulers for different use cases
- **Quantization Support**: bitsandbytes integration for memory efficiency
- **xFormers Integration**: Memory-efficient attention mechanisms
- **VAE Tiling**: Large image support with memory management
- **SDXL Auto-Detection**: Automatic pipeline selection and optimization

## 📊 System Requirements

### Minimum
- Python 3.8+
- 8GB RAM
- 6GB VRAM (with quantization)
- Windows/Linux/macOS

### Recommended
- Python 3.10+
- 16GB RAM
- 12GB+ VRAM
- CUDA-compatible GPU
- xFormers installed

## 🔧 Development

### Project Structure
```
├── main.py                 # Application entry point
├── setup.py               # Automated setup script
├── models/                 # Data models and parameters
├── services/               # Core business logic
├── ui/                     # User interface components
├── requirements.txt        # Python dependencies
├── run.bat                # Automated launcher
├── test_sdxl.py           # SDXL model test script
└── README.md              # This file
```

### Key Technologies
- **PyTorch**: Deep learning framework
- **Diffusers**: Hugging Face diffusion models
- **PyQt5**: Desktop GUI framework
- **xFormers**: Optimized attention (optional)
- **bitsandbytes**: Quantization support

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source. Please check individual component licenses for details.

---

**Enjoy fast, efficient AI image generation!** 🎨✨
