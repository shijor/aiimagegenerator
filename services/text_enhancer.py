"""
Text enhancement service for AI Image Generator.
Handles loading text models and enhancing user prompts.
"""
import os
import gc
import time
import threading
from typing import Optional, Callable, Any
from PyQt5.QtCore import QThread, pyqtSignal, QObject

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  transformers library not available - ML-based enhancement disabled")

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
    print("✅ llama-cpp-python available - GGUF model support enabled")
except ImportError as e:
    LLAMA_CPP_AVAILABLE = False
    print("⚠️  llama-cpp-python not available - GGUF model support disabled")
    print("   Using transformers-based text enhancement instead")
    print()
    print("   Recommended transformers models for text enhancement:")
    print("   • gpt2-medium (balanced quality/speed)")
    print("   • microsoft/DialoGPT-medium (conversational)")
    print("   • EleutherAI/gpt-neo-1.3B (high quality)")
    print("   • distilgpt2 (fast, lightweight)")
    print()
    print("   These models work out-of-the-box with GPU acceleration!")
    print()
    print(f"   GGUF setup (if needed): conda install -c conda-forge llama-cpp-python")
    print(f"   Import error: {e}")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
    print("✅ OpenAI library available - OpenAI API enhancement enabled")
except ImportError as e:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI library not available - OpenAI API enhancement disabled")
    print("   Install with: pip install openai")
    print(f"   Import error: {e}")


class TextEnhancementWorker(QThread):
    """Worker thread for text enhancement to avoid blocking UI."""

    finished = pyqtSignal(str)  # Enhanced prompt
    error = pyqtSignal(str)     # Error message

    def __init__(self, model_path: str, enhancement_prompt: str, user_prompt: str,
                 enable_openai: bool = False, openai_api_key: str = None, openai_model: str = "gpt-3.5-turbo"):
        super().__init__()
        self.model_path = model_path
        self.enhancement_prompt = enhancement_prompt
        self.user_prompt = user_prompt
        self.enable_openai = enable_openai
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model

    def run(self):
        """Run text enhancement in background thread."""
        try:
            # Check if OpenAI API should be used
            if self.enable_openai and self.openai_api_key:
                print("🔄 Using OpenAI API for text enhancement")
                enhanced_prompt = self._run_openai_enhancement()
                print(f"✅ OpenAI enhancement succeeded: {enhanced_prompt[:100]}...")
            else:
                # Determine model type based on file extension
                is_gguf = self.model_path.lower().endswith('.gguf')

                if is_gguf:
                    # Use GGUF enhancement
                    enhanced_prompt = self._run_gguf_enhancement()
                    print(f"✅ GGUF enhancement succeeded: {enhanced_prompt[:100]}...")
                else:
                    # Use transformers enhancement
                    enhanced_prompt = self._run_transformers_enhancement()
                    print(f"✅ Transformers enhancement succeeded: {enhanced_prompt[:100]}...")

            self.finished.emit(enhanced_prompt)

        except Exception as e:
            error_msg = f"Text enhancement failed: {str(e)}"
            print(error_msg)
            self.error.emit(error_msg)

    def _run_transformers_enhancement(self):
        """Run enhancement using transformers library."""
        if not TRANSFORMERS_AVAILABLE:
            raise Exception("Transformers library not available. Please install transformers and torch.")

        # Load model and tokenizer
        print(f"Loading transformers text model from: {self.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True
        )

        # Create text generation pipeline
        text_generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

        # Prepare input prompt
        full_prompt = f"{self.enhancement_prompt}\n\nOriginal prompt: {self.user_prompt}\n\nCreate an enhanced version:"
        print(f"🔍 DEBUG: Full prompt sent to transformers model:\n{full_prompt}\n")

        # Generate enhanced prompt
        print("Generating enhanced prompt with transformers...")
        outputs = text_generator(
            full_prompt,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )

        # Extract the enhanced prompt (remove the input part)
        generated_text = outputs[0]['generated_text']
        enhanced_prompt = generated_text.replace(full_prompt, "").strip()

        # Clean up
        del text_generator
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return enhanced_prompt

    def _run_openai_enhancement(self):
        """Run enhancement using OpenAI API."""
        if not OPENAI_AVAILABLE:
            raise Exception("OpenAI library not available. Please install openai.")

        if not self.openai_api_key:
            raise Exception("OpenAI API key not provided.")

        try:
            # Initialize OpenAI client
            client = OpenAI(api_key=self.openai_api_key)

            # Prepare the prompt for OpenAI
            system_prompt = self.enhancement_prompt
            user_message = f"User prompt: {self.user_prompt}\n\nPlease enhance this prompt for AI image generation."

            print("Sending request to OpenAI API...")

            # Make API call
            response = client.chat.completions.create(
                model=self.openai_model,  # Use selected OpenAI model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=200,
                temperature=0.7,
                top_p=0.9
            )

            # Extract the enhanced prompt
            enhanced_prompt = response.choices[0].message.content.strip()

            print("✅ OpenAI API call completed successfully!")
            return enhanced_prompt

        except Exception as e:
            error_msg = f"OpenAI API enhancement failed: {str(e)}"
            print(f"OpenAI API error: {error_msg}")
            raise Exception(f"OpenAI API enhancement failed: {error_msg}")

    def _run_gguf_enhancement(self):
        """Run enhancement using llama-cpp-python for GGUF models via subprocess."""
        print(f"Loading GGUF text model from: {self.model_path}")

        # Use subprocess approach to avoid Qt threading issues
        try:
            from gguf_subprocess_enhancer import enhance_with_gguf_subprocess

            print("Running GGUF enhancement via subprocess to avoid Qt threading issues...")

            # Call the subprocess enhancer
            enhanced_prompt = enhance_with_gguf_subprocess(
                self.model_path,
                self.user_prompt,
                self.enhancement_prompt
            )

            print("✅ GGUF enhancement completed via subprocess!")
            return enhanced_prompt

        except Exception as e:
            error_msg = f"GGUF subprocess enhancement failed: {str(e)}"
            print(f"GGUF subprocess error: {error_msg}")
            raise Exception(f"GGUF enhancement failed via subprocess: {error_msg}")




class TextEnhancerService:
    """Service for text prompt enhancement using local language models."""

    def __init__(self, settings_manager):
        self.settings_manager = settings_manager
        self.current_model_path: Optional[str] = None
        self.last_used_time = 0
        self.cleanup_timer: Optional[threading.Timer] = None
        self.is_loading = False
        self.current_worker: Optional[TextEnhancementWorker] = None  # Track current worker thread

    def enhance_prompt_async(self, model_path: str, enhancement_prompt: str, user_prompt: str,
                           on_success: Callable[[str], None], on_error: Callable[[str], None],
                           enable_openai: bool = False, openai_api_key: str = None, openai_model: str = "gpt-3.5-turbo") -> bool:
        """
        Enhance a prompt asynchronously using a local text model.

        Args:
            model_path: Path to the text model
            enhancement_prompt: Instructions for how to enhance prompts
            user_prompt: The user's original prompt
            on_success: Callback for successful enhancement
            on_error: Callback for errors

        Returns:
            bool: True if enhancement started, False if failed to start
        """
        # Check if we have the required libraries for the model type
        is_gguf = model_path.lower().endswith('.gguf')
        if is_gguf and not LLAMA_CPP_AVAILABLE:
            on_error("llama-cpp-python library not available. Please install llama-cpp-python for GGUF support.")
            return False
        elif not is_gguf and not TRANSFORMERS_AVAILABLE:
            on_error("Transformers library not available. Please install transformers and torch.")
            return False

        # Validate model path/identifier
        if is_gguf:
            # For GGUF files, check if the file exists
            if not os.path.exists(model_path):
                on_error(f"GGUF model file not found at: {model_path}")
                return False
        else:
            # For transformers models, just validate the identifier format
            # The actual model will be downloaded if not present locally
            if not model_path or not model_path.strip():
                on_error("Text model identifier is empty")
                return False
            # Basic validation for Hugging Face model identifiers
            if not any(char in model_path for char in ['/', '-']) and len(model_path.split()) > 1:
                on_error(f"Invalid model identifier format: {model_path}")
                return False

        if self.is_loading:
            on_error("Text enhancement already in progress.")
            return False

        # Cancel any pending cleanup
        if self.cleanup_timer and self.cleanup_timer.is_alive():
            self.cleanup_timer.cancel()

        # Cancel any existing worker
        if self.current_worker and self.current_worker.isRunning():
            print("Cancelling previous text enhancement worker...")
            self.current_worker.requestInterruption()
            self.current_worker.wait(2000)  # Wait up to 2 seconds
            if self.current_worker.isRunning():
                self.current_worker.terminate()
                self.current_worker.wait(1000)

        # Start enhancement worker
        self.is_loading = True
        self.current_worker = TextEnhancementWorker(
            model_path, enhancement_prompt, user_prompt,
            enable_openai, openai_api_key, openai_model
        )

        def on_worker_finished(enhanced_prompt: str):
            self.is_loading = False
            self.last_used_time = time.time()
            self.current_model_path = model_path
            self.current_worker = None  # Clear reference
            self._schedule_cleanup()
            on_success(enhanced_prompt)

        def on_worker_error(error_msg: str):
            self.is_loading = False
            self.current_worker = None  # Clear reference
            on_error(error_msg)

        self.current_worker.finished.connect(on_worker_finished)
        self.current_worker.error.connect(on_worker_error)
        self.current_worker.start()

        return True

    def unload_model(self):
        """Force unload the current text model and clear memory."""
        print("Unloading text model...")
        self.current_model_path = None
        self.last_used_time = 0

        # Cancel cleanup timer
        if self.cleanup_timer and self.cleanup_timer.is_alive():
            self.cleanup_timer.cancel()
            self.cleanup_timer = None

        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _schedule_cleanup(self):
        """Schedule automatic cleanup after inactivity."""
        settings = self.settings_manager.load_settings()
        if settings.text_model_timeout_enabled:
            if self.cleanup_timer and self.cleanup_timer.is_alive():
                self.cleanup_timer.cancel()

            timeout_seconds = settings.text_model_timeout_minutes * 60
            self.cleanup_timer = threading.Timer(timeout_seconds, self._cleanup_task)
            self.cleanup_timer.daemon = True
            self.cleanup_timer.start()

    def _cleanup_task(self):
        """Cleanup task that runs after inactivity timeout."""
        if time.time() - self.last_used_time >= 180:  # 3 minutes
            print("Text model cleanup: 3 minutes of inactivity reached")
            self.unload_model()

    def is_model_loaded(self) -> bool:
        """Check if a text model is currently loaded."""
        return self.current_model_path is not None

    def cancel_enhancement(self) -> bool:
        """Cancel the current text enhancement operation."""
        if not self.current_worker or not self.current_worker.isRunning():
            return False

        print("Cancelling text enhancement...")
        self.current_worker.requestInterruption()
        self.current_worker.wait(2000)  # Wait up to 2 seconds for clean shutdown

        if self.current_worker.isRunning():
            print("Force terminating text enhancement thread...")
            self.current_worker.terminate()
            self.current_worker.wait(1000)

        # Reset state
        self.is_loading = False
        self.current_worker = None

        return True

    def get_time_since_last_use(self) -> float:
        """Get seconds since text model was last used."""
        if self.last_used_time == 0:
            return float('inf')
        return time.time() - self.last_used_time
