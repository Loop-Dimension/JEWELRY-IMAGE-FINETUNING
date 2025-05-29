import os
import streamlit as st
from PIL import Image, UnidentifiedImageError
import time
from glob import glob
import shutil
import sys
import torch
import asyncio
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import random
import subprocess
import json
import tempfile

# --- Configuration ---
APP_TITLE = "Jewelry Image Generator & Fine-Tuner"
APP_ICON = "💎"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUTS_DIR = os.path.join(ROOT_DIR, "inputs")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
BASE_MODELS_DIR = os.path.join(ROOT_DIR, "base_models") # Folder containing base models

# Subfolder for LoRA weights within each product's input directory
MODEL_SUBFOLDER = "model"

# Base model configuration (using local path)
BASE_MODEL_FOLDER_NAME = "stable-diffusion-v1-5" # The folder name created by git clone
BASE_MODEL_PATH = os.path.join(BASE_MODELS_DIR, BASE_MODEL_FOLDER_NAME)
BASE_MODEL_REPO_ID = "sd-legacy/stable-diffusion-v1-5" # For user instructions

# LoRA configuration
LORA_WEIGHT_NAME = "pytorch_lora_weights.safetensors" # Standard filename saved by diffusers script

# Fine-Tuning configuration
DIFFUSERS_REPO_PATH = os.path.join(ROOT_DIR, "diffusers_repo") # Assumes diffusers repo cloned here
TRAIN_SCRIPT_REL_PATH = "examples/text_to_image/train_text_to_image_lora.py"
TRAIN_SCRIPT_PATH = os.path.join(DIFFUSERS_REPO_PATH, TRAIN_SCRIPT_REL_PATH)
GENERIC_PROMPT = "a beautiful piece of jewelry" # Default prompt for training metadata

# --- Setup ---
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide"
)

# Create base directories if they don't exist
os.makedirs(INPUTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(BASE_MODELS_DIR, exist_ok=True) # Ensure base models dir exists

# Initialize PyTorch with proper error handling and event loop management
try:
    # Set up event loop if not already running
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Initialize PyTorch
    torch.manual_seed(42)
    if torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.float16
        torch.cuda.manual_seed_all(42)
        st.sidebar.success("GPU detected. Using CUDA.")
    else:
        device = "cpu"
        torch_dtype = torch.float32
        st.sidebar.warning("No GPU detected. Using CPU (will be significantly slower).")
except Exception as e:
    st.sidebar.error(f"Error initializing PyTorch: {e}. Defaulting to CPU.")
    device = "cpu"
    torch_dtype = torch.float32

# --- Helper Functions ---

def get_client_folders():
    """Gets a sorted list of client folders from the inputs directory."""
    try:
        folders = [f for f in os.listdir(INPUTS_DIR)
                   if os.path.isdir(os.path.join(INPUTS_DIR, f))]
        return sorted(folders)
    except FileNotFoundError:
        st.error(f"Inputs directory not found: {INPUTS_DIR}")
        return []
    except Exception as e:
        st.error(f"Error listing client folders: {e}")
        return []

def create_client_folder(folder_name):
    """Creates a new client folder in the inputs directory."""
    # Sanitize folder name
    safe_folder_name = "".join(c for c in folder_name if c.isalnum() or c in (' ', '_', '-')).rstrip().strip()
    if not safe_folder_name:
        st.error("Invalid folder name (must contain alphanumeric characters).")
        return None
    new_folder_path = os.path.join(INPUTS_DIR, safe_folder_name)
    try:
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)
            st.success(f"Created client folder: {safe_folder_name}")
            return safe_folder_name
        else:
            st.error(f"Folder '{safe_folder_name}' already exists!")
            return None
    except Exception as e:
        st.error(f"Failed to create client folder '{safe_folder_name}': {e}")
        return None

def get_product_folders(client_folder):
    """Gets a sorted list of product folders for a given client in the inputs directory."""
    client_path = os.path.join(INPUTS_DIR, client_folder)
    if not os.path.exists(client_path):
        return []
    try:
        folders = [f for f in os.listdir(client_path)
                   if os.path.isdir(os.path.join(client_path, f))]
        return sorted(folders)
    except Exception as e:
        st.error(f"Error listing product folders for '{client_folder}': {e}")
        return []

def create_product_folder(client_folder, product_name):
    """
    Creates matching product folders in input and output directories,
    including the 'model' subfolder in the input path.
    """
    # Sanitize product name
    safe_product_name = "".join(c for c in product_name if c.isalnum() or c in (' ', '_', '-')).rstrip().strip()
    if not safe_product_name:
        st.error("Invalid product name (must contain alphanumeric characters).")
        return None

    input_product_path = os.path.join(INPUTS_DIR, client_folder, safe_product_name)
    output_product_path = os.path.join(OUTPUTS_DIR, client_folder, safe_product_name)
    model_dir_path = os.path.join(input_product_path, MODEL_SUBFOLDER) # Path for model weights

    success = True
    paths_to_create = [input_product_path, output_product_path, model_dir_path]

    for path in paths_to_create:
        try:
            # Use exist_ok=True to prevent errors if the directory already exists
            # and to create parent directories if needed.
            os.makedirs(path, exist_ok=True)
        except Exception as e:
            st.error(f"Failed to create directory: {path}. Error: {e}")
            success = False
            break # Stop if one creation fails

    if success:
        st.success(f"Created product structure for: '{safe_product_name}'")
        st.info(f"- Input/Training Images: `{input_product_path}`")
        st.info(f"- Output Designs: `{output_product_path}`")
        st.info(f"- LoRA Model Weights: `{model_dir_path}`")
        return safe_product_name
    else:
        st.error(f"Failed to create one or more required folders for '{safe_product_name}'!")
        # Attempt cleanup (optional)
        for path in paths_to_create:
            if os.path.exists(path) and path.startswith(INPUTS_DIR) or path.startswith(OUTPUTS_DIR):
                try: shutil.rmtree(path)
                except: pass # Ignore cleanup errors
        return None

def count_images_in_folder(folder_path):
    """Counts image files (common extensions) in a specified folder."""
    if not os.path.isdir(folder_path):
        return 0
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']
    image_files = []
    try:
        for ext in image_extensions:
            # Use recursive=False if you only want top-level images
            image_files.extend(glob(os.path.join(folder_path, ext)))
        return len(image_files)
    except Exception as e:
        st.warning(f"Error counting images in {folder_path}: {e}")
        return 0

def display_folder_images(folder_path, columns=4, caption_prefix=""):
    """Displays images from a folder in responsive columns."""
    if not os.path.isdir(folder_path):
        st.info(f"{caption_prefix} folder not found or is empty: `{os.path.basename(folder_path)}`")
        return

    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']
    image_files = []
    try:
        for ext in image_extensions:
            image_files.extend(glob(os.path.join(folder_path, ext)))
    except Exception as e:
        st.error(f"Error reading images from {folder_path}: {e}")
        return

    if not image_files:
        st.info(f"No images found in {caption_prefix} folder: `{os.path.basename(folder_path)}`")
        return

    st.write(f"Number of images found: {len(image_files)}")
    cols = st.columns(columns)
    # Sort reverse=True to show newest files first based on modification time or name
    # Sorting by name is more deterministic here
    for i, image_path in enumerate(sorted(image_files, reverse=True)):
        try:
            with cols[i % columns]:
                img = Image.open(image_path)
                st.image(img, caption=os.path.basename(image_path), use_container_width=True)
        except UnidentifiedImageError:
             st.error(f"Could not load image (unsupported format?): {os.path.basename(image_path)}", icon="❓")
        except Exception as e:
            st.error(f"Error loading image {os.path.basename(image_path)}: {e}", icon="⚠️")


# --- Core Application Logic ---

@st.cache_resource(max_entries=5) # Cache loaded pipelines (base + LoRA applied)
def load_pipeline(lora_weights_folder):
    """
    Loads the base SD v1.5 pipeline from a LOCAL path and applies LoRA weights
    from the specified folder.

    Args:
        lora_weights_folder (str): Path to the folder containing the LoRA weights file
                                   (e.g., inputs/Client/Product/model).

    Returns:
        DiffusionPipeline or None: The loaded pipeline with LoRA applied, or None on failure.
    """
    # --- Check if the local base model path exists ---
    if not os.path.isdir(BASE_MODEL_PATH):
        st.error(f"Base model not found locally at: '{BASE_MODEL_PATH}'", icon="🚨")
        st.error(f"Please download the base model ('{BASE_MODEL_REPO_ID}') into that directory.")
        st.error(f"Example: `git clone https://huggingface.co/{BASE_MODEL_REPO_ID} {BASE_MODEL_PATH}`")
        st.error("Ensure the clone completes WITHOUT 'checkout failed' warnings.")
        return None

    st.info(f"Loading base model from local path: {BASE_MODEL_PATH}")
    try:
        # 1. Load the base pipeline FROM LOCAL PATH
        pipeline = DiffusionPipeline.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch_dtype,
            # Add safety_checker=None etc. if needed and trusted
        )
        # Use a potentially better scheduler
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to(device)
        st.success(f"Base model loaded successfully from local path to {device.upper()}!")

    except OSError as e:
         # Specific check for missing files often due to incomplete clone
         if "model_index.json" in str(e) or ".bin" in str(e) or ".safetensors" in str(e):
             st.error(f"OS error loading base model (essential file likely missing): {BASE_MODEL_PATH}. Error: {e}", icon="🚨")
             st.error("This often happens if the `git clone` had 'checkout failed' errors. Please delete the folder and re-clone carefully.")
         else:
             st.error(f"OS error loading base model (check path and permissions): {BASE_MODEL_PATH}. Error: {e}", icon="🚨")
         st.exception(e)
         return None
    except Exception as e:
        st.error(f"Failed to load the BASE diffusion pipeline from local path: {BASE_MODEL_PATH}. Error: {e}", icon="🚨")
        st.exception(e)
        return None

    # 2. Check and Load LoRA weights
    lora_weights_path = os.path.join(lora_weights_folder, LORA_WEIGHT_NAME)
    st.info(f"Looking for LoRA weights at: {lora_weights_path}")

    if not os.path.isdir(lora_weights_folder):
         st.error(f"LoRA directory specified does not exist: {lora_weights_folder}", icon="🚨")
         st.info("Proceeding with base model only.")
         return pipeline # Option: return base model if LoRA dir is invalid

    if not os.path.exists(lora_weights_path):
        st.warning(f"LoRA weights file '{LORA_WEIGHT_NAME}' not found in '{lora_weights_folder}'. Fine-tuning not applied.", icon="⚠️")
        st.info("Proceeding with base model only.")
        return pipeline # Return base model if LoRA file is missing

    try:
        # load_lora_weights typically takes the directory or the specific file path
        # Passing the folder is generally preferred as it might load related configs if present.
        st.info(f"Applying LoRA weights from: {lora_weights_folder}")
        pipeline.load_lora_weights(lora_weights_folder, weight_name=LORA_WEIGHT_NAME)
        st.success(f"Successfully loaded and applied LoRA weights from {os.path.basename(lora_weights_folder)}!")
        return pipeline # Return the pipeline with LoRA applied

    except Exception as e:
        st.error(f"Failed to load LoRA weights from {lora_weights_folder}. Error: {e}", icon="🚨")
        st.exception(e)
        st.info("Proceeding with base model only due to LoRA load error.")
        # Attempt to unload LoRA if partially applied, although this might not be standard
        try: pipeline.unload_lora_weights()
        except: pass
        return pipeline # Option: Fallback to base model


def generate_images(pipeline, prompt, negative_prompt, num_images, guidance_scale, num_steps, seed, output_folder):
    """
    Generates images using the provided pipeline and parameters.

    Args:
        pipeline (DiffusionPipeline): The loaded pipeline (potentially with LoRA).
        prompt (str): The text prompt.
        negative_prompt (str): The negative text prompt.
        num_images (int): Number of images to generate.
        guidance_scale (float): Classifier-Free Guidance scale.
        num_steps (int): Number of inference steps.
        seed (int): Seed for reproducibility. If -1, random seeds are used per image.
        output_folder (str): Directory to save generated images.

    Returns:
        bool: True if generation and saving were successful, False otherwise.
    """
    if pipeline is None:
        st.error("Pipeline is not available. Cannot generate images.")
        return False

    os.makedirs(output_folder, exist_ok=True)
    generated_files_count = 0
    image_placeholder = st.empty() # Placeholder to show generating status/image

    # Determine seed strategy
    use_random_seed_per_image = (seed == -1)
    if not use_random_seed_per_image:
        st.info(f"Using fixed seed for all images in this batch: {seed}")
        master_generator = torch.Generator(device=device).manual_seed(seed)
    else:
        st.info("Using a new random seed for each image in this batch.")
        master_generator = None # Will create new generator each time

    try:
        with st.spinner(f"Generating {num_images} image(s)..."):
            start_time_total = time.time()
            for i in range(num_images):
                start_time_img = time.time()

                # Get generator for this image
                if use_random_seed_per_image:
                    current_seed = random.randint(0, 2**32 - 1) # Standard 32-bit seed range
                    generator = torch.Generator(device=device).manual_seed(current_seed)
                    seed_info = f"(Seed: {current_seed})"
                else:
                    # Use the single master generator for reproducibility across batch
                    generator = master_generator
                    current_seed = seed # Log the master seed
                    seed_info = f"(Master Seed: {current_seed})"

                image_placeholder.text(f"Generating image {i+1}/{num_images} {seed_info}...")

                # Perform inference
                with torch.inference_mode():
                    result = pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=num_steps,
                        guidance_scale=guidance_scale,
                        num_images_per_prompt=1, # Generate one at a time
                        generator=generator
                    )
                    # Check for safety checker results if enabled
                    if result.nsfw_content_detected is not None and any(result.nsfw_content_detected):
                        st.warning(f"Potential NSFW content detected in image {i+1}. Skipping save.", icon="🔞")
                        continue # Skip saving this image

                    image = result.images[0]

                # Save the image
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_filename = f"gen_{timestamp}_{i+1}_s{current_seed}.png"
                output_path = os.path.join(output_folder, output_filename)
                try:
                    image.save(output_path, "PNG")
                    generated_files_count += 1
                    end_time_img = time.time()
                    # Show the generated image immediately in the placeholder
                    image_placeholder.image(image, caption=f"Generated: {output_filename} ({end_time_img - start_time_img:.2f}s)", use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to save generated image {output_filename}. Error: {e}")

            end_time_total = time.time()

        if generated_files_count > 0:
            # Clear the placeholder after loop finishes if needed, or keep last image
            # image_placeholder.empty()
            st.success(f"Successfully generated and saved {generated_files_count} image(s) in {end_time_total - start_time_total:.2f} seconds!")
            return True
        elif num_images > 0:
             st.warning("No images were generated or saved (check logs/safety warnings).")
             return False
        else:
             return True # No images requested is technically success

    except torch.cuda.OutOfMemoryError:
        st.error("CUDA out of memory! Try reducing image resolution/batch size or ensure sufficient VRAM.", icon="🔥")
        return False
    except Exception as e:
        st.error(f"An error occurred during image generation: {e}", icon="🚨")
        st.exception(e)
        return False

# --- Fine-Tuning Functions ---

def prepare_training_data(input_image_dir, temp_prepare_dir, generic_prompt_text):
    """
    Copies images from input_image_dir to temp_prepare_dir and creates metadata.jsonl.
    Skips non-image files.

    Args:
        input_image_dir (str): Path to the directory containing source training images.
        temp_prepare_dir (str): Path to the temporary directory to prepare data in.
        generic_prompt_text (str): The caption to assign to all images in metadata.

    Returns:
        str or None: The path to the prepared directory containing images and metadata.jsonl,
                     or None if preparation failed.
    """
    st.info(f"Preparing data from: {input_image_dir}")
    os.makedirs(temp_prepare_dir, exist_ok=True)
    metadata_filename = "metadata.jsonl"
    metadata_path = os.path.join(temp_prepare_dir, metadata_filename)
    # Common image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
    images_copied = 0
    metadata_entries = []

    try:
        files = os.listdir(input_image_dir)
        if not files:
            st.error(f"No files found in the input image directory: `{input_image_dir}`")
            return None

        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Copying images and creating metadata...")

        for i, filename in enumerate(files):
            file_base, file_ext = os.path.splitext(filename)
            if file_ext.lower() in image_extensions:
                input_image_path = os.path.join(input_image_dir, filename)
                output_image_path = os.path.join(temp_prepare_dir, filename)
                try:
                    # Try opening image to verify it's valid before copying
                    with Image.open(input_image_path) as img:
                        img.verify() # Verify headers, structure
                    # Copy the valid image
                    shutil.copy2(input_image_path, output_image_path)
                    # Create metadata entry using relative filename
                    metadata_entry = {"file_name": filename, "text": generic_prompt_text}
                    metadata_entries.append(metadata_entry)
                    images_copied += 1
                except FileNotFoundError:
                    st.warning(f"Skipping {filename}: File not found (likely removed during process).")
                except (IOError, SyntaxError, UnidentifiedImageError) as img_err:
                    st.warning(f"Skipping invalid/corrupt image file {filename}: {img_err}", icon="🖼️")
                except Exception as copy_err:
                    st.warning(f"Could not copy image {filename}: {copy_err}")
            # Update progress bar regardless of file type
            progress_bar.progress((i + 1) / len(files))

        status_text.text(f"Finished processing {len(files)} files.")

        if not metadata_entries:
            st.error("No valid image files were found or processed in the input directory.")
            return None

        # Write metadata file
        with open(metadata_path, 'w') as f:
            for entry in metadata_entries:
                f.write(json.dumps(entry) + '\n')

        st.success(f"Prepared {images_copied} valid images with metadata in temporary directory.")
        return temp_prepare_dir

    except FileNotFoundError:
        st.error(f"Input image directory not found: `{input_image_dir}`")
        return None
    except Exception as e:
        st.error(f"Error during data preparation: {e}")
        return None


def run_lora_training(params, log_expander):
    """
    Constructs and runs the accelerate launch command for LoRA training using subprocess.

    Args:
        params (dict): Dictionary containing training parameters.
                       Expected keys match the arguments of the training script.
        log_expander (st.expander): Streamlit expander to display logs within.

    Returns:
        bool: True if the training process exited with code 0, False otherwise.
    """
    log_area = log_expander.empty() # Create placeholder inside expander

    # Check if training script exists
    if not os.path.isfile(TRAIN_SCRIPT_PATH):
        log_area.error(f"Training script not found at: `{TRAIN_SCRIPT_PATH}`")
        log_area.error("Please ensure the 'diffusers' repository is cloned correctly (`diffusers_repo` folder) relative to the app.")
        return False

    # Get the directory containing the script for `cwd`
    script_directory = os.path.dirname(TRAIN_SCRIPT_PATH)

    # Construct the command arguments list
    # Using BASE_MODEL_PATH which points to local base model
    command_list = [
        "accelerate", "launch", TRAIN_SCRIPT_PATH,
        f"--pretrained_model_name_or_path={params['base_model_path']}",
        f"--train_data_dir={params['prepared_data_dir']}",
        "--image_column=file_name", # Matches key in metadata.jsonl
        "--caption_column=text",    # Matches key in metadata.jsonl
        f"--dataloader_num_workers={params['dataloader_num_workers']}",
        f"--resolution={params['resolution']}", "--center_crop", "--random_flip",
        f"--train_batch_size={params['train_batch_size']}",
        f"--gradient_accumulation_steps={params['gradient_accumulation_steps']}",
        f"--mixed_precision={params['mixed_precision']}",
        f"--rank={params['lora_rank']}",
        f"--max_train_steps={params['max_train_steps']}",
        f"--learning_rate={params['learning_rate']}",
        "--max_grad_norm=1",
        f"--lr_scheduler={params['lr_scheduler']}",
        f"--lr_warmup_steps={params['lr_warmup_steps']}",
        f"--output_dir={params['temp_output_dir']}", # Train in temp dir
        f"--seed={params['seed']}",
        "--report_to=tensorboard", # Or "none"
        # Validation prompt helps visualize progress during training
        f"--validation_prompt={GENERIC_PROMPT}",
        f"--validation_epochs=5", # How often to run validation
        # Save checkpoints periodically
        f"--checkpointing_steps={params['checkpointing_steps']}",
        "--resume_from_checkpoint=latest" # Automatically resume if checkpoint exists
    ]

    # Add boolean flags if they are True
    if params.get('gradient_checkpointing', False):
        command_list.append("--gradient_checkpointing")
    if params.get('use_8bit_adam', False):
        command_list.append("--use_8bit_adam")

    # Log the command being run
    log_content = "--- Training Command ---\n"
    log_content += " ".join(command_list) + "\n"
    log_content += "----------------------\n"
    log_content += f"Running from directory: {script_directory}\n"
    log_content += "Starting training process... (This can take a long time)\n\n"
    log_area.text_area("Training Log", value=log_content, height=400, key="log_area_initial")

    # Run the process using subprocess.Popen for real-time output
    process = None
    try:
        process = subprocess.Popen(
            command_list,
            cwd=script_directory, # Run script from its own directory
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # Redirect stderr to stdout
            text=True,
            bufsize=1, # Line buffered
            encoding='utf-8',
            errors='replace' # Replace undecodable characters
        )

        # Stream output to the log area
        while True:
            line = process.stdout.readline()
            if not line:
                break # Process finished
            log_content += line
            # Update the log area dynamically
            log_area.text_area("Training Log", value=log_content, height=400, key="log_area_update")

        process.stdout.close()
        return_code = process.wait() # Wait for the process to terminate completely

        if return_code == 0:
            log_content += "\n--- Training Finished Successfully ---\n"
            log_area.text_area("Training Log", value=log_content, height=400, key="log_area_final_ok")
            st.success("Training process completed successfully!")
            return True
        else:
            log_content += f"\n--- Training Failed (Exit Code: {return_code}) ---\n"
            log_area.text_area("Training Log", value=log_content, height=400, key="log_area_final_err")
            st.error(f"Training process failed with return code: {return_code}")
            return False

    except FileNotFoundError:
        log_content += "\n--- ERROR: 'accelerate' command not found. Is Accelerate installed and configured? ---"
        log_area.text_area("Training Log", value=log_content, height=400, key="log_area_accel_err")
        st.error("'accelerate' command not found. Please ensure it's installed and in your PATH.")
        return False
    except Exception as e:
        log_content += f"\n--- An unexpected error occurred launching training: {e} ---"
        log_area.text_area("Training Log", value=log_content, height=400, key="log_area_launch_err")
        st.error(f"An unexpected error occurred launching training: {e}")
        st.exception(e)
        # Ensure process is terminated if it started
        if process and process.poll() is None:
            process.terminate()
            process.wait()
        return False
    finally:
        # Ensure stdout is closed if process exists
        if process and process.stdout and not process.stdout.closed:
            process.stdout.close()


# --- Streamlit UI ---
def main():
    """Main function to run the Streamlit application."""
    st.title(f"{APP_ICON} {APP_TITLE}")

    # --- Sidebar ---
    st.sidebar.header("Client & Product Management")

    # Client Selection/Creation
    st.sidebar.subheader("Client")
    client_folders = get_client_folders()
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        new_client_name = st.text_input("New Client Name", key="new_client_input", placeholder="Enter client name...")
    with col2:
        # Add vertical space to align button
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Create", key="create_client_button", use_container_width=True):
            if new_client_name:
                created_client = create_client_folder(new_client_name)
                if created_client:
                    st.success(f"Client '{created_client}' created.")
                    # Optionally set created client as selected? Or just rerun.
                    st.rerun()
            else:
                st.sidebar.warning("Please enter a client name.")

    # Client Selection Dropdown
    if not client_folders:
        st.info("No client folders found. Create a client folder using the sidebar to begin.")
        st.stop() # Stop execution if no clients exist

    selected_client_index = 0 # Default index
    # Try to preserve selection across reruns using session state
    if 'client_select' in st.session_state and st.session_state.client_select in client_folders:
        selected_client_index = client_folders.index(st.session_state.client_select)

    selected_client = st.sidebar.selectbox(
        "Select Client",
        client_folders,
        index=selected_client_index,
        key="client_select" # Use key to access value in session state
    )

    # Product Selection/Creation
    st.sidebar.subheader(f"Product for '{selected_client}'")
    product_folders = get_product_folders(selected_client) # Get products for the selected client
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        new_product_name = st.text_input("New Product Name", key="new_product_input", placeholder="Enter product name...")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Create", key="create_product_button", use_container_width=True):
            if new_product_name:
                # Pass selected client to create function
                created_product = create_product_folder(selected_client, new_product_name)
                if created_product:
                     st.success(f"Product '{created_product}' created for client '{selected_client}'.")
                     st.rerun() # Refresh product list
            else:
                st.sidebar.warning("Enter a product name.")

    # Product Selection Dropdown
    selected_product = None
    lora_weights_folder = None # Path for LoRA weights (inputs/.../model)
    pipeline = None # Holds the loaded pipeline

    if product_folders:
        selected_product_index = 0
        # Preserve selection across reruns
        if 'product_select' in st.session_state and st.session_state.product_select in product_folders:
             selected_product_index = product_folders.index(st.session_state.product_select)

        selected_product = st.sidebar.selectbox(
            "Select Product",
            product_folders,
            index=selected_product_index,
            key="product_select"
        )
        # --- Dynamic Model Loading ---
        if selected_product:
            lora_weights_folder = os.path.join(INPUTS_DIR, selected_client, selected_product, MODEL_SUBFOLDER)
            st.sidebar.caption(f"LoRA Path: `.../{selected_client}/{selected_product}/{MODEL_SUBFOLDER}`")
            # Attempt to load pipeline (will be cached based on lora_weights_folder)
            # The key for caching is the folder path, so it reloads if product changes
            pipeline = load_pipeline(lora_weights_folder)
            # Error/warning messages are handled within load_pipeline

    else:
        st.sidebar.info(f"No product folders found for '{selected_client}'. Create one to start.")

    # --- Main Area ---
    st.header(f"Workspace: `{selected_client}` / `{selected_product if selected_product else 'No Product Selected'}`")

    # Define tabs
    tab_titles = ["✨ Generate Image", "🖼️ Generated Designs", "⚙️ Fine-Tune Model (LoRA)"]
    tab1, tab2, tab3 = st.tabs(tab_titles)

    # --- Generation Tab (tab1 - Minimalist) ---
    with tab1:
        st.subheader("Generate New Jewelry Image")
        if selected_client and selected_product:
            if pipeline: # Check if pipeline (Base model, maybe + LoRA) is loaded
                # --- Hardcoded parameters (NOT shown to user) ---
                prompt = GENERIC_PROMPT # Use the consistent generic prompt
                # Keeping a standard negative prompt is usually beneficial
                neg_prompt = "low quality, blurry, text, watermark, signature, deformed, ugly, worst quality, lowres, jpeg artifacts, bad anatomy, extra limbs, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame"
                num_images = 1       # Generate one image per click
                guidance_scale = 7.5 # Default CFG scale
                num_steps = 30       # Default steps
                seed_value = -1      # Signal to use random seed in generate_images

                # --- ONLY the Generate button is displayed ---
                if st.button("Generate Image", type="primary", key="generate_button_minimal", use_container_width=True):
                    # Define output path for generated images
                    output_gen_path = os.path.join(OUTPUTS_DIR, selected_client, selected_product)
                    # Call the generation function
                    success = generate_images(
                        pipeline=pipeline,
                        prompt=prompt,
                        negative_prompt=neg_prompt,
                        num_images=num_images,
                        guidance_scale=guidance_scale,
                        num_steps=num_steps,
                        seed=seed_value,
                        output_folder=output_gen_path
                    )
                    if success:
                        st.balloons()
                        st.success("Generation complete! Check the 'Generated Designs' tab (might need a moment to update).")
                        # Optional: st.rerun() # To force immediate refresh of tab2, but loses logs/image in tab1
            else:
                # If pipeline failed to load (e.g., base model missing)
                st.error(f"Cannot generate images. The base model or LoRA for '{selected_product}' failed to load. Check sidebar messages and file paths.", icon="🚫")
                st.info("Ensure the base model is downloaded and LoRA file (if needed) exists.")
        else:
            st.warning("Please select a Client and Product in the sidebar to enable generation.")

    # --- Generated Designs Tab (tab2) ---
    with tab2:
        st.subheader("View Generated Designs")
        if selected_client and selected_product:
             output_product_path = os.path.join(OUTPUTS_DIR, selected_client, selected_product)
             st.write(f"Displaying images from: `{output_product_path}`")

             if os.path.exists(output_product_path):
                 # Use columns=5 for potentially better layout on wide screens
                 display_folder_images(output_product_path, columns=5, caption_prefix="Generated")

                 # --- Add download button for this product's generated images ---
                 num_generated = count_images_in_folder(output_product_path)
                 if num_generated > 0:
                     st.divider()
                     # Create a unique zip filename per client/product
                     zip_filename_prod = f"{selected_client}_{selected_product}_generated_designs.zip"
                     # Use a temporary location for the zip file (in ROOT_DIR for simplicity)
                     zip_base_name = os.path.join(ROOT_DIR, f"{selected_client}_{selected_product}_temp_zip")

                     try:
                         # Ensure old temp zip file is removed if it exists
                         temp_zip_path = zip_base_name + ".zip"
                         if os.path.exists(temp_zip_path):
                             os.remove(temp_zip_path)

                         # Create the new zip archive from the output product folder
                         shutil.make_archive(
                             base_name=zip_base_name,
                             format='zip',
                             root_dir=output_product_path # Zip contents of this folder
                         )

                         if os.path.exists(temp_zip_path):
                             with open(temp_zip_path, "rb") as fp:
                                 st.download_button(
                                     label=f"Download All {num_generated} Generated '{selected_product}' Images (.zip)",
                                     data=fp,
                                     file_name=zip_filename_prod,
                                     mime="application/zip",
                                     # Use a unique key combining client and product
                                     key=f"download_zip_{selected_client}_{selected_product}"
                                 )
                             # Consider adding a cleanup mechanism for the temp zip file,
                             # e.g., using streamlit-cleanup or deleting on session end/start.
                             # For now, it gets overwritten on next download action.
                         else:
                             st.error("Failed to create ZIP file for download (archive not found after creation attempt).")
                     except Exception as e:
                         st.error(f"Could not create or provide ZIP file for download. Error: {e}")
                         # Clean up potentially failed attempt
                         if os.path.exists(temp_zip_path):
                             try: os.remove(temp_zip_path)
                             except: pass # Ignore cleanup error
             else:
                 st.info(f"Output directory for '{selected_product}' does not exist yet. Generate images first.")
        else:
             st.warning("Please select a Client and Product in the sidebar to view generated designs.")

    # --- Fine-Tuning Tab (tab3) ---
    with tab3:
        st.subheader("Fine-Tune Base Model with LoRA")

        if not selected_client or not selected_product:
            st.warning("Please select a Client and Product in the sidebar first.")
            st.stop() # Don't show the rest if no client/product selected

        # Define paths based on current selection
        # Training images are expected directly in the product folder (NOT in /model)
        input_image_dir = os.path.join(INPUTS_DIR, selected_client, selected_product)
        # LoRA weights will be saved to the 'model' subfolder
        final_lora_output_dir = os.path.join(input_image_dir, MODEL_SUBFOLDER)

        st.info(f"This tool will fine-tune the base model (`{BASE_MODEL_FOLDER_NAME}`) using images found directly inside:")
        st.code(input_image_dir)
        st.info(f"The resulting LoRA weights file (`{LORA_WEIGHT_NAME}`) will be saved to:")
        st.code(final_lora_output_dir)
        st.warning("""
        **Important Notes:**
        - **GPU Required:** Fine-tuning is computationally intensive and requires a capable NVIDIA GPU with sufficient VRAM (e.g., T4, A10G, RTX 3090+ recommended). CPU training is impractically slow.
        - **Time:** Training can take hours depending on dataset size, steps, and hardware.
        - **Dependencies:** Ensure `accelerate`, `bitsandbytes`, `transformers`, `diffusers` (from source often needed), `torch` (with CUDA) are correctly installed in the environment running this app.
        - **`accelerate config`:** Run `accelerate config` in your terminal once to set up default settings (usually single GPU, no distributed training).
        - **Input Images:** Place your training images (JPG, PNG, WEBP) directly inside the input product folder shown above.
        """, icon="⚠️")

        # Display input images for confirmation
        st.markdown("---")
        st.subheader("Training Images Preview")
        num_train_images = count_images_in_folder(input_image_dir)
        if num_train_images > 0:
            st.write(f"Found {num_train_images} potential training images in `{input_image_dir}`.")
            display_folder_images(input_image_dir, columns=6, caption_prefix="Input") # Show more columns for preview
        else:
            st.error(f"No training images found directly in `{input_image_dir}`. Please add images to this folder before starting fine-tuning.", icon="🖼️")
            st.stop() # Stop if no images

        st.markdown("---")
        st.subheader("Training Parameters")

        # Use columns for better layout of parameters
        col_param1, col_param2 = st.columns(2)

        with col_param1:
            resolution = st.select_slider("Resolution", options=[512, 768], value=512, help="Image size for training. 512 is standard for SD v1.5. Ensure images are suitable or allow cropping.")
            max_train_steps = st.number_input("Max Training Steps", min_value=100, max_value=20000, value=1500, step=100, help=f"Total optimization steps. Adjust based on dataset size (~{num_train_images} images) & desired epochs. E.g., (Images / Effective_Batch_Size) * Epochs.")
            lora_rank = st.slider("LoRA Rank (Complexity)", min_value=4, max_value=128, value=16, step=4, help="Higher rank captures more detail but uses more memory/time & can overfit. 4-32 is common.")

        with col_param2:
            # Effective Batch Size = train_batch_size * gradient_accumulation_steps
            train_batch_size = st.select_slider("Batch Size (per GPU)", options=[1, 2, 4], value=1, help="Images per step per GPU. Keep low (1) for limited VRAM (e.g., <=16GB).")
            gradient_accumulation_steps = st.select_slider("Gradient Accumulation", options=[1, 2, 4, 8, 16, 32], value=4, help="Accumulate gradients over N steps to simulate larger batch size. Effective Batch Size = Batch Size * Accumulation.")
            learning_rate = st.select_slider("Learning Rate", options=[5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4], value=1e-4, help="How fast the model adapts. 1e-4 to 5e-5 is often a good starting range for LoRA.")

        st.caption(f"Effective Batch Size = {train_batch_size} * {gradient_accumulation_steps} = {train_batch_size * gradient_accumulation_steps}")

        # Advanced options in expander
        with st.expander("Advanced Training Options"):
            mixed_precision = st.selectbox("Mixed Precision", ["fp16", "bf16", "no"], index=0, help="Use 16-bit precision for speed/memory saving. 'fp16' widely supported, 'bf16' on newer GPUs (e.g., Ampere+). 'no' uses float32 (slow, high memory).")
            use_8bit_adam = st.checkbox("Use 8-bit Adam Optimizer", value=(device=="cuda"), help="Saves significant GPU memory during optimization (requires `bitsandbytes`). Recommended on GPU.")
            gradient_checkpointing = st.checkbox("Use Gradient Checkpointing", value=True, help="Saves GPU memory by recomputing some values during backward pass (slightly slower training). Recommended for larger models/limited VRAM.")
            lr_scheduler = st.selectbox("Learning Rate Scheduler", ["cosine", "linear", "constant", "constant_with_warmup"], index=0, help="How the learning rate changes over training steps.")
            lr_warmup_steps = st.number_input("Warmup Steps", min_value=0, value=0, step=50, help="Number of initial steps with linearly increasing LR (used with 'constant_with_warmup').")
            checkpointing_steps = st.number_input("Save Checkpoint Every N Steps", min_value=100, max_value=5000, value=500, step=100, help="Saves intermediate LoRA weights. Useful for resuming or finding best version.")
            seed = st.number_input("Training Seed", value=42, help="Seed for random number generators (shuffling, dropout) for reproducibility.")
            dataloader_num_workers = st.slider("Dataloader Workers", 0, os.cpu_count() // 2 if os.cpu_count() else 2, 2, help="Number of CPU processes to load data. 0 means data loaded in main process.")

        st.markdown("---")

        # Placeholder for logs
        log_expander = st.expander("Training Logs", expanded=False) # Start collapsed

        # Start Training Button
        if st.button("🚀 Start Fine-Tuning", type="primary", key="start_tuning_button", use_container_width=True):
            if device == "cpu":
                st.error("Fine-tuning requires a GPU (CUDA). Training cannot start on CPU.", icon="🚫")
            else:
                # Expand the log area when starting
                log_expander.expanded = True
                log_area_placeholder = log_expander.empty() # Use empty to replace content
                log_area_placeholder.info("Initializing training setup...")

                # Use temporary directories for prepared data and training output
                # These are cleaned up automatically when the 'with' block exits
                try:
                    with tempfile.TemporaryDirectory() as temp_prepare_dir, \
                         tempfile.TemporaryDirectory() as temp_output_dir:

                        log_area_placeholder.info(f"Preparing data in temporary directory: `{temp_prepare_dir}`")
                        # 1. Prepare Data (Copy images, create metadata.jsonl)
                        prepared_data_path = prepare_training_data(input_image_dir, temp_prepare_dir, GENERIC_PROMPT)

                        if prepared_data_path:
                            log_area_placeholder.info(f"Training output will be stored temporarily in: `{temp_output_dir}`")
                            # 2. Define parameters dictionary for the training function
                            training_params = {
                                "base_model_path": BASE_MODEL_PATH, # Use local base model
                                "prepared_data_dir": prepared_data_path,
                                "temp_output_dir": temp_output_dir,
                                "resolution": resolution,
                                "train_batch_size": train_batch_size,
                                "gradient_accumulation_steps": gradient_accumulation_steps,
                                "gradient_checkpointing": gradient_checkpointing,
                                "mixed_precision": mixed_precision,
                                "lora_rank": lora_rank,
                                "max_train_steps": max_train_steps,
                                "learning_rate": learning_rate,
                                "lr_scheduler": lr_scheduler,
                                "lr_warmup_steps": lr_warmup_steps,
                                "use_8bit_adam": use_8bit_adam,
                                "seed": seed,
                                "checkpointing_steps": checkpointing_steps,
                                "dataloader_num_workers": dataloader_num_workers,
                            }

                            # 3. Run Training via subprocess
                            # Pass the expander itself to the function
                            training_success = run_lora_training(training_params, log_expander)

                            # 4. Copy final weights if successful
                            if training_success:
                                # The script saves the final LoRA file at the root of output_dir
                                final_weights_temp_path = os.path.join(temp_output_dir, LORA_WEIGHT_NAME)
                                if os.path.exists(final_weights_temp_path):
                                    try:
                                        # Ensure the final destination directory exists
                                        os.makedirs(final_lora_output_dir, exist_ok=True)
                                        final_lora_destination = os.path.join(final_lora_output_dir, LORA_WEIGHT_NAME)

                                        log_expander.info(f"Copying final weights from `{final_weights_temp_path}` to `{final_lora_destination}`...")
                                        shutil.copy2(final_weights_temp_path, final_lora_destination)
                                        st.success(f"Successfully copied final LoRA weights to the product's model folder!")
                                        st.info("Fine-tuning complete. You can now use this model in the 'Generate Image' tab after re-selecting this product (to reload).")
                                        # Optionally clear the pipeline cache to force reload next time
                                        st.cache_resource.clear()
                                    except Exception as copy_err:
                                        st.error(f"Training succeeded, but failed to copy final LoRA weights: {copy_err}")
                                else:
                                    st.error(f"Training reported success, but final LoRA file '{LORA_WEIGHT_NAME}' was not found in the temporary output directory `{temp_output_dir}`. Check training logs.")
                            else:
                                st.error("Training failed. See logs in the expander above for details.")
                        else:
                            # Error message already shown by prepare_training_data
                            log_area_placeholder.error("Data preparation failed. Cannot start training.")
                except Exception as e:
                     st.error(f"An error occurred managing temporary directories or during the training process setup: {e}")
                     st.exception(e)


    # --- Sidebar Footer ---
    st.sidebar.divider()
    st.sidebar.header("Setup & Instructions")
    st.sidebar.info(f"""
    **One-Time Setup:**
    1. **Base Model:** Download SD v1.5 locally:
       ```
       git clone https://huggingface.co/{BASE_MODEL_REPO_ID} {BASE_MODEL_PATH}
       ```
       Ensure clone finishes **without errors**.
    2. **Diffusers Repo:** Clone for training script:
       ```
       git clone https://github.com/huggingface/diffusers.git {DIFFUSERS_REPO_PATH}
       ```
    3. **Dependencies:** Install all required libraries (`torch` with CUDA, `accelerate`, `bitsandbytes`, `transformers`, `diffusers` etc.) in your Python environment.
    4. **Accelerate Config:** Run `accelerate config` in terminal once.

    **Workflow:**
    1. **Create/Select Client & Product** in the sidebar.
    2. **Prepare:** Add training images (JPG/PNG etc.) directly into the `inputs/{{Client}}/{{Product}}/` folder.
    3. **Fine-Tune (Optional):** Use the **'⚙️ Fine-Tune'** tab. Adjust parameters, click 'Start'. Wait for completion (can take hours!). Final LoRA saved to `.../{{Product}}/model/`.
    4. **Generate:** Use the **'✨ Generate Image'** tab. Select Client/Product. If fine-tuned, the LoRA is loaded automatically. Click 'Generate Image'.
    5. **View/Download:** Check **'🖼️ Generated Designs'** tab.
    """)
    st.sidebar.divider()
    # Display Status Information
    st.sidebar.caption(f"Status | Device: {device.upper()} | Torch: {torch.__version__}")
    st.sidebar.caption(f"Base Model Path: `{BASE_MODEL_PATH}`")
    if lora_weights_folder:
        lora_exists = os.path.exists(os.path.join(lora_weights_folder, LORA_WEIGHT_NAME))
        st.sidebar.caption(f"LoRA Path: `{lora_weights_folder}` (Exists: {lora_exists})")
    if pipeline:
        st.sidebar.caption("Pipeline Status: Loaded (Base + LoRA if found)")
    elif selected_product:
         st.sidebar.caption("Pipeline Status: Failed to Load or LoRA not found")
    else:
         st.sidebar.caption("Pipeline Status: Not Loaded (Select Product)")


# --- Application Entry Point ---
if __name__ == "__main__":
    # Add basic check for diffusers installation
    try:
        import diffusers
        import accelerate
        import transformers
        import bitsandbytes # Check if installed, fine if not on CPU
    except ImportError as e:
        st.error(f"Missing required library: {e.name}. Please install all dependencies.", icon="🚨")
        st.error("Run: pip install -r requirements.txt (if you have one) or install manually.")
        st.stop()

    # Check if base model directory looks plausible (very basic check)
    if not os.path.exists(os.path.join(BASE_MODEL_PATH, "model_index.json")):
         st.warning(f"Base model indicator (`model_index.json`) not found in `{BASE_MODEL_PATH}`. Please ensure it's downloaded correctly.", icon="⚠️")

    # Check if training script exists
    if not os.path.exists(TRAIN_SCRIPT_PATH):
        st.warning(f"Diffusers training script not found at `{TRAIN_SCRIPT_PATH}`. Fine-tuning tab will fail. Ensure `diffusers_repo` is cloned.", icon="⚠️")

    main()
