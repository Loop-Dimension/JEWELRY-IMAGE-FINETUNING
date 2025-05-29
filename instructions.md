# 💎 Jewelry Image Generator & Fine-Tuner - Instructions

## Introduction

Welcome! This tool allows you to:

1.  **Generate** unique jewelry images using pre-trained AI models.
2.  **Fine-tune** the AI model on your own specific jewelry images (using LoRA technology) to create customized styles.
3.  **Manage** different jewelry styles or client projects in separate folders.

This guide will walk you through setting up and using the application.

## Prerequisites (What You Need)

1.  **Computer:**
    *   A reasonably modern computer (Windows, macOS, or Linux).
    *   **For Fine-Tuning:** A powerful **NVIDIA GPU** (like RTX 20xx, 30xx, 40xx series or professional equivalents like T4, A10G) with **at least 8-12GB of VRAM** is *highly recommended*. Fine-tuning on CPU is possible but extremely slow (days instead of hours). Generation can work on CPU but will also be much slower than on GPU.
    *   **Disk Space:** At least **20-30 GB** of free space for Python, libraries, the base AI model, the training script repository, and your generated images/fine-tuned models.
    *   **Internet:** Needed for initial download of libraries and models.
2.  **Software:**
    *   **Python:** Version 3.9, 3.10, or 3.11 recommended. ([Download Python](https://www.python.org/downloads/))
    *   **Git:** Needed to download the base model and training scripts. ([Download Git](https://git-scm.com/downloads/))

## Setup Steps (One-Time Only)

Follow these steps carefully to prepare your system.

**Step 1: Create a Project Folder**

*   Create a new folder on your computer where you will store the application code, models, and your jewelry images. Let's call it `JewelryAI`.
*   The `app.py` script should be placed inside this `JewelryAI` folder.
*   All subsequent commands should be run *inside* this `JewelryAI` folder using a terminal or command prompt.

**Step 2: Set Up Python Environment**

*   Open your terminal or command prompt (like Terminal on macOS/Linux, Command Prompt or PowerShell on Windows).
*   Navigate *into* your `JewelryAI` folder using the `cd` command. Example: `cd path/to/your/JewelryAI`
*   **Create a Virtual Environment (Recommended):** This keeps the project's libraries separate from your system's Python.
    ```
    python -m venv env
    ```
*   **Activate the Virtual Environment:**
    *   **Windows (Command Prompt):** `env\Scripts\activate.bat`
    *   **Windows (PowerShell):** `env\Scripts\Activate.ps1` (You might need to set execution policy: `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` first)
    *   **macOS / Linux:** `source env/bin/activate`
    *   You should see `(env)` appear at the beginning of your terminal prompt. Keep this terminal open and active for the following steps.

**Step 3: Download Application Code**

*   Save the Python script (`app.py` that runs the Streamlit application) inside your `JewelryAI` folder [1].
*   If you have a `requirements.txt` file listing necessary Python libraries, save it into the same `JewelryAI` folder.

**Step 4: Install Dependencies (Libraries)**

*   **Crucial - Install PyTorch (GPU or CPU):**
    *   Visit the official PyTorch website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
    *   Select your Operating System, Package (`Pip`), Language (`Python`), and Compute Platform (`CUDA` version if you have NVIDIA GPU, or `CPU`).
    *   Copy the generated installation command (it will look something like `pip3 install torch torchvision torchaudio --index-url ...` for CUDA, or `pip3 install torch torchvision torchaudio` for CPU).
    *   Run **that specific command** in your activated `(env)` terminal. This ensures you get the correct version for your hardware.
*   **Install Other Libraries:** Once PyTorch is installed, if you have a `requirements.txt` file, install the rest using:
    ```
    pip install -r requirements.txt
    ```
    If you don't have a `requirements.txt`, you will need to manually install libraries like `streamlit`, `diffusers`, `transformers`, `accelerate`, `Pillow`, `bitsandbytes` (for 8-bit Adam optimizer on GPU).
    *   *(Note: `bitsandbytes` might fail on Windows/macOS if not using WSL. The app will still function for generation, but the "8-bit Adam" optimizer option during fine-tuning won't be available or effective.)* [1]
*   **Configure `accelerate`:** This library helps launch the training script. Run its configuration command once in your terminal:
    ```
    accelerate config
    ```
    *   Answer the questions. For typical local use with a single GPU:
        *   Compute environment: `This machine` (usually option 0)
        *   Distributed type: `No` (usually option 0)
        *   (Answer No if asked about DeepSpeed/FP8/FSDP unless you know you need it)
        *   GPU selection: Choose `all` if you have one GPU, or specify comma-separated IDs if you have multiple and want to use specific ones.
        *   Mixed precision: Choose `fp16` (common) or `bf16` (newer GPUs) if your GPU supports it for faster training and less memory usage. Otherwise, choose `no`.

**Step 5: Download Base AI Model (Stable Diffusion v1.5)**

*   This model is the foundation. The application is configured to use `sd-legacy/stable-diffusion-v1-5` locally [1]. Run this command in your `(env)` terminal, inside the `JewelryAI` folder:
    ```
    git clone https://huggingface.co/sd-legacy/stable-diffusion-v1-5 base_models/stable-diffusion-v1-5
    ```
*   **Check for Errors:** Watch the output carefully. Make sure it completes **without** any `checkout failed` or other critical errors. This download can be large and take time.
*   This will create a folder structure like: `JewelryAI/base_models/stable-diffusion-v1-5/` containing model files [1].
*   **If `git clone` fails or gets stuck:** See the "Troubleshooting" section below for manually downloading base model files.

**Step 6: Download Training Script Repository (Diffusers)**

*   This contains the script needed for LoRA fine-tuning. Run this command in your `(env)` terminal, inside the `JewelryAI` folder:
    ```
    git clone https://github.com/huggingface/diffusers.git diffusers_repo
    ```
*   This creates `JewelryAI/diffusers_repo/` containing the necessary training code. The application expects the training script at `diffusers_repo/examples/text_to_image/train_text_to_image_lora.py` [1].

**Setup Complete!** You shouldn't need to repeat these steps unless you move to a new computer or need to significantly update libraries.

## Running the Application

1.  **Activate Environment:** Open a new terminal, navigate to your `JewelryAI` folder, and activate the virtual environment:
    *   Windows: `env\Scripts\activate.bat`
    *   macOS/Linux: `source env/bin/activate`
2.  **Run Streamlit:** Start the application using (ensure `app.py` is in your `JewelryAI` folder):
    ```
    streamlit run app.py
    ```
3.  **Access in Browser:** Your web browser should automatically open to the application's interface (usually `http://localhost:8501`).

## Using the Application (Workflow)

The application interface has a sidebar for client/product management and setup, and a main area with tabs for different actions [1].

**1. Create Client & Product Folders**

*   **Use the Sidebar:**
    *   Under "Client", enter a unique name for a "Client" (e.g., `ClientA`, `MyDesigns`) in the "New Client Name" field and click "Create" [1].
    *   Select the newly created Client from the "Select Client" dropdown.
    *   Under "Product for 'selected\_client'", enter a unique name for a "Product" (e.g., `Gold_Rings`, `Silver_Necklaces_Style1`) in the "New Product Name" field and click "Create" [1].
*   **What happens:** This creates folders automatically within your `JewelryAI` project [1]:
    *   `JewelryAI/inputs/YourClient/YourProduct/` (For your training images)
    *   `JewelryAI/inputs/YourClient/YourProduct/model/` (Where fine-tuned LoRA weights will be saved/loaded from)
    *   `JewelryAI/outputs/YourClient/YourProduct/` (Where generated images will be saved)

**2. Add Training Images (for Fine-Tuning)**

*   **Find the Folder:** Locate the `JewelryAI/inputs/YourClient/YourProduct/` folder on your computer (e.g., `JewelryAI/inputs/MyDesigns/Gold_Rings/`).
*   **Copy Images:** Place your training images (e.g., `.jpg`, `.png`, `.webp` files) **directly** into this folder. Do *not* put them in the `model` subfolder within it [1].
*   **Quality & Quantity:** Use clear, well-lit images representative of the style you want to fine-tune. Aim for at least 10-20 high-quality images. More images (e.g., 50-200) can often lead to better results, but also require more training time.

**3. Fine-Tune Model (Optional - Requires GPU!)**

*   **Select Client/Product:** In the sidebar, ensure the correct Client and Product (for which you've added training images) are selected.
*   **Go to Tab:** Click the **'⚙️ Fine-Tune Model (LoRA)'** tab in the main area [1].
*   **Verify Images:** You should see a preview of the images you added in Step 2. If not, double-check that you placed them in the correct `inputs/YourClient/YourProduct/` folder [1].
*   **Adjust Parameters:** [1]
    *   **Resolution:** (Default: 512) Image size for training. 512 is standard for SD v1.5.
    *   **Max Training Steps:** (Default: 1500) Total optimization steps. Adjust based on dataset size and desired intensity.
    *   **LoRA Rank (Complexity):** (Default: 16) Controls capacity. 4-32 common. Higher can capture more detail but uses more memory.
    *   **Batch Size (per GPU):** (Default: 1) Images processed per step per GPU. Keep low for limited VRAM.
    *   **Gradient Accumulation:** (Default: 4) Simulates larger batch size. Effective Batch Size = Batch Size * Accumulation.
    *   **Learning Rate:** (Default: 1e-4) How fast the model adapts.
    *   **Advanced Training Options (in expander):**
        *   **Mixed Precision:** (Default: fp16) Use 16-bit precision for speed/memory saving.
        *   **Use 8-bit Adam Optimizer:** (Default: On if GPU) Saves GPU memory.
        *   **Use Gradient Checkpointing:** (Default: On) Saves GPU memory by recomputing (slightly slower).
        *   **Learning Rate Scheduler:** (Default: cosine) How learning rate changes.
        *   **Warmup Steps:** (Default: 0) Initial steps with increasing LR.
        *   **Save Checkpoint Every N Steps:** (Default: 500) Saves intermediate LoRA weights.
        *   **Training Seed:** (Default: 42) For reproducibility.
        *   **Dataloader Workers:** (Default: 2) CPU processes to load data.
*   **Start Fine-Tuning:** Click the **'🚀 Start Fine-Tuning'** button [1].
*   **Monitor Progress:** An expander labeled "Training Logs" will appear and show live output. This will take time, potentially minutes to hours. **Do not close the terminal running Streamlit or the browser tab during training.** [1]
*   **Completion:** If successful, logs indicate completion, and a message appears. The custom LoRA file, `pytorch_lora_weights.safetensors`, will be saved into `JewelryAI/inputs/YourClient/YourProduct/model/` [1].

**4. Generate Images**

*   **Select Client/Product:** In the sidebar, choose the Client and Product.
    *   If you just finished fine-tuning, **re-selecting the product in the sidebar is important** to ensure the application loads the newly created LoRA weights [2]. The app also tries to clear cache to help [1].
*   **Go to Tab:** Click the **'✨ Generate Image'** tab [1].
*   **Click Generate:** Simply click the **'Generate Image'** button.
    *   The application uses a default prompt: `"a beautiful piece of jewelry"` and other predefined settings (negative prompt, steps, guidance) for simplicity [1]. You do not need to enter a prompt.
    *   If a LoRA file (`pytorch_lora_weights.safetensors`) exists in the selected product's `model` folder, it's automatically applied [1]. Otherwise, the base model is used.
*   **View Image:** The generated image appears on this tab and is saved to `JewelryAI/outputs/YourClient/YourProduct/` [1].

**5. View & Download Generated Designs**

*   **Select Client/Product:** Ensure correct selection in the sidebar.
*   **Go to Tab:** Click the **'🖼️ Generated Designs'** tab [1].
*   **Browse Images:** Displays all images previously generated for the selected product.
*   **Download All:** A button "Download All ... Generated 'ProductName' Images (.zip)" allows downloading all images for that product [1].

**6. Adding Pre-trained LoRA Styles (Optional)**

If you wish to use the existing finetuned LoRA models (Necklaces/Earrings):

*   **Example: Necklace LoRA**
    1.  **Create Product in App:** In the sidebar, create a new "Client" if needed, then select it. Create a "Product" named `Necklaces` (or your preferred name). This creates `JewelryAI/inputs/YourClient/Necklaces/model/`.
    2.  **Download LoRA File:**
        *   Go to: `https://huggingface.co/abdulrahmanrihan/Necklaces`
        *   Navigate to the "Files and versions" tab.
        *   Download the `.safetensors` file (e.g., it might be named `Necklaces.safetensors` or similar).
    3.  **Place and Rename LoRA File:**
        *   Copy the downloaded `.safetensors` file into the product's model folder: `JewelryAI/inputs/YourClient/Necklaces/model/`.
        *   **Crucially, rename the file to exactly `pytorch_lora_weights.safetensors`**. The application specifically looks for this filename [1].
    4.  **Use in App:** In the Streamlit app, select your Client and the `Necklaces` Product in the sidebar. The app will now load this LoRA for generation.

*   **Example: Earrings LoRA**
    1.  **Create Product in App:** Create a "Product" named `Earrings` (or similar). This creates `JewelryAI/inputs/YourClient/Earrings/model/`.
    2.  **Download LoRA File:**
        *   Go to: `https://huggingface.co/abdulrahmanrihan/Earrings`
        *   Navigate to the "Files and versions" tab.
        *   Download the `.safetensors` file.
    3.  **Place and Rename LoRA File:**
        *   Copy the downloaded `.safetensors` file into `JewelryAI/inputs/YourClient/Earrings/model/`.
        *   **Rename the file to `pytorch_lora_weights.safetensors`** [1].
    4.  **Use in App:** Select your Client and the `Earrings` Product in the sidebar to use this LoRA.

## Troubleshooting

*   **`ModuleNotFoundError` (e.g., `No module named 'diffusers'`):** You missed installing a required library. Activate your virtual environment (`env`) and try `pip install -r requirements.txt` (if you have it) or `pip install <missing_library_name>`. Double-check PyTorch installation.
*   **`git` not found:** Install Git from [https://git-scm.com/downloads/](https://git-scm.com/downloads/).
*   **`accelerate` not found / `accelerate config` fails:** Ensure `accelerate` is installed (`pip show accelerate`). Try `accelerate config` again.
*   **Base Model Download/Load Errors (`checkout failed`, `model_index.json not found`, error loading `.bin` or `.safetensors` files, "Base model not found locally"):**
    *   The `git clone` for `sd-legacy/stable-diffusion-v1-5` might have failed or been incomplete [1].
    *   **Option 1: Retry Clone:** Delete the `JewelryAI/base_models/stable-diffusion-v1-5` folder and run the `git clone` command (Step 5) again. Ensure stable internet and sufficient disk space.
    *   **Option 2: Manual Download (If `git clone` repeatedly fails):**
        1.  Go to the Hugging Face repository: `https://huggingface.co/sd-legacy/stable-diffusion-v1-5/tree/main`.
        2.  Manually download essential files and place them into the correct subdirectories within `JewelryAI/base_models/stable-diffusion-v1-5/`. Key files and folders include:
            *   `model_index.json` (place in `stable-diffusion-v1-5/` root)
            *   **`unet` folder:** Download `config.json` and `diffusion_pytorch_model.safetensors` (or `.bin`) into `stable-diffusion-v1-5/unet/`.
            *   **`vae` folder:** Download `config.json` and `diffusion_pytorch_model.safetensors` (or `.bin`) into `stable-diffusion-v1-5/vae/`.
            *   **`text_encoder` folder:** Download `config.json` and `pytorch_model.safetensors` (or `.bin`) into `stable-diffusion-v1-5/text_encoder/`.
            *   **`tokenizer` folder:** Download all files (e.g., `vocab.json`, `merges.txt`, `tokenizer_config.json`, `special_tokens_map.json`) into `stable-diffusion-v1-5/tokenizer/`.
            *   **`scheduler` folder:** Download `scheduler_config.json` into `stable-diffusion-v1-5/scheduler/`.
            *   *(Safety checker files are often optional if the pipeline is configured to not use it).*
        3.  Ensure all paths and filenames match the standard Hugging Face model structure. Prefer `.safetensors` files over `.bin` if available. This method is more prone to errors than `git clone`.
*   **Fine-Tuning Fails Immediately:** Check `accelerate config`. Verify `diffusers_repo` and the training script path. Ensure training images are in `inputs/Client/Product/`. Check GPU and CUDA setup.
*   **Fine-Tuning Fails Mid-Way (`CUDA out of memory`):** Your GPU has insufficient VRAM. Try:
    *   Reduce `Batch Size (per GPU)` (try 1).
    *   Increase `Gradient Accumulation`.
    *   Ensure `Use Gradient Checkpointing` is enabled.
    *   Use `Mixed Precision: fp16`.
    *   Reduce `LoRA Rank (Complexity)`.
    *   Ensure `Resolution` is 512.
*   **Generated Images Don't Reflect Fine-Tuning:**
    *   Confirm `pytorch_lora_weights.safetensors` is in `JewelryAI/inputs/YourClient/YourProduct/model/`.
    *   **Re-select the Client and Product in the sidebar** after fine-tuning or adding a LoRA manually.
    *   Check training logs for successful completion.

## Important Notes

*   **Fine-tuning is resource-intensive!** Patience and a capable NVIDIA GPU are key.
*   **File Locations Matter:**
    *   Training images: `inputs/Client/Product/`
    *   LoRA models: `inputs/Client/Product/model/pytorch_lora_weights.safetensors`
    *   Generated images: `outputs/Client/Product/`
*   **Backups:** Regularly back up your `JewelryAI/inputs/` (especially `model` subfolders) and `JewelryAI/outputs/`.
*   **Experimentation:** Fine-tuning often requires trying different parameters. Start with shorter runs to test.

Have fun creating unique jewelry designs! ✨
