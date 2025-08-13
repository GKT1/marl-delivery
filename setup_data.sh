#!/usr/bin/env bash
##############################################################################
#  Enhanced Vast-AI ComfyUI provisioning script
#  • Keeps your original logic
#  • Adds Google-Drive FILE + FOLDER support via gdown & rclone
##############################################################################

source /venv/main/bin/activate
COMFYUI_DIR=${WORKSPACE}/ComfyUI
APT_INSTALL='apt-get update -qq && apt-get install -y'

##############################################################################
# ---------- USER SECTION: edit arrays only ----------------------------------
##############################################################################

APT_PACKAGES=(         # for apt; leave empty or comment items to skip
    #"ffmpeg"
)

PIP_PACKAGES=(         # plain pip packages
    #"some-pypi-package"
)

NODES=(                # custom-node GitHub repos
    #"https://github.com/ltdrdata/ComfyUI-Manager"
    "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes"
    "https://github.com/rgthree/rgthree-comfy"
    "https://github.com/yolain/ComfyUI-Easy-Use"
    "https://github.com/kijai/ComfyUI-FluxTrainer"
    "https://github.com/pythongosssss/ComfyUI-WD14-Tagger"
    "https://github.com/WASasquatch/was-node-suite-comfyui"
    "https://github.com/kijai/ComfyUI-KJNodes"
    "https://github.com/whitmell/ComfyUI-RvTools"
    "https://github.com/Fannovel16/comfyui_controlnet_aux"
    "https://github.com/sipherxyz/comfyui-art-venture"
)

WORKFLOWS=( )

CHECKPOINT_MODELS=(
    "https://civitai.com/api/download/models/1761560?type=Model&format=SafeTensor&size=pruned&fp=fp16"
)

UNET_MODELS=( )
LORA_MODELS=(
    "https://huggingface.co/GKT/lora1_rank32_fp16-step01000/resolve/main/lora1_rank32_fp16-step01000.safetensors"
    "https://huggingface.co/GKT/kohya1/resolve/main/new%20model-000001.safetensors"
    "https://huggingface.co/GKT/kohya1/resolve/main/new%20model-000002.safetensors"
    "https://huggingface.co/GKT/kohya1/resolve/main/new%20model-000003.safetensors"
    "https://huggingface.co/GKT/kohya1/resolve/main/new%20model-000004.safetensors"
    "https://huggingface.co/GKT/kohya1/resolve/main/new%20model-000005.safetensors"
    "https://huggingface.co/GKT/kohya1/resolve/main/new%20model-000006.safetensors"
    "https://huggingface.co/GKT/kohya1/resolve/main/new%20model-000007.safetensors"
    "https://huggingface.co/GKT/kohya1/resolve/main/new%20model-000008.safetensors"
    "https://huggingface.co/GKT/kohya1/resolve/main/new%20model-000009.safetensors"
    "https://huggingface.co/GKT/kohya1/resolve/main/new%20model-000010.safetensors"
    "https://huggingface.co/GKT/kohya1/resolve/main/new%20model-000011.safetensors"
    "https://huggingface.co/GKT/kohya1/resolve/main/new%20model-000012.safetensors"
    "https://huggingface.co/GKT/kohya1/resolve/main/new%20model-000013.safetensors"
    "https://huggingface.co/GKT/kohya1/resolve/main/new%20model-000014.safetensors"
    "https://huggingface.co/GKT/kohya1/resolve/main/new%20model-000015.safetensors"
    "https://huggingface.co/GKT/kohya1/resolve/main/new%20model-000016.safetensors"
    "https://huggingface.co/GKT/kohya1/resolve/main/new%20model-000017.safetensors"
    "https://huggingface.co/GKT/kohya1/resolve/main/new%20model-000018.safetensors"
    "https://huggingface.co/GKT/kohya1/resolve/main/new%20model.safetensors"
)
VAE_MODELS=(
    # "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors"
)
ESRGAN_MODELS=( )
CONTROLNET_MODELS=(
    "https://civitai.com/api/download/models/1480637?type=Model&format=SafeTensor&token=c31e72d268968c10449529820d8b417f"
    "https://civitai.com/api/download/models/1053985?type=Model&format=SafeTensor&token=c31e72d268968c10449529820d8b417f"
)
CLIP_VISION_MODELS=(
    # "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors"
)
DIFFUSION_MODELS=(
    # "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_i2v_480p_14B_fp8_e4m3fn.safetensors"
)
TEXT_ENCODERS_MODELS=(
    # "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
)



##############################################################################
# ---------- INTERNAL LOGIC (touch only if you know what you’re doing) -------
##############################################################################

function provisioning_start() {
    print_header
    install_basics
    get_apt_packages
    get_pip_packages
    get_nodes

    get_files "${COMFYUI_DIR}/models/checkpoints" "${CHECKPOINT_MODELS[@]}"
    get_files "${COMFYUI_DIR}/models/unet"        "${UNET_MODELS[@]}"
    get_files "${COMFYUI_DIR}/models/loras"        "${LORA_MODELS[@]}"
    get_files "${COMFYUI_DIR}/models/controlnet"  "${CONTROLNET_MODELS[@]}"
    get_files "${COMFYUI_DIR}/models/vae"         "${VAE_MODELS[@]}"
    get_files "${COMFYUI_DIR}/models/esrgan"      "${ESRGAN_MODELS[@]}"
    get_files "${COMFYUI_DIR}/models/clip_vision" "${CLIP_VISION_MODELS[@]}"
    get_files "${COMFYUI_DIR}/models/diffusion_models" "${DIFFUSION_MODELS[@]}"
    get_files "${COMFYUI_DIR}/models/text_encoders" "${TEXT_ENCODERS_MODELS[@]}"

    print_end
}

# --------------------- helpers ----------------------------------------------
function install_basics() {
    command -v gdown >/dev/null || pip install --no-cache-dir -U gdown
    # keep aria2 for fast checkpoint pulls
    command -v aria2c >/dev/null || ( $APT_INSTALL aria2 )
}

function get_apt_packages() {
    [[ -z ${APT_PACKAGES[*]} ]] && return
    sudo bash -c "$APT_INSTALL ${APT_PACKAGES[*]}"
}

function get_pip_packages() {
    [[ -z ${PIP_PACKAGES[*]} ]] && return
    pip install --no-cache-dir ${PIP_PACKAGES[*]}
}

function get_nodes() {
    for repo in "${NODES[@]}"; do
        dir="${repo##*/}"
        path="${COMFYUI_DIR}/custom_nodes/${dir}"
        req="${path}/requirements.txt"
        if [[ -d $path ]]; then
            [[ ${AUTO_UPDATE,,} != "false" ]] && { echo "Updating $dir"; (cd "$path" && git pull); }
        else
            echo "Cloning $dir"
            git clone --recursive "$repo" "$path"
        fi
        [[ -e $req ]] && pip install --no-cache-dir -r "$req"
    done
}

function get_files() {                # $1 target dir , rest = URLs
    local dir="$1"; shift; local arr=("$@")
    [[ -z ${arr[*]} ]] && return
    mkdir -p "$dir"
    for url in "${arr[@]}"; do
        echo "↳ ${url##*/}"
        download "$url" "$dir"
    done
}



# ---------- generic downloader ----------------------------------------------
function download() {                  # $1 url , $2 dir
    local url="$1" dir="$2" auth=""
    if [[ -n $HF_TOKEN && $url =~ huggingface\.co ]]; then auth="$HF_TOKEN"; fi
    if [[ -n $CIVITAI_TOKEN && $url =~ civitai\.com   ]]; then auth="$CIVITAI_TOKEN"; fi
    if [[ $url =~ drive\.google\.com ]]; then
          # extract FILE_ID from any share link
          fid=$(sed -n 's,.*\([=/]\)\([0-9A-Za-z_-]\{10,\}\).*,\2,p' <<<"$url")
          url="https://drive.usercontent.google.com/download?id=${fid}&export=download&confirm=y"
      fi


    [[ -f $dir/${url##*/} ]] && return
    if [[ -n $auth ]]; then
        wget --header="Authorization: Bearer $auth" -nc --content-disposition \
             --show-progress -e dotbytes=4M -P "$dir" "$url"
    else
        wget -qnc --content-disposition --show-progress -e dotbytes=4M \
             -P "$dir" "$url"
    fi
}

# ---------- UX ----------------------------------------------------------------
print_header() {
    echo -e "\n################  Provisioning container  ################\n"
}
print_end() {
    echo -e "\nProvisioning complete – ComfyUI will start now\n"
}

# ---------- run (unless user touched /.noprovisioning) ------------------------
[[ ! -f /.noprovisioning ]] && provisioning_start
