import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    from dataset_generate import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_custom_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_custom_nodes()


from nodes import (
    EmptyLatentImage,
    NODE_CLASS_MAPPINGS,
    CLIPTextEncode,
    VAEDecode,
    LoraLoader,
    KSampler,
    LatentUpscale,
    SaveImage,
    CheckpointLoaderSimple,
)


def background_generate(background_prompt,width,height):
    import_custom_nodes()
    with torch.inference_mode():
        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_4 = checkpointloadersimple.load_checkpoint(
            ckpt_name="Juggernaut_X_RunDiffusion_Hyper.safetensors"
        )

        loraloader = LoraLoader()
        loraloader_51 = loraloader.load_lora(
            lora_name="epi_noiseoffset2.safetensors",
            strength_model=1,
            strength_clip=1,
            model=get_value_at_index(checkpointloadersimple_4, 0),
            clip=get_value_at_index(checkpointloadersimple_4, 1),
        )

        cliptextencode = CLIPTextEncode()
        cliptextencode_6 = cliptextencode.encode(
            text=background_prompt,
            clip=get_value_at_index(loraloader_51, 1),
        )

        cliptextencode_7 = cliptextencode.encode(
            text="Unrealistic pictures, high saturation, cartoon style, deformed, disharmonious pictures.", clip=get_value_at_index(loraloader_51, 1)
        )

        saveimage = SaveImage()
        saveimage_9 = saveimage.save_images(filename_prefix="ComfyUI")

        bria_rmbg_modelloader_zho = NODE_CLASS_MAPPINGS["BRIA_RMBG_ModelLoader_Zho"]()
        bria_rmbg_modelloader_zho_21 = bria_rmbg_modelloader_zho.load_model()

        emptylatentimage = EmptyLatentImage()
        emptylatentimage_26 = emptylatentimage.generate(
            width=width, height=height, batch_size=1
        )

        layereddiffusionapply = NODE_CLASS_MAPPINGS["LayeredDiffusionApply"]()
        ksampler = KSampler()
        vaedecode = VAEDecode()
        bria_rmbg_zho = NODE_CLASS_MAPPINGS["BRIA_RMBG_Zho"]()
        latentupscale = LatentUpscale()

        for q in range(1):
            layereddiffusionapply_24 = layereddiffusionapply.apply_layered_diffusion(
                config="SDXL, Attention Injection",
                weight=1,
                model=get_value_at_index(loraloader_51, 0),
            )

            ksampler_3 = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=50,
                cfg=8,
                sampler_name="dpmpp_2m",
                scheduler="karras",
                denoise=1,
                model=get_value_at_index(layereddiffusionapply_24, 0),
                positive=get_value_at_index(cliptextencode_6, 0),
                negative=get_value_at_index(cliptextencode_7, 0),
                latent_image=get_value_at_index(emptylatentimage_26, 0),
            )

            vaedecode_50 = vaedecode.decode(
                samples=get_value_at_index(ksampler_3, 0),
                vae=get_value_at_index(checkpointloadersimple_4, 2),
            )

            bria_rmbg_zho_20 = bria_rmbg_zho.remove_background(
                rmbgmodel=get_value_at_index(bria_rmbg_modelloader_zho_21, 0),
                image=get_value_at_index(vaedecode_50, 0),
            )

            latentupscale_47 = latentupscale.upscale(
                upscale_method="nearest-exact",
                width=1152,
                height=1152,
                crop="disabled",
                samples=get_value_at_index(ksampler_3, 0),
            )


# if __name__ == "__main__":
#     main()
