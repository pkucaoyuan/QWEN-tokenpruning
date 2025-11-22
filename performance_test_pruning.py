from run_with_token_pruning import setup_pipeline_with_pruning
import os
from PIL import Image
import torch

from pruning_modules import global_pruning_cache  # pyright: ignore[reportMissingImports]


# Qwen-Image-Edit-2509 with pruning:       
# 显存占用：     70G   平均时间：2.89s   错误：  40-43, 46-47, 56, 58, 59, 64, 65, 70, 71, 73-79
# Qwen-Image-Edit-2509 without pruning:    
# 显存占用：     61G   平均时间：3.51s   无错误
            
import time



OUTPUT_DIR = "outputs_pruning_qwenimage_edit_2509"
ENABLE_PRUNING = False
DEVICE_ID = 0



def main():
    image_paths = [os.path.join("datasets/edit-test-img", f"{i}.png") for i in range(80)]
    
    image_pils = [Image.open(image_path).convert("RGB") for image_path in image_paths]
    caption_path = "datasets/edit-test-img/captions.txt"
    output_dir = os.path.join(OUTPUT_DIR, f"{'pruning' if ENABLE_PRUNING else 'no_pruning'}")
    os.makedirs(output_dir, exist_ok=True)
    output_image_paths = [os.path.join(output_dir, f"{i}.png") for i in range(80)]

    prompts = open(caption_path, "r").readlines()

    pipeline = setup_pipeline_with_pruning(enable_pruning=ENABLE_PRUNING)

    total_inference_time = 0
    failed_count = 0

    with torch.inference_mode():

        for image_pil, prompt, output_image_path in zip(image_pils, prompts, output_image_paths):

            inference_params = {
                "image": image_pil,
                "prompt": prompt,
                "negative_prompt": " ",
                "num_inference_steps": 4,
                "true_cfg_scale": 1.0,
                "generator": torch.manual_seed(42),
            }

            global_pruning_cache.clear_caches()
            global_pruning_cache.current_step = 0

            cur_inference_start = time.time()
            try:
                output = pipeline(**inference_params)
            except Exception as e:
                print(f"Error: {e} at image {output_image_path}")
                failed_count += 1
                continue
            output_image = output.images[0]
            cur_inference_time = time.time() - cur_inference_start
            print(f"Inference time: {cur_inference_time:.2f} seconds")
            total_inference_time += cur_inference_time
            output_image.save(output_image_path)

    print(f"Mode: {'pruning' if ENABLE_PRUNING else 'no_pruning'} - Total inference time: {total_inference_time:.2f} seconds")
    print(f"Mode: {'pruning' if ENABLE_PRUNING else 'no_pruning'} - Average inference time: {total_inference_time / (80 - failed_count):.2f} seconds")

if __name__ == "__main__":
    main()