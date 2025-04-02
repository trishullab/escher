<div align="center">
  <h1>Escher: Self-Evolving Visual Concept Library using Vision-Language Critics</h1>
    <p>
        <a href="https://trishullab.github.io/escher-web/">
        <img src="https://trishullab.github.io/escher-web/static/images/teaser.svg" alt="Escher Teaser" width="800"/>
        </a>
        <!-- add a caption -->
        <br>
         Prior work in concept-bottleneck visual recognition aims to leverage discriminative visual concepts to enable more accurate object classification. Escher is an approach for iteratively evolving a visual concept library using feedback from a VLM critic to discover descriptive visual concepts. 
    </p>
</div>

More details can be found on the [project page](https://trishullab.github.io/escher-web/).

## Getting started

__With Escher__:
```bash
$ conda create --name escher-dev python=3.12
$ conda activate escher-dev
$ pip install -e . # Installs escher in editable mode
$ vim escher/cbd_utils/server.py
# Edit this file to point to the correct GPT model, API key location, etc.
$ CUDA_VISIBLE_DEVICES=4 python escher/iteration.py --dataset cub --topk 50 --prompt_type confound_w_descriptors_with_conversational_history --distance_type confusion --subselect -1 --decay_factor 10 --classwise_topk 10 --num_iters 100 --perc_labels 0.0 --perc_initial_descriptors 1.00 --algorithm lm4cv --salt "1.debug"
# ^ This runs with openai-gpt-3.5-turbo, no vllm instance required.
$ CUDA_VISIBLE_DEVICES=1 python escher/iteration.py --dataset cub --topk 50 --openai_model LOCAL:meta-llama/Llama-3.1-8B-Instruct --prompt_type confound_w_descriptors_with_conversational_history --distance_type confusion --subselect -1 --decay_factor 10 --classwise_topk 10 --num_iters 100 --perc_labels 0.0 --perc_initial_descriptors 1.00 --algorithm lm4cv --salt "1.debug"
# ^ Same command but the LOCAL: prefix makes it use the vllm_client defined in `serve.py` and calls the LLama-3.1-8B-Instruct model.

# To run on multiple datasets, read cmds.sh
```


__With vLLM__ (if using local LLM):
```bash
$ conda env create --name vllm python=3.9
$ conda activate vllm
$ pip install vllm
$ CUDA_VISIBLE_DEVICES=0,1,2,3 vllm meta-llama/Llama-3.1-8B-Instruct --api-key token-abc123 --port 11440 --tensor-parallel-size 4 --disable-log-requests
# ^ This is a debugging server. Running with meta-llama/Llama-3.3-70B-Instruct is recommended.
$ CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve meta-llama/Llama-3.3-70B-Instruct  --api-key token-abc123 --port 11440 --tensor-parallel-size 4 --disable-log-requests
# ^ See if this fits. I wasn't able to get it work and had to use a quantized model (which also work well).
```

## General Structure

- `escher/`
    - `iteration.py`: Main file to run the self-evolving process. This parses the command line arguments, loads the initial set of concepts defined in `descriptors/cbd_descriptors/descriptors_{dataset}.json`, and instantiates the model.
    - `library/`
      - `library.py`: A wrapper around the library of concepts. This class is responsible for loading the concepts, updating the concepts, and saving the concepts.
      - `history_conditioned_library.py`: An extension of `library.py` that uses the history of the concepts to condition the library generation.
    - `models/`
      - `model.py`: The abstract class for a concept-bottleneck based model.
      - `model_zero_shot.py`: Implementation of the model which does not train any parameters.
      - `model_lm4cv.py`: Implementation of the model which trains the parameters of the model.
    - `utils/dataset_loader.py`: Main entry point for loading the dataset.
    - `cbd_utils`: Many utility functions for GPT calling / training / evaluation / caching useful for implementing a "zeroshot" Escher model.
    - `lm4cv`: Many utility functions for implementing a "lm4cv" Escher model.
- `descriptors/`: Contains the initial descriptors for each dataset.
- `cmds.sh`: A set of hacky scripts to run `escher` on multiple datasets.


