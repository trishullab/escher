import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np
from tqdm import tqdm

from escher.cbd_utils import cache_completion
from escher.cbd_utils.server import OPENAI_TEMP, openai_client


def generate_prompt(category_name: str):
    # you can replace the examples with whatever you want; these were random and worked, could be improved
    return f"""Q: What are useful visual features for distinguishing a lemur in a photo? Give me atleast seven features.
A: There are several useful visual features to tell there is a lemur in a photo:
- four-limbed primate
- black, grey, white, brown, or red-brown
- wet and hairless nose with curved nostrils
- long tail
- large eyes
- furry bodies
- clawed hands and feet

Q: What are useful visual features for distinguishing a television in a photo? Give me atleast eight features.
A: There are several useful visual features to tell there is a television in a photo:
- electronic device
- black or grey
- a large, rectangular screen
- a stand or mount to support the screen
- one or more speakers
- a power cord
- input ports for connecting to other devices
- a remote control

Q: What are useful features for distinguishing a {category_name} in a photo? Give me atleast 19 features.
A: There are several useful visual features to tell there is a {category_name} in a photo:
-"""


@cache_completion("gpt_completions.db")
def get_descriptions(prompt):
    time.sleep(np.random.choice([1e-2, 1e-1, 1, 2, 3]))
    completion = openai_client.completions.create(
        prompt=prompt,
        model="gpt-3.5-turbo-instruct",
        max_tokens=1024,
        n=1,
        temperature=OPENAI_TEMP,
    )
    return completion.choices[0].text.strip()


def generate_cbd_descriptors(classes: List[str]):
    prompts = []
    for i, class_name in enumerate(classes):
        prompt = generate_prompt(class_name)
        prompts.append(prompt)

    with ThreadPoolExecutor(max_workers=10) as executor:
        outputs = list(
            tqdm(executor.map(get_descriptions, prompts), total=len(prompts))
        )

    outputs = [list(map(lambda s: s.strip("- *"), o.split("\n"))) for o in outputs]
    outputs = [list(filter(lambda s: len(s) > 0, o)) for o in outputs]
    return outputs
