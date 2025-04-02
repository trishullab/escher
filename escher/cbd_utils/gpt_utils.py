import ast
import functools
import json
import pickle
import re
import sqlite3
import threading
from typing import List, Tuple

import numpy as np

from .server import openai_client, vllm_client

lock = threading.Lock()  # Lock for database operations
find_json_block = re.compile(r"```json(.*?)```", re.DOTALL)


def get_db_connection(db_loc):
    conn = sqlite3.connect(db_loc)
    c = conn.cursor()
    return c, conn


def cache_completion(db_loc):
    def decorator(f):
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            c, conn = get_db_connection(db_loc)
            function_name = f.__name__

            # Ensure the gpt_completions table exists
            with lock:
                c.execute(
                    """
                    CREATE TABLE IF NOT EXISTS gpt_completions_2 (
                        function_name TEXT,
                        argument_blob BLOB,
                        result_blob BLOB,
                        PRIMARY KEY (function_name, argument_blob)
                    )
                """
                )
                conn.commit()

            try:
                # Serialize arguments to a blob
                args_blob = pickle.dumps((function_name, args, kwargs))

                # Check the cache
                with lock:
                    c.execute(
                        "SELECT result_blob FROM gpt_completions_2 WHERE function_name = ? AND argument_blob = ?",
                        (function_name, args_blob),
                    )
                    result = c.fetchone()
                    if result:
                        # If found in cache, return the deserialized result
                        return pickle.loads(result[0])

                # Compute the result since it's not cached
                result = f(*args, **kwargs)

                # Serialize and store the result
                result_blob = pickle.dumps(result)
                with lock:
                    # Try to insert the new result
                    c.execute(
                        "INSERT INTO gpt_completions_2 (function_name, argument_blob, result_blob) VALUES (?, ?, ?)",
                        (function_name, args_blob, result_blob),
                    )
                    conn.commit()
                return result
            finally:
                conn.close()

        return wrapped

    return decorator


def get_embedding(attribute):
    embedding_resp = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=attribute,
    )
    embedding = np.array(embedding_resp.data[0].embedding)
    return embedding


@cache_completion(
    "/var/local/atharvas/f/learning_descriptors/gpt_completions.db"
)  # Used for debugging only. Comment out for active code.
def get_completion(*args, **kwargs):
    if "model" in kwargs and kwargs["model"].startswith("LOCAL:"):
        kwargs["model"] = kwargs["model"].replace("LOCAL:", "")
        client = vllm_client
    else:
        client = openai_client
    return client.chat.completions.create(*args, **kwargs)


def gpt_parse_suggestions(suggestions):
    if not len(suggestions):
        return {}
    try:
        suggestions = json.loads(suggestions)
        return suggestions
    except json.JSONDecodeError:
        # find the ```json block```
        if match := find_json_block.search(suggestions):
            suggestions = match.group(1)
            try:
                suggestions = json.loads(suggestions)
                return suggestions
            except json.JSONDecodeError:
                try:
                    suggestions = ast.literal_eval(suggestions)
                    return suggestions
                except Exception:
                    print(
                        f"Could not decode JSON w/Python literal eval : {suggestions}"
                    )
                print(f"Could not decode JSON : {suggestions}")
                return None
        print(f"Could not find JSON : {suggestions}")
        return None


def message_builder_question_question(question):
    question_proc = question.strip().replace("\n", "")
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant that returns output as a list of JSON objects with a list of follow up observational questions, and the probability of that question being helful in visually identifying the object.\n"
            + "For instance, if I ask `Is this a picture of a lemur? (a) Yes (b) No`, you should return a list of questions in JSON format like"
            + """[{ "question": "Does the animal have a long tail? (a) Yes (b) No", "importance": 0.9 }, { "question": "Does the animal have a black and white striped tail? (a) Yes (b) No", "importance": 0.8 }]""",
        },
        {
            "role": "user",
            "content": 'What observational questions will help me answer the question:  "Is this a picture of a bald eagle? (a) Yes (b) No." An observational question only inquires about the characteristics of an object that can be observed visually.',
        },
        {
            "role": "assistant",
            "content": """[
  {
    "question": "Does the bird have a white head? (a) Yes (b) No.",
    "importance": 0.9
  },
  {
    "question": "Does the bird have a curved yellow beak? (a) Yes (b) No.",
    "importance": 0.8
  },
  {
    "question": "Does the bird have white tail feathers? (a) Yes (b) No.",
    "importance": 0.7
  },
  {
    "question": "Does the bird have yellow talons? (a) Yes (b) No.",
    "importance": 0.7
  },
  {
    "question": "Is this an image of an eagle? (a) Yes (b) No.",
    "importance": 0.8
  },
  {
    "question": "Is this an image of a bird? (a) Yes (b) No.",
    "importance": 0.8
  }
]""",
        },
        {
            "role": "user",
            "content": f'What observational questions will help me answer the question: "{question_proc}". An observational question only inquires about the characteristics of an object that can be observed visually.',
        },
    ]


def message_builder(category_name):
    """
    Recommend _new features_ for a category
    """
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant that suggests visual features useful for discriminating a specific object in an image. Format your suggestions as a JSON list. Restrict to one suggestion per item.",
        },
        {
            "role": "user",
            "content": """What are useful visual features for distinguishing a lemur in a photo?  Do not give any extra information like links, sources, etc.""",
        },
        {
            "role": "assistant",
            "content": """```json
["four-limbed primate",
"black, grey, white, brown, or red-brown",
"wet and hairless nose with curved nostrils",
"long tail",
"large eyes",
"furry bodies",
"clawed hands and feet"
]```""",
        },
        {
            "role": "user",
            "content": """What are useful visual features for distinguishing a television in a photo? Do not give any extra information like links, sources, etc.""",
        },
        {
            "role": "assistant",
            "content": """```json
["electronic device",
"black or grey",
"a large, rectangular screen",
"a stand or mount to support the screen",
"one or more speakers",
"a power cord",
"input ports for connecting to other devices",
"a remote control"
]```""",
        },
        {
            "role": "user",
            "content": """What are useful visual features for distinguishing a \"{category_name}\" in a photo? Do not give any extra information like links, sources, etc.""".format(
                category_name=category_name
            ),
        },
    ]


def message_builder_programlike(clsA, clsB, descA, descB):
    descA_string = ",".join([f'classify("{desc}")' for desc in descA])
    descB_string = ",".join([f'classify("{desc}")' for desc in descB])

    message = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": 'Suggest program modifications that enhance the program\'s ability to visually distinguish an image containing "{clsA}" from an image containing "{clsB}." Either add, delete, or modify the existing descriptors.\n\ndef mean_classify(category="{clsA}"):\n    return mean([{descA}])\n\ndef mean_classify(category="{clsB}"):\n    return mean([{descB}])\n\nOnly return the modified programs.'.format(
                        clsA=clsA, clsB=clsB, descA=descA_string, descB=descB_string
                    ),
                }
            ],
        }
    ]
    return message


def program_builder(i, cls, desc):
    """
    def program1(image):\n    \"\"\"Target class: bald eagle\"\"\"\n    return mean_image_score(image, [\"white head\", \"curved yellow beak\", \"white tail feathers\", \"yellow talons\", \"eagle\", \"bird\"])
    """
    desc_string = '", "'.join(desc)
    return f'def program{i}(image):\n    """Target class: {cls}"""\n    return mean_image_score(image, ["{desc_string}"])'


def program_builder_repair(i, cls, desc, score, reasoning):
    """
    def program1(image):\n    \"\"\"\n    Target class: black-footed albatross.\n    Grade: 6 / 10.\n    Reasoning: Program1 focuses on specific features related to the black-footed albatross, but the use of vague descriptors like 'a seabird' and 'black and white' may reduce clarity. However, it captures key characteristics, resulting in moderate interpretability.\n    \"\"\"\n    return mean_image_score(image, [\"a white head and neck\", \"long, narrow wings\", \"a long, hooked bill\", \"a seabird\", \"black and white\", \"black feet\",],)\n )
    """
    # in the reason replace all Program 1 / program 1 / Program1 / program1 with program{i}
    regex = re.compile(r"Program\s*\d+", re.IGNORECASE)
    reasoning = regex.sub(f"program{i}", reasoning)
    desc_string = '", "'.join(desc)
    return f'def program{i}(image):\n    """\n    Target class: {cls}.\n    Grade: {score} / 10.\n    Reasoning: {reasoning}\n    """\n    return mean_image_score(image, ["{desc_string}"])'


def message_builder_interpretability_classifier(
    cls_desc_pairs: List[Tuple[str, List[str]]],
):
    programs = [
        program_builder(i, cls, desc) for i, (cls, desc) in enumerate(cls_desc_pairs, 1)
    ]
    programs_str = "\n".join(programs)
    message = [
        {
            "role": "system",
            "content": [
                {
                    "text": 'You are a helpful assistant tasked with rating the interpretability of a program that classifies an image into a specific class. Your rating should be on a scale from 1 to 10, where 1 indicates low interpretability or incorrectness, and 10 represents high interpretability. In your evaluation, consider the following characteristics, even though they may sometimes conflict, requiring trade-offs between them. Use your judgment along with these guidelines to assess how well the program captures the intended class:\n\n1. **Parsimony**\n2. **Tractability**\n3. **Conceptual insightfulness**\n4. **Generalizability**\n5. **Falsifiability**\n6. **Empirical consistency**\n7. **Predictive precision**\n\nNote: These criteria may not always align perfectly, so you will need to balance them in your final rating.\n\nFormat your answer as a JSON list of interpretability ratings for each program provided:\n```json\n{\n  "reasoning": [\n    "reasoning for program1",\n    "reasoning for program2",\n    "reasoning for programN"\n  ],\n  "grade": [\n    "grade for program1",\n    "grade for program2",\n    "grade for programN"\n  ]\n}\n```',
                    "type": "text",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": 'Rate the interpretability of the following programs:\n```python\ndef mean_image_score(image: np.ndarray, classes: list[str]):\n     """Returns the activation of each class on an image averaged together. Assume competent model. Implementation irrelevant"""\n    pass\n{programs_str}\n```'.format(
                        programs_str=programs_str
                    ),
                }
            ],
        },
    ]
    return message


def message_builder_interpretability_repair(
    cls_desc_score_reason_tuples: List[Tuple[str, List[str], float, str]],
):
    programs = [
        program_builder_repair(i, cls, desc, score, reasoning)
        for i, (cls, desc, score, reasoning) in enumerate(
            cls_desc_score_reason_tuples, 1
        )
    ]
    programs_str = "\n".join(programs)
    message = [
        {
            "role": "system",
            "content": [
                {
                    "text": 'You are a helpful assistant that repairs a poor performing image classification program. Given the program and the reasoning for poor performance of the program, repair the program accordingly. Attempt to retain descriptor strings whenever possible. Restrict your output to the following JSON format:\n```json\n{\n  "reasoning": "reasoning1",\n  "repaired_program": "repaired_program1"\n}\n```',
                    "type": "text",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "```python\nfrom metrics import mean_image_score # mean_image_score takes an image numpy array and a list of visaul attributes (strings) and returns the mean of the probability of the image containing that attribute.\n{programs_str}\n```".format(
                        programs_str=programs_str
                    ),
                }
            ],
        },
    ]
    return message


find_list_regex = re.compile(r'\[\s*"(.*?)"(?:,\s*"(.*?)")*\s*\]')


def parse_program(gpt_output):
    """
    Extracts just the list of descriptors from the GPT output
    Input
    -----
    gpt_output: str
    Returns
    -------
    list[str]

    Example
    -------
    gpt_output = "        def mean_image_score(image: np.ndarray, classes: list[str]):\n     \"\"\"Returns the activation of each class on an image averaged together. Assume competent model. Implementation irrelevant\"\"\"\n    pass\n\ndef program1(image):\n    \"\"\"\n    Target class: black-footed albatross.\n    Grade: 8 / 10.\n    Reasoning: The function now uses more specific and identifiable features that are directly related to the black-footed albatross, enhancing clarity and performance.\n    \"\"\"\n    return mean_image_score(image, [\n        \"white head and neck with distinct markings\",\n        \"long, narrow wings with a wingspan of over 2 meters\",\n        \"long, hooked bill for catching fish\",\n        \"large seabird species recognized for its grace\",\n        \"primarily white with black tips on wings\",\n        \"black feet with webbing\",\n    ]"
    output = ["white head and neck with distinct markings", "long, narrow wings with a wingspan of over 2 meters", "long, hooked bill for catching fish", "large seabird species recognized for its grace", "primarily white with black tips on wings", "black feet with webbing"]
    """
    # m = find_list_regex.search(gpt_output)
    # if m:
    #     descriptors = [descriptor.strip() for descriptor in re.findall(r'"(.*?)"', m.group())]
    #     return descriptors
    # return []
    if isinstance(gpt_output, list):
        gpt_output = "\n".join(map(str, gpt_output))
    try:
        tree = ast.parse(gpt_output)
        for node in ast.walk(tree):
            if isinstance(node, ast.Return):
                for elt in node.value.args:
                    if isinstance(elt, ast.List):
                        return [
                            e.s
                            for e in elt.elts
                            if isinstance(e, ast.Constant) and isinstance(e.s, str)
                        ]
    except Exception as e:
        print(f"Error parsing program: {e}")
        print("Fallback to regex")

    try:
        m = find_list_regex.search(gpt_output)
        if m:
            descriptors = [
                descriptor.strip() for descriptor in re.findall(r'"(.*?)"', m.group())
            ]
            return descriptors
        else:
            print(f"Could not find list in {gpt_output}")
    except Exception as e:
        print(f"Error parsing program: {e}")
        print("Fallback to empty list")

    return []


def get_class_suggestions_updated(clsA: str, clsB: str):
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    #   "text": "You are a helpful assistant that brainstorms visual markers to distinguish between two objects. We will give the names of two objects. A classifier often mistakes the first object for the second object. You will brainstorm visual features of the first object that will help the classifier distinguish the first object from the second object. Your visual features must be simple and concise. Each feature should be only one sentence long. We will also ask you to provide your reasoning for your suggested new features. Your output should be a JSON object that looks like this:\n{\n \"reasoning\": [\"reasoning for feature1\", \"reasoning for feature2\", ..., \"reasoning for featureN\"],\n \"features\": [\"feature1\", \"feature2\", ..., \"featureN\"]\n}\n\nFor example:\nQ: Please propose visual features of a phone that distinguish it from a television.\nA: {\n \"reasoning\": [\"a phone has curved edges, but a TV has sharp edges.\", \"most phones have cameras on their back, that are usually circular, while most TVs do not have any cameras.\", \"A phone is often times connected to a small charging cable. A TV is usually connected to a large thick power cord, not a small charging cable.\"],\n \"features\": [\"curved edges\", \"circular cameras on the back\", \"small charging cable\"]\n}\n\nQ: Please propose visual features of a yellow warbler that distinguish it from a hooded warbler.\nA: {\n \"reasoning\": [\"A yellow warbler has faint olive or brownish tones on wings and back, whereas a hooded warbler does not.\", \"A yellow warbler has a uniformly yellow face, while a hooded warbler has a yellow eye ring and a black hood that contrasts sharply with its yellow face.\"],\n \"features\": [\"olive or brownish tones on wings and back\", \"uniformly yellow face\"]\n}"
                    "text": 'Point out visual features that can be used to distinguish between two image descriptions. Your output should be a JSON object that looks like this:\n{\n "reasoning": ["reasoning for feature1", "reasoning for feature2", ..., "reasoning for featureN"],\n "features": ["feature1", "feature2", ..., "featureN"]\n}\n\nFor example:\nQ: Please propose visual features of a phone that distinguish it from a television.\nA: {\n "reasoning": ["a phone has curved edges, but a TV has sharp edges.", "most phones have cameras on their back, that are usually circular, while most TVs do not have any cameras.", "A phone is often times connected to a small charging cable. A TV is usually connected to a large thick power cord, not a small charging cable."],\n "features": ["curved edges", "circular cameras on the back", "small charging cable"]\n}\n\nQ: Please propose visual features of a yellow warbler that distinguish it from a hooded warbler.\nA: {\n "reasoning": ["A yellow warbler has faint olive or brownish tones on wings and back, whereas a hooded warbler does not.", "A yellow warbler has a uniformly yellow face, while a hooded warbler has a yellow eye ring and a black hood that contrasts sharply with its yellow face."],\n "features": ["olive or brownish tones on wings and back", "uniformly yellow face"]\n}',
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f'Please propose visual features of "{clsA}" that distinguish it from "{clsB}."',
                }
            ],
        },
    ]
    return messages


def get_class_suggestions(clsA: str, clsB: str):
    messages = [
        {
            "role": "system",
            "content": """You are a helpful assistant that brainstorms visual features to distinguish between two objects.
Your visual features must be simple, concise, and described in 1-2 sentences. Do not explain the features in the same line.
Each feature should be followed by reasoning that justifies the feature. Your output should look like this:
```json
{
    "reasoning": ["reasoning for feature1", "reasoning for feature2", ..., "reasoning for featureN"],
    "features": ["feature1", "feature2", ..., "featureN"]
}
```""",
        },
        {
            "role": "user",
            "content": """Please suggest visual features of a phone that distinguish it from a television. Keep your suggested features as brief, concise, and simple as possible. Please explain your reasoning as well.""",
        },
        {
            "role": "assistant",
            "content": """```json
{
    "reasoning": ["a phone has curved edges, but a TV has sharp edges.", "most phones have cameras on their back, that are usually circular, while most TVs do not have any cameras.", "A phone is often times connected to a small charging cable. A TV is usually connected to a large thick power cord, not a small charging cable."],
    "features": ["curved edges", "circular cameras on the back", "small charging cable"]
}
```""",
        },  # TODO(pyuan95): maybe change this example to not use classes inside CUB.
        {
            "role": "user",
            "content": """Please suggest visual features of a yellow warbler that distinguish it from a hooded warbler. Keep your suggested features as brief, concise, and simple as possible. Please explain your reasoning as well.""",
        },
        {
            "role": "assistant",
            "content": """```json
{
    "reasoning": ["A yellow warbler has faint olive or brownish tones on wings and back, whereas a hooded warbler does not.", "A yellow warbler has a uniformly yellow face, while a hooded warbler has a yellow eye ring and a black hood that contrasts sharply with its yellow face."],
    "features": ["olive or brownish tones on wings and back", "uniformly yellow face"]
}
```""",
        },
        {
            "role": "user",
            "content": """Please suggest visual features of a {clsA} that distinguish it from a {clsB}. Keep your suggested features as brief, concise, and simple as possible. Please explain your reasoning as well.""".format(
                clsA=clsA, clsB=clsB
            ),
        },
    ]
    return messages


def get_class_suggestions_w_descriptors(
    clsA: str, clsB: str, clsA_features: List[str], clsB_features: List[str]
):
    messages = [
        {
            "role": "system",
            "content": """You are a helpful assistant that brainstorms visual features to distinguish between two objects.
You will be provided with a description of each object class and their current feature descriptors. Your task is to suggest new visual features that clearly distinguish these two objects from one another.
Your visual features must be simple and concise. Each feature should be described in 1-2 sentences. Do not explain the features in the same line. Avoid repeating the current descriptors.
Each feature should be followed by reasoning that justifies the feature, especially focusing on why it distinguishes these two classes. Your output should look like this:
```json
{
    "reasoning": ["reasoning for feature1", "reasoning for feature2", ..., "reasoning for featureN"],
    "features": ["feature1", "feature2", ..., "featureN"]
}
```""",
        },
        {
            "role": "user",
            "content": """Please suggest visual features that distinguish a phone from a television.
The current descriptors for a phone include: curved edges, circular cameras on the back, small charging cable.
The current descriptors for a television include: sharp edges, no cameras on the back, large thick power cord.

Focus on suggesting new or refined features that clearly separate these two categories. Keep your suggested features brief and simple. Provide your reasoning separately.""",
        },
        {
            "role": "assistant",
            "content": """```json
{
 "reasoning": [
   "Phones are typically handheld and can be used with one hand, whereas televisions are designed to be stationary and require more space for viewing.",
   "The screen size of phones is generally small, usually ranging from 5 to 7 inches, whereas televisions have much larger screens, starting from around 32 inches and going up significantly larger."
 ],
 "features": [
   "Handheld size",
   "Large screen size"
 ]
}
```""",
        },
        {
            "role": "user",
            "content": f"""Please suggest visual features that distinguish a {clsA} from a {clsB}.
    
The current descriptors for {clsA} include: {', '.join(clsA_features)}.
The current descriptors for {clsB} include: {', '.join(clsB_features)}.

Focus on suggesting new or refined features that clearly separate these two categories. Keep your suggested features brief and simple. Provide your reasoning separately.""",
        },
    ]
    return messages


def get_class_suggestions_w_descriptors_history(
    clsA: str,
    clsB: str,
    clsA_features: List[str],
    clsB_features: List[str],
    conversational_history: dict,
):
    # conversational_history = {iteration: [{clsA, clsB, query, response}, ...]}
    # subselect all iterations in conversational history where clsA and clsB appear
    relevant_conversations = [
        (it, conversation)
        for it, history in conversational_history.items()
        for conversation in history
        if (clsA, clsB) == conversation["classes"]
    ]
    if len(relevant_conversations) == 0:
        return get_class_suggestions_w_descriptors(
            clsA, clsB, clsA_features, clsB_features
        )

    messages = [
        {
            "role": "system",
            "content": """You are a helpful assistant that brainstorms visual features to distinguish between two objects.
You will be provided with a description of each object class and their current feature descriptors. Your task is to suggest new visual features that clearly distinguish these two objects from one another.
Your visual features must be simple and concise. Each feature should be described in 1-2 sentences. Do not explain the features in the same line. Avoid repeating the current descriptors.
Each feature should be followed by reasoning that justifies the feature, especially focusing on why it distinguishes these two classes. Your output should look like this:
```json
{
    "reasoning": ["reasoning for feature1", "reasoning for feature2", ..., "reasoning for featureN"],
    "features": ["feature1", "feature2", ..., "featureN"]
}
```""",
        },
        {
            "role": "user",
            "content": f"""Please suggest visual features that distinguish a {clsA} from a {clsB}.
    
The current descriptors for {clsA} include: {', '.join(clsA_features)}.
The current descriptors for {clsB} include: {', '.join(clsB_features)}.

Focus on suggesting new or refined features that clearly separate these two categories. Keep your suggested features brief and simple. Provide your reasoning separately.""",
        },
    ]
    begin_message = messages[0]
    prompt_message = messages[-1]

    iteration_messages = []
    for it, conversation in relevant_conversations:
        prompt = conversation["query"][-1]
        # prompt['content']
        response = {"role": "assistant", "content": conversation["response"]}
        feedback = {
            "role": "user",
            "content": "After using the previous prompts, the misprediction rate between {clsA} and {clsB} is {feedback}. Use this feedback to improve the model. ".format(
                clsA=clsA, clsB=clsB, feedback=conversation["feedback"]
            ),
        }
        iteration_messages.append(prompt)
        iteration_messages.append(response)
        iteration_messages.append(feedback)

    all_messages = [begin_message] + iteration_messages + [prompt_message]

    return all_messages


def get_class_suggestions_programmatic(
    cls_desc_pairs: List[Tuple[str, str]], correlation
):
    assert len(cls_desc_pairs) == 2
    programs = [
        program_builder(i, cls, desc) for i, (cls, desc) in enumerate(cls_desc_pairs, 1)
    ]
    programs_str = "\n".join(programs)
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "text": 'You are an assistant that generates visual discriminators to help differentiate between two image classification models that often confuse similar categories. I will give you the programs and the target class for each program. Your goal is to modify program1 so that it doesn\'t get confused for program2 anymore. Structure your output in JSON as follows:\n```json\n{\n    "reasoning" ["reasoning for changes to program 1"],\n    "program" ["modified program1"]\n}\n```',
                    "type": "text",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "program1 and program2 have a correlation of {correlation:.2f}% on an image dataset. How should I modify program1 so it does not get confused with program 2?\n```python\n{program_str}\n```".format(
                        correlation=correlation,
                        program_str=programs_str,
                    ),
                }
            ],
        },
    ]
    return messages
