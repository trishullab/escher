import json

from .gpt_utils import get_completion, gpt_parse_suggestions

initial_prompt = """You are a helpful assistant that helps people distinguish between objects by generating high quality visual descriptions and attributes. Our algorithm is having trouble distinguishing a {clsA} from a {clsB}. Please generate new visual attributes or descriptors of {clsA} and {clsB} that could help distinguish the two classes.
Our algorithm runs many iterations. In each iteration, it produces a list of descriptors for {clsA} and a list of descriptors for {clsB}. For each iteration, you will be given the following information:
1) A list of descriptors for {clsA}
2) A list of descriptors for {clsB}
3) The proportion of real instances of {clsA} that were mistaken for {clsB}.

Your goal is to output a list of descriptors for {clsA} and a list of descriptors for {clsB} that will decrease the proportion of real instances of {clsA} that were mistaken for {clsB} as low as possible.
Output two lists of descriptors: one for {clsA}, the other for {clsB}."""


def message_builder(
    clsA, clsB, cls2concepts_history, confusion_matrix_history, classes, distance_type
):
    bar = "=" * 20
    per_iteration_format = """iteration {i}:
{descriptors}
Proportion of real instances of {clsA} that were mistaken for {clsB}: {error_rate:.4f}"""
    per_iteration_format = bar + "\n" + per_iteration_format + "\n"

    clsA_descriptor_history = cls2concepts_history[clsA]
    clsB_descriptor_history = cls2concepts_history[clsB]
    prompt = "History start:"
    iter_reported = 0
    # basically, for each pair of descriptor lists, we only want
    # the most recent one, if there are duplicates
    descriptor_list_pairs = list(zip(clsA_descriptor_history, clsB_descriptor_history))
    same_pair_later = []
    for i, p in enumerate(descriptor_list_pairs):
        if p in descriptor_list_pairs[i + 1 :]:
            same_pair_later.append(1)
        else:
            same_pair_later.append(0)
    for i in range(len(clsA_descriptor_history)):
        if same_pair_later[i]:
            continue
        confusion_matrix = confusion_matrix_history[i]
        clsA_predictions = confusion_matrix[classes.index(clsA)]
        # import IPython; IPython.embed()
        if distance_type == "pearson":
            error_rate = clsA_predictions[classes.index(clsA)]
        elif distance_type == "pca" or distance_type == "emd":
            error_rate = (
                1 - clsA_predictions[classes.index(clsB)] / clsA_predictions.max()
            )
        else:
            error_rate = clsA_predictions[classes.index(clsB)] / clsA_predictions.sum()

        descriptors = json.dumps(
            {
                clsA: clsA_descriptor_history[i],
                clsB: clsB_descriptor_history[i],
            },
            indent=4,
        )
        prompt += per_iteration_format.format(
            i=iter_reported,
            clsA=clsA,
            clsB=clsB,
            descriptors=descriptors,
            error_rate=error_rate,
        )
        iter_reported += 1
    prompt += bar
    prompt += "History end\n"
    prompt += f"Think critically about how the model's performance can be improved. Based on your reasoning, suggest visual features that distinguish a {clsA} from a {clsB}. Use the past history to help in your decision. Make sure the visual features you output decrease the proportion of instances of {clsA} that were mistaken for {clsB}. Your features should be non-trivial, insightful, novel, and accurate. Answers in short phrases are preferred. For example: Prefer 'pointed snout' to 'Lemurs have a pointed snout compared to chimpanzees.' "
    prompt += f"""Your output must follow the following format:
{clsA}:
- descriptor one
- descriptor two
- ...
- descriptor N

{clsB}:
- descriptor one
- descriptor two
- ...
- descriptor N"""
    messages = [
        {
            "role": "system",
            "content": initial_prompt.format(clsA=clsA, clsB=clsB),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]
    return messages


def get_llm_output(
    clsA,
    clsB,
    cls2concepts_history,
    confusion_matrix_history,
    classes,
    openai_model,
    openai_temp,
    distance_type,
):
    completion = get_completion(
        model=openai_model,
        messages=message_builder(
            clsA,
            clsB,
            cls2concepts_history,
            confusion_matrix_history,
            classes,
            distance_type=distance_type,
        ),
        temperature=openai_temp,
    )
    content = completion.choices[0].message.content
    # try:
    #     json_d = gpt_parse_suggestions(content)
    #     return json_d[clsA], json_d[clsB]
    # except Exception as e:
    try:
        content = content.strip().split("\n")
        content = [line.strip("- ").strip("-").strip() for line in content]
        assert content.count("") == 1, "Output should have exactly one empty line"
        idx = content.index("")
        return content[1:idx], content[idx + 2 :]
    except Exception as e:
        print(e)
        json_d = gpt_parse_suggestions(content)
        return json_d[clsA], json_d[clsB]
