import os
import subprocess

import argparse
import yaml

def print_header(text):
    ENDC = '\033[m' # reset to the defaults
    TGREEN =  '\033[32m' # Green Text
    print(TGREEN+"-"*len(text))
    print(text)
    print("-"*len(text), ENDC)


parser = argparse.ArgumentParser(description='Execute learning pipeline')
parser.add_argument("pipeline_path")

args = parser.parse_args()

with open(args.pipeline_path, "r") as pipeline_file:
    config = yaml.load(pipeline_file)


print_header("Generating fields")
field_paths = []
for field_description in config["fields"]:
    field_paths.append(field_description["path"].format(**config["setup"]))
    if "ai" in field_description:
        ai = field_description["ai"].format(**config["setup"])
        ai_data = field_description["ai-data"].format(**config["setup"])
        nr_fields = field_description["nr-fields"]
        path = field_description["path"].format(**config["setup"])
        print("Using:", ai, ai_data, "->", path)
        err = os.system(f"python generate_fields.py --ai {ai} --ai-data {ai_data} --nr-fields {nr_fields} --dump {path}")
        if err:
            exit()

print_header("Generating scores")
for score_description in config["score"]:
    ai = score_description["ai"].format(**config["setup"])
    ai_data = score_description["ai-data"].format(**config["setup"])
    path = score_description["path"].format(**config["setup"])
    nr_fields = score_description["nr-fields"]
    strategy = score_description["strategy"]
    fields = " ".join(field_paths)
    print("Using:", ai, ai_data, "->", path)
    err = os.system(f"python generate_new.py --ai {ai} --ai-data {ai_data} --nr-fields {nr_fields} --dump {path} --score {strategy} --fields {fields}")
    if err:
        exit()

print_header("Learning")
evaluations = []
for learning_description in config["learn"]:
    ai = learning_description["ai"].format(**config["setup"])
    if learning_description.get("ai-data-in"):
        ai_data_in = "--ai-data-in " + learning_description["ai-data-in"].format(**config["setup"])
    else:
        ai_data_in = ""
    ai_data_out = learning_description["ai-data-out"].format(**config["setup"])
    batch_size = learning_description["batch-size"]
    training_data = learning_description["training-data"].format(**config["setup"])
    epochs = learning_description["epochs"]
    print("Learning:", ai, training_data, "(", epochs, batch_size, ") ->", ai_data_out)
    err = os.system(f"""python learn.py --ai {ai} {ai_data_in} --training-data {training_data} --epochs {epochs} --batch-size {batch_size} --ai-data-out {ai_data_out}""")
    if err:
        exit()
    print("Evaluating")
    result = subprocess.run(["python", "play.py", "--ai", ai, "--ai-data", ai_data_out, "--nr-games", "5", "--show", "avg_pieces"], stdout=subprocess.PIPE).stdout.decode('utf-8')
    score = float(result.split("\n")[-2].split()[0])
    evaluations.append((ai_data_out, score))
print(evaluations)
evaluations.sort(key=lambda x: x[1])
for model, score in evaluations:
    print(model, score)


print("DONE")
