import importlib
import os
import requests
import sys
import yaml

from argparse import ArgumentParser
from git import Repo

GITHUB_UPSTREAM_RELEASE_ENDPOINT = "https://api.github.com/repos/pytorch/pytorch/releases"

def load_config(filename="config.yml"):
    with open(filename, "rb") as f:
        config = yaml.safe_load(f)
    return config

def load_template(filename="template.html"):
    with open(filename, "r", encoding="utf8") as f:
        template = f.read()
    return template

def checkout(repo_path, commit):
    pytorch_repo = Repo(repo_path)

    if commit == "latest":
        commit = requests.get(GITHUB_UPSTREAM_RELEASE_ENDPOINT).json()[0]["tag_name"]
    if commit not in pytorch_repo.heads:
        pytorch_repo.create_head(commit, pytorch_repo.remotes[0].fetch(commit)[0])
    pytorch_repo.heads[commit].checkout()
    return commit

def load_supported_ops(commit="main"):
    pytorch_path = os.path.abspath("./pytorch")
    commit = checkout(pytorch_path, commit)
    if sys.path[0] != pytorch_path:
        sys.path.insert(0, pytorch_path)
    gen_module = importlib.import_module("torchgen.gen")
    model = importlib.import_module("torchgen.model")
    
    # The reload order matters
    importlib.reload(model)
    importlib.reload(gen_module)

    aten_path = "pytorch/aten/src/ATen/"
    native_yaml_path = os.path.join(aten_path, "native/native_functions.yaml")
    tags_yaml_path = os.path.join(aten_path, "native/tags.yaml")

    parsed_yaml = gen_module.parse_native_yaml(native_yaml_path, tags_yaml_path)

    native_functions, backend_indices = (
        parsed_yaml.native_functions,
        parsed_yaml.backend_indices,
    )

    MPS_supported_ops = {}
    for k, v in backend_indices[model.DispatchKey.MPS].index.items():
        function_name = f"{k.name.base}{ '.' + k.overload_name if k.overload_name != '' else ''}"
        kernel_name = v.kernel
        structured = v.structured
        cpp_namespace = v.cpp_namespace
        MPS_supported_ops[function_name] = {
            "kernel": kernel_name,
            "structuted": structured,
            "cpp_namespace": cpp_namespace,
        }
    return MPS_supported_ops, commit

def update(output_file="index.html"):
    config = load_config()
    template = load_template()

    master_supported_ops, _ = load_supported_ops()
    latest_release_supported_ops, latest_release_commit = load_supported_ops("latest")

    table_header = f"""<tr>
                <th data-sortable="true">Function name</th>
                <th data-sortable="true">Kernel Name</th>
                <th data-sortable="true">In {latest_release_commit}</th>
                <th data-sortable="true">Starting from macOS version</th>
                <th>Note</th>
            </tr>"""
    table_rows = []

    defaults = {
        "starting_macOS_version": "N/A",
        "note": "N/A",
    }
    
    master_supported_ops.update(config["external"])

    for k, v in master_supported_ops.items():
        included_in_latest = True if k in latest_release_supported_ops else False

        if k not in config["details"]:
            config["details"][k] = {}

        included_in_latest = config["details"][k].get("included_in_latest", included_in_latest)
        included_in_latest_color_code = "#0ff000" if included_in_latest else "#fff000"
        starting_macOS_version = config["details"][k].get("starting_macOS_version", defaults["starting_macOS_version"])
        note = config["details"][k].get("note", defaults["note"])

        table_rows.append(f"<tr><td>{k}</td><td>{v['kernel']}</td><td style='background-color:{included_in_latest_color_code};'>{included_in_latest}</td><td>{starting_macOS_version}</td><td>{note}</td></tr>")
    
    with open(output_file, "w", encoding="utf8") as f:
        f.write(template.format(table_header, "\n".join(table_rows)))
    
if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("--task", choices=["check", "update"], required=True)

    args = parser.parse_args()

    if args.task == "update":
        update()