import os
import json
import shutil

input_dir = "train_output_resnet50"
output_dir = "scripts_resnet50_seed"
# input_dir = "train_output_vitb16"
# output_dir = "scripts_vitb16_seed"

target_num_files = 20

def collect(expname):
    result_list = []
        # print(expname)
    if len(os.listdir(os.path.join(input_dir, expname))) != target_num_files:
        print(len(os.listdir(os.path.join(input_dir, expname))))
    # assert len(os.listdir(os.path.join(input_dir, expname))) == 2*10
    for file in os.listdir(os.path.join(input_dir, expname)):
        if "hparams" in file:
            continue    
        assert "metrics" in file
        trial = file.split("_metrics")[0]
        with open(os.path.join(input_dir, expname, file), "r") as f:
            result = json.load(f)
        result_list.append((result['final'], trial))
    max_result = sorted(result_list, key=lambda x: x[0]["default"], reverse=True)[0]
    return max_result

for child in os.listdir(input_dir):
    if "py" not in child and "_to_" not in child:
        items = child.split("_")
        dataset, test_domain, algorithm, pretrain = items[:4]
        # if not (algorithm == "W2D" and dataset == "DomainNet" and pretrain == "Supervised" and "-" not in test_domain):
            # continue
        cur_result = collect(child)
        hparams_dir = os.path.join(output_dir, dataset, test_domain)
        os.makedirs(hparams_dir, exist_ok=True)
        shutil.copy(os.path.join(input_dir, child, "%s_hparams.json" % cur_result[1]), hparams_dir)
        os.rename(os.path.join(hparams_dir, "%s_hparams.json" % cur_result[1]), os.path.join(hparams_dir, "%s_%s.json" % (algorithm, pretrain)))
        print(child + ": " + str(cur_result))    