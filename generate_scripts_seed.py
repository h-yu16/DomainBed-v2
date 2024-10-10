import os
lp=False
arch="resnet50" # vitb16
linear_probe = " --linear_probe" if lp else ""
lp_prefix = "_lp" if lp else ""


algorithms = ["ERM", "Fishr", "RSC", "SWAD", "CORAL", "IRM", "GroupDRO", "Mixup", "MMD", "SagNet"]

domain_dict = {
    "PACS": ["photo", "art", "cartoon", "sketch"],
    "VLCS": ["PASCAL", "LABELME", "CALTECH", "SUN"],
    "OfficeHome": ["Art", "Clipart", "Product", "Real"],
    "TerraInc": ["L38", "L43", "L46", "L100"],
    "DomainNet": ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"],
    "NICO": ['autumn rock', 'dim grass', 'outdoor water'],
}


pretrains = ["Supervised", "MoCo-v2"] # MoCo-v3
seed = 1
min_data_seed=2
max_data_seed=3

for dataset, domain_list in domain_dict.items():
    os.makedirs(dataset, exist_ok=True)
    for domain in domain_list:
        domain_concat = "-".join(domain.split(" "))
        os.makedirs(os.path.join(dataset, domain_concat), exist_ok=True)
        for alg in algorithms:
            for pretrain in pretrains:
                result_name = "%s_%s_%s_%s_%s"%(dataset, domain_concat, alg, pretrain, arch)
                hparams_config = os.path.join("scripts_%s%s_seed"% (arch, lp_prefix), dataset, domain_concat, "%s_%s.json"% (alg, pretrain))
                with open(os.path.join(dataset, domain_concat, "%s_%s.sh"% (alg, pretrain)), "w") as f:
                    source = " ".join([d for d in domain_list if d != domain])
                    f.write("gpu=1\n")
                    f.write("min=%d\nmax=%d\n" % (min_data_seed, max_data_seed))
                    f.write("seed=%d\n" % seed)
                    f.write("for dataseed in `seq $min $max`\ndo\n")
                    arg_alg = alg if alg != "SWAD" else "ERM --swad"
                    f.write("CUDA_VISIBLE_DEVICES=$gpu python -m domainbed.scripts.train \\\n--arch %s --pretrain %s --algorithm %s%s \\\n--dataset %s \\\n--source %s --target %s \\\n--seed $seed --data_seed $dataseed \\\n--hparams_fixed_config %s \\\n--result_name %s --output_dir train_output_%s%s_seed\n" % (arch, pretrain, arg_alg, linear_probe, dataset, source, domain, hparams_config, result_name, arch, lp_prefix))
                    f.write("done\n")

