import wandb
api = wandb.Api()
def add_rn(name):
    return f"deyu-zou20-the-chinese-university-of-hong-kong/RAG-webqsp/runs/{name}"
metrics = ["corrected/f1", "original/hit", "corrected_sub/f1", "corrected_sub/micro_f1", "original_sub/hit", "corrected_sub/hit", "corrected/hal_score"]
run_list = [
    add_rn("6pjektpl")
]
for run in run_list:
    run = api.run(run)
    run.summary['corrected_sub/micro_f1'] *= 100
    print(' '.join([str(round(run.summary[each_metric], 2)) for each_metric in metrics]))