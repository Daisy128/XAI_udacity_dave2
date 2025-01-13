import os
from perturbationdrive.perturb import perturbed_simulate as run_perturb_tool

if __name__ == '__main__':

    data={}
    data['model_name'] = "track1-steer-throttle.h5"
    data['model_path'] = os.path.join("./model/ckpts/ads", data['model_name'])
    data['perturbations'] = ["static_lightning_filter", "static_object_overlay"]
    data['track_index'] = 1

    run_perturb_tool(data)

    print("Finished all, exit")