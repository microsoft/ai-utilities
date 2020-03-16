import json
import os
from datetime import datetime

import requests
from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient
from azureml.core import Experiment
from azureml.core.authentication import AzureCliAuthentication
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.datastore import Datastore
from azureml.core.runconfig import CondaDependencies, RunConfiguration
from azureml.core.runconfig import DEFAULT_CPU_IMAGE  # , DEFAULT_GPU_IMAGE
from azureml.data.data_reference import DataReference
from azureml.data.datapath import DataPath, DataPathComputeBinding
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.core.graph import PipelineParameter
from azureml.pipeline.core.schedule import ScheduleRecurrence, Schedule
from azureml.pipeline.steps import PythonScriptStep, MpiStep
from dotenv import set_key

from configuration.notebook_config import project_configuration_file
from configuration.project_configuration import ProjectConfiguration
from machine_learning.contexts.workspace_contexts import WorkspaceContext


def test_ml_batch_score():
    project_config = ProjectConfiguration(configuration_file=project_configuration_file)
    ws = WorkspaceContext.get_or_create_workspace()
    now_str = datetime.now().strftime("%y%m%d%H%M%S")
    # AML workspace and compute target
    AML_WORKSPACE = "ws{}".format(now_str)
    AML_COMPUTE_NAME = "cmp{}".format(now_str)  # limit to 16 chars

    # AML scheduling
    SCHED_FREQUENCY = "Hour"
    SCHED_INTERVAL = 1

    # Scoring script
    PIP_PACKAGES = ["numpy", "scipy", "scikit-learn", "pandas"]
    PYTHON_VERSION = "3.6.7"
    PYTHON_SCRIPT_NAME = "predict.py"
    PYTHON_SCRIPT_DIRECTORY = "scripts"

    # Blob storage
    BLOB_ACCOUNT = "ba{}".format(now_str)  # limit to 24 chars
    MODELS_CONTAINER = "models"
    PREDS_CONTAINER = "preds"
    DATA_CONTAINER = "data"
    DATA_BLOB = "sensor_data.csv"  # name of data file to be copied to blob storage

    blob_datastore = ws.get_default_datastore()

    blob_account = blob_datastore.account_name
    blob_key = blob_datastore.account_key

    # Create models, predictions and data containers
    service = BlobServiceClient(
        account_url="https://" + blob_account + ".blob.core.windows.net/",
        credential=blob_key,
    )

    for container in [MODELS_CONTAINER, PREDS_CONTAINER, DATA_CONTAINER]:
        try:
            service.create_container(container)
        except ResourceExistsError:
            print(str(container) + " - Exists")

    try:
        data_container = service.get_container_client(DATA_CONTAINER)

        import requests

        with requests.get(
            "https://raw.githubusercontent.com/microsoft/az-ml-batch-score/master/data/sensor_data.csv",
            stream=True,
        ) as data:
            data_container.upload_blob("sensor_data.csv", data)

    except ResourceExistsError:
        print("sensor_data.csv - Exists")

    try:
        models_container = service.get_container_client(MODELS_CONTAINER)

        for x in range(1, 5):
            for y in range(1, 3):
                import requests

                with requests.get(
                    "https://raw.githubusercontent.com/microsoft/az-ml-batch-score/master/models/model_"
                    + str(y)
                    + "_"
                    + str(x),
                    stream=True,
                ) as data:
                    models_container.upload_blob("model_" + str(y) + "_" + str(x), data)

    except ResourceExistsError:
        print("models dir - Exists")

    pipeline_config = {
        "resource_group_name": project_config.get_value("resource_group"),
        "subscription_id": project_config.get_value("subscription_id"),
        "aml_work_space": project_config.get_value("workspace_name"),
        "experiment_name": "mm_score",
        "cluster_name": AML_COMPUTE_NAME,
        "workspace_region": project_config.get_value("workspace_region"),
        "blob_account": BLOB_ACCOUNT,
        "blob_key": blob_key,
        "models_blob_container": MODELS_CONTAINER,
        "data_blob_container": DATA_CONTAINER,
        "data_blob": DATA_BLOB,
        "preds_blob_container": PREDS_CONTAINER,
        "pip_packages": PIP_PACKAGES,
        "python_version": PYTHON_VERSION,
        "python_script_name": PYTHON_SCRIPT_NAME,
        "python_script_directory": PYTHON_SCRIPT_DIRECTORY,
        "sched_frequency": SCHED_FREQUENCY,
        "sched_interval": SCHED_INTERVAL,
        "device_ids": [1, 2, 3],
        "sensors": [1, 2, 3, 4, 5],
    }
    with open("pipeline_config.json", "w") as f:
        json.dump(pipeline_config, f, indent=4)

    AML_VM_SIZE = "Standard_D2"
    AML_MIN_NODES = 2
    AML_MAX_NODES = 2

    provisioning_config = AmlCompute.provisioning_configuration(
        vm_size=AML_VM_SIZE, min_nodes=AML_MIN_NODES, max_nodes=AML_MAX_NODES
    )
    j = pipeline_config
    compute_target = ComputeTarget.create(ws, j["cluster_name"], provisioning_config)
    compute_target.wait_for_completion(show_output=True)

    # Pipeline input and output
    data_ds = Datastore.register_azure_blob_container(
        ws,
        datastore_name="data_ds",
        container_name=j["data_blob_container"],
        account_name=blob_account,
        account_key=blob_key,
    )
    data_dir = DataReference(datastore=data_ds, data_reference_name="data")

    models_ds = Datastore.register_azure_blob_container(
        ws,
        datastore_name="models_ds",
        container_name=j["models_blob_container"],
        account_name=blob_account,
        account_key=blob_key,
    )
    models_dir = DataReference(datastore=models_ds, data_reference_name="models")

    preds_ds = Datastore.register_azure_blob_container(
        ws,
        datastore_name="preds_ds",
        container_name=j["preds_blob_container"],
        account_name=blob_account,
        account_key=blob_key,
    )
    preds_dir = PipelineData(name="preds", datastore=preds_ds, is_directory=True)

    # Run config
    conda_dependencies = CondaDependencies.create(
        pip_packages=j["pip_packages"], python_version=j["python_version"]
    )
    run_config = RunConfiguration(conda_dependencies=conda_dependencies)
    run_config.environment.docker.enabled = True

    if not os.path.isfile(j["python_script_directory"] + "/" + j["python_script_name"]):
        os.makedirs(j["python_script_directory"], exist_ok=True)

        create_model_py = """
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle
import sys
import json
import datetime
import pandas as pd
import io
import os


# query params
device = int(sys.argv[1])
sensor = int(sys.argv[2])
models_dir = sys.argv[3]
data_dir = sys.argv[4]
data_file_name = sys.argv[5]
preds_dir = sys.argv[6]

model_name = "model_{0}_{1}".format(device, sensor)

# get data
data = pd.read_csv(data_dir + "/" + data_file_name)
data = data[(data["Device"] == device) & (data["Sensor"] == sensor)]
tss = data["TS"]
vals = np.array(data["Value"])

# load model
print("Loading model")
with open(models_dir + "/" + model_name, "rb") as f:
    pipe = pickle.load(f)


# predict
preds = pipe.predict(vals.reshape(-1, 1))
preds = np.where(preds == 1, 0, 1)  # 1 indicates an anomaly, 0 otherwise

# csv results
res = pd.DataFrame(
    {
        "TS": tss,
        "Device": np.repeat(device, len(preds)),
        "Sensor": np.repeat(sensor, len(preds)),
        "Val": vals,
        "Prediction": preds,
    }
)
res = res[["TS", "Device", "Sensor", "Val", "Prediction"]]

res_file_name = "preds_{0}_{1}_{2}.csv".format(
    device, sensor, datetime.datetime.now().strftime("%y%m%d%H%M%S")
)


# save predictions
print("Saving predictions")
os.makedirs(preds_dir)
with open(preds_dir + "/" + res_file_name, "w") as f:
    res.to_csv(f, index=None)
        
"""
        with open(
            j["python_script_directory"] + "/" + j["python_script_name"], "w"
        ) as file:
            file.write(create_model_py)

    # Create a pipeline step for each (device, sensor) pair
    steps = []
    for device_id in j["device_ids"]:
        for sensor in j["sensors"]:
            preds_dir = PipelineData(
                name="preds", datastore=preds_ds, is_directory=True
            )
            step = PythonScriptStep(
                name="{}_{}".format(device_id, sensor),
                script_name=j["python_script_name"],
                arguments=[
                    device_id,
                    sensor,
                    models_dir,
                    data_dir,
                    j["data_blob"],
                    preds_dir,
                ],
                inputs=[models_dir, data_dir],
                outputs=[preds_dir],
                source_directory=j["python_script_directory"],
                compute_target=compute_target,
                runconfig=run_config,
                allow_reuse=False,
            )
            steps.append(step)

    pipeline = Pipeline(workspace=ws, steps=steps)
    pipeline.validate()

    # Publish pipeline
    pipeline_name = "scoring_pipeline_{}".format(datetime.now().strftime("%y%m%d%H%M"))
    published_pipeline = pipeline.publish(name=pipeline_name, description=pipeline_name)

    # Schedule pipeline
    experiment_name = "exp_" + datetime.now().strftime("%y%m%d%H%M%S")
    recurrence = ScheduleRecurrence(
        frequency=j["sched_frequency"], interval=j["sched_interval"]
    )
    schedule = Schedule.create(
        workspace=ws,
        name="{}_sched".format(j["resource_group_name"]),
        pipeline_id=published_pipeline.id,
        experiment_name=experiment_name,
        recurrence=recurrence,
        description="{}_sched".format(j["resource_group_name"]),
    )


def test_deep_batch_score():
    project_config = ProjectConfiguration(configuration_file=project_configuration_file)
    ws = WorkspaceContext.get_or_create_workspace()

    # Also create a Project and attach to Workspace
    project_folder = "scripts"
    run_history_name = project_folder

    if not os.path.isdir(project_folder):
        os.mkdir(project_folder)

    style_transfer_node_count = 4
    ffmpeg_node_count = 1

    vm_dict = {
        "NC": {"size": "STANDARD_NC6", "cores": 6},
        "NCSv3": {"size": "STANDARD_NC6s_v3", "cores": 6},
        "DSv2": {"size": "STANDARD_DS3_V2", "cores": 4},
    }

    cpu_cluster_name = "ffmpeg-cluster"
    try:
        cpu_cluster = AmlCompute(ws, cpu_cluster_name)
        print("Found existing cluster.")
    except:
        print("Creating {}".format(cpu_cluster_name))
        provisioning_config = AmlCompute.provisioning_configuration(
            vm_size=vm_dict["DSv2"]["size"],
            min_nodes=ffmpeg_node_count,
            max_nodes=ffmpeg_node_count,
        )

        # create the cluster
        cpu_cluster = ComputeTarget.create(ws, cpu_cluster_name, provisioning_config)
        cpu_cluster.wait_for_completion(show_output=True)

    # GPU compute
    gpu_cluster_name = "style-cluster"
    try:
        gpu_cluster = AmlCompute(ws, gpu_cluster_name)
        print("Found existing cluster.")
    except:
        print("Creating {}".format(gpu_cluster_name))
        provisioning_config = AmlCompute.provisioning_configuration(
            vm_size=vm_dict["NC"]["size"],
            min_nodes=style_transfer_node_count,
            max_nodes=style_transfer_node_count,
        )

        # create the cluster
        gpu_cluster = ComputeTarget.create(ws, gpu_cluster_name, provisioning_config)
        gpu_cluster.wait_for_completion(show_output=True)

    my_datastore = ws.get_default_datastore()

    # Upload files in models folder to a directory called models
    my_datastore.upload_files(
        ["./models/model.pth"], target_path="models", overwrite=True
    )

    # Upload orangutan.mp4 video
    my_datastore.upload_files(["./orangutan.mp4"], overwrite=True)

    model_dir = DataReference(
        data_reference_name="model_dir",
        datastore=my_datastore,
        path_on_datastore="models",
        mode="download",
    )

    output_video = PipelineData(name="output_video", datastore=my_datastore)

    default_datastore = ws.get_default_datastore()

    ffmpeg_audio = PipelineData(name="ffmpeg_audio", datastore=default_datastore)
    ffmpeg_images = PipelineData(name="ffmpeg_images", datastore=default_datastore)
    processed_images = PipelineData(
        name="processed_images", datastore=default_datastore
    )

    ffmpeg_cd = CondaDependencies()
    ffmpeg_cd.add_channel("conda-forge")
    ffmpeg_cd.add_conda_package("ffmpeg")

    ffmpeg_run_config = RunConfiguration(conda_dependencies=ffmpeg_cd)
    ffmpeg_run_config.environment.docker.enabled = True
    ffmpeg_run_config.environment.docker.gpu_support = False
    ffmpeg_run_config.environment.docker.base_image = DEFAULT_CPU_IMAGE
    ffmpeg_run_config.environment.spark.precache_packages = False

    style_transfer_cd = CondaDependencies()
    style_transfer_cd.add_channel("pytorch")
    style_transfer_cd.add_conda_package("pytorch")

    style_transfer_run_config = RunConfiguration(conda_dependencies=style_transfer_cd)
    style_transfer_run_config.environment.docker.enabled = True
    style_transfer_run_config.environment.docker.gpu_support = True
    style_transfer_run_config.environment.docker.base_image = "pytorch/pytorch"
    style_transfer_run_config.environment.spark.precache_packages = False

    video_path_default = DataPath(
        datastore=my_datastore, path_on_datastore="orangutan.mp4"
    )
    video_path_param = (
        PipelineParameter(name="video_path", default_value=video_path_default),
        DataPathComputeBinding(),
    )

    if not os.path.isfile("scripts/preprocess_video.py"):
        os.makedirs("scripts", exist_ok=True)

        create_model_py = """
import argparse
import glob
import os
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input video")
    parser.add_argument(
        '--input-video', 
        help="Path to the input video file (include ext)",
        required=True
    )
    parser.add_argument(
        '--output-audio', 
        help="The name of the output folder to store the audio clip in.",
        required=True
    )
    parser.add_argument(
        '--output-images', 
        help="The name of the output image folder to store the output frames in.",
        required=True
    )

    args = parser.parse_args()

    os.makedirs(args.output_audio, exist_ok=True)
    os.makedirs(args.output_images, exist_ok=True)

    subprocess.run("ffmpeg -i {} {}/audio.aac"
                  .format(args.input_video, args.output_audio),
                   shell=True, check=True
                  )

    subprocess.run("ffmpeg -i {} {}/%05d_video.jpg -hide_banner"
                  .format(args.input_video, args.output_images),
                   shell=True, check=True
                  )

"""
        with open("scripts/preprocess_video.py", "w") as file:
            file.write(create_model_py)

    preprocess_video_step = PythonScriptStep(
        name="preprocess video",
        script_name="preprocess_video.py",
        arguments=[
            "--input-video",
            video_path_param,
            "--output-audio",
            ffmpeg_audio,
            "--output-images",
            ffmpeg_images,
        ],
        compute_target=cpu_cluster,
        inputs=[video_path_param],
        outputs=[ffmpeg_images, ffmpeg_audio],
        runconfig=ffmpeg_run_config,
        source_directory=project_folder,
        allow_reuse=False,
    )

    if not os.path.isfile("scripts/style_transfer_mpi.py"):
        os.makedirs("scripts", exist_ok=True)

        create_model_py = """
# Original source: https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/neural_style.py
import argparse
import os
import sys
import re

from PIL import Image
import torch
from torchvision import transforms

from mpi4py import MPI


def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img


def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


class TransformerNet(torch.nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()

    def forward(self, X):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(mode='nearest', scale_factor=upsample)
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out
    

def stylize(args, comm):

    rank = comm.Get_rank()
    size = comm.Get_size()

    device = torch.device("cuda" if args.cuda else "cpu")
    with torch.no_grad():
        style_model = TransformerNet()
        state_dict = torch.load(os.path.join(args.model_dir, "model.pth"))
        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]
        style_model.load_state_dict(state_dict)
        style_model.to(device)

        filenames = os.listdir(args.content_dir)
        filenames = sorted(filenames)
        partition_size = len(filenames) // size
        partitioned_filenames = filenames[rank * partition_size : (rank + 1) * partition_size]
        print("RANK {} - is processing {} images out of the total {}".format(rank, len(partitioned_filenames), len(filenames)))

        output_paths = []
        for filename in partitioned_filenames:
            # print("Processing {}".format(filename))
            full_path = os.path.join(args.content_dir, filename)
            content_image = load_image(full_path, scale=args.content_scale)
            content_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.mul(255))
            ])
            content_image = content_transform(content_image)
            content_image = content_image.unsqueeze(0).to(device)

            output = style_model(content_image).cpu()

            output_path = os.path.join(args.output_dir, filename)
            save_image(output_path, output[0])

            output_paths.append(output_path)

        print("RANK {} - number of pre-aggregated output files {}".format(rank, len(output_paths)))

        output_paths_list = comm.gather(output_paths, root=0)
        
        if rank == 0:
          print("RANK {} - number of aggregated output files {}".format(rank, len(output_paths_list)))
          print("RANK {} - end".format(rank))


def main():
    arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")

    arg_parser.add_argument("--content-scale", type=float, default=None,
                                 help="factor for scaling down the content image")
    arg_parser.add_argument("--model-dir", type=str, required=True,
                                 help="saved model to be used for stylizing the image.")
    arg_parser.add_argument("--cuda", type=int, required=True,
                                 help="set it to 1 for running on GPU, 0 for CPU")
    arg_parser.add_argument("--content-dir", type=str, required=True,
            help="directory holding the images")
    arg_parser.add_argument("--output-dir", type=str, required=True,
            help="directory holding the output images")
    args = arg_parser.parse_args()

    comm = MPI.COMM_WORLD
    
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)
    os.makedirs(args.output_dir, exist_ok=True)
    stylize(args, comm)


if __name__ == "__main__":
    main()

"""
        with open("scripts/style_transfer_mpi.py", "w") as file:
            file.write(create_model_py)

    distributed_style_transfer_step = MpiStep(
        name="mpi style transfer",
        script_name="style_transfer_mpi.py",
        arguments=[
            "--content-dir",
            ffmpeg_images,
            "--output-dir",
            processed_images,
            "--model-dir",
            model_dir,
            "--cuda",
            1,
        ],
        compute_target=gpu_cluster,
        node_count=4,
        process_count_per_node=1,
        inputs=[model_dir, ffmpeg_images],
        outputs=[processed_images],
        pip_packages=["image", "mpi4py", "torch", "torchvision"],
        runconfig=style_transfer_run_config,
        use_gpu=True,
        source_directory=project_folder,
        allow_reuse=False,
    )

    if not os.path.isfile("scripts/postprocess_video.py"):
        os.makedirs("scripts", exist_ok=True)

        create_model_py = """
import argparse
import os
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input video")
    parser.add_argument(
        '--video',
        help="Name of the output video (excluding ext)"
    )
    parser.add_argument(
        '--images-dir', 
        help="The input image directory of frames to stitch together.",
        required=True
    )
    parser.add_argument(
        '--input-audio', 
        help="The input audio directory containing the audio file.",
        required=True
    )
    parser.add_argument(
        '--output-dir', 
        help="The output directory to save the stitched-together video into.",
        required=True
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    subprocess.run("ffmpeg -framerate 30 -i {}/%05d_video.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p "
                   "-y {}/video_without_audio.mp4"
                   .format(args.images_dir, args.output_dir),
                   shell=True, check=True
                  )

    video_name = args.video or 'video'

    subprocess.run("ffmpeg -i {}/video_without_audio.mp4 -i {}/audio.aac -map 0:0 -map 1:0 -vcodec "
                   "copy -acodec copy -y {}/{}_processed.mp4"
                   .format(args.output_dir, args.input_audio, args.output_dir, video_name),
                   shell=True, check=True
                  )
        
"""
        with open("scripts/postprocess_video.py", "w") as file:
            file.write(create_model_py)

    postprocess_video_step = PythonScriptStep(
        name="postprocess video",
        script_name="postprocess_video.py",
        arguments=[
            "--images-dir",
            processed_images,
            "--input-audio",
            ffmpeg_audio,
            "--output-dir",
            output_video,
        ],
        compute_target=cpu_cluster,
        inputs=[processed_images, ffmpeg_audio],
        outputs=[output_video],
        runconfig=ffmpeg_run_config,
        source_directory=project_folder,
        allow_reuse=False,
    )

    steps = [postprocess_video_step]
    pipeline = Pipeline(workspace=ws, steps=steps)
    pipeline_run = Experiment(ws, "style_transfer_mpi").submit(
        pipeline,
        pipeline_params={
            "video_path": DataPath(
                datastore=my_datastore, path_on_datastore="orangutan.mp4"
            )
        },
    )

    pipeline_run.wait_for_completion(show_output=True)

    step_id = pipeline_run.find_step_run("postprocess video")[0].id

    my_datastore.download(
        target_path="aml_test_orangutan", prefix=step_id,
    )

    published_pipeline = pipeline.publish(
        name="style transfer", description="some description"
    )

    cli_auth = AzureCliAuthentication()
    aad_token = cli_auth.get_authentication_header()

    response = requests.post(
        published_pipeline.endpoint,
        headers=aad_token,
        json={
            "ExperimentName": "My_Pipeline",
            "DataPathAssignments": {
                "video_path": {
                    "DataStoreName": "workspaceblobstore",
                    "RelativePath": "orangutan.mp4",
                }
            },
        },
    )

    run_id = response.json()["Id"]
    print(run_id)
