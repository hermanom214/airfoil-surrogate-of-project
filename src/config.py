from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class ProjectPaths:
    project_root: Path
    generated_profiles: Path
    sampling_table: Path
    openfoam_run_root: Path
    openfoam_bash: Path
    openfoam_case_sim: Path
    flow_fields_output: Path


@dataclass
class SamplingConfig:
    chord: float
    aoa_values: list[float]
    inlet_velocity_values: list[float]
    camber_values: list[int]
    camber_position_values: list[int]
    thickness_values: list[int]


@dataclass
class BlockMeshBuildConfig:
    z_half: float
    x_min: float
    x_max: float
    x_far: float
    y_min: float
    y_max: float
    n_airfoil_half: int
    le_cluster_exp: float
    n_streamwise_near: int
    n_wall_normal: int
    n_wake_x: int
    n_z: int
    grading_to_wall: float
    grading_le_tangent: float
    grading_wake_x: float
    n_far_wake_x: int
    grading_far_wake_x: float
    enable_le_cap: bool
    le_cap_fraction: float
    le_cap_power: float
    le_topology_fraction: float
    n_le_cap_normal: int
    te_transition_fraction: float
    wake_cut_length: float


@dataclass
class DatasetBuildConfig:
    template_case_relpath: str
    blockmesh_cases_subdir: str
    sampling: SamplingConfig
    blockmesh: BlockMeshBuildConfig


@dataclass
class SolverRunStep:
    status_key: str | None
    command: list[str]
    log_name: str


@dataclass
class CaseRunnerConfig:
    n_procs: int
    required_case_subdirs: list[str]
    run_steps: list[SolverRunStep]


@dataclass
class FoamToVtkConfig:
    command: list[str]
    log_name: str


@dataclass
class RunCasesConfig:
    max_workers: int
    cases_subdir: str


@dataclass
class SolverConfig:
    case_runner: CaseRunnerConfig
    foam_to_vtk: FoamToVtkConfig
    run_cases: RunCasesConfig


@dataclass
class SimpleUNetModelParamsConfig:
    in_channels: int
    out_channels: int
    encoder_channels: list[int]
    bottleneck_channels: int


@dataclass
class RansPinnModelParamsConfig:
    in_channels: int
    out_channels: int
    hidden_channels: int
    depth: int


@dataclass
class MLModelParamsConfig:
    simple_unet: SimpleUNetModelParamsConfig
    rans_pinn: RansPinnModelParamsConfig


@dataclass
class MLModelConfig:
    name: str
    params: MLModelParamsConfig


@dataclass
class MLTrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    validation_split: float
    split_seed: int


@dataclass
class MLPhysicsLossConfig:
    nu: float
    weight: float
    warmup_epochs: int
    u_scale: float
    p_scale: float
    pressure_is_kinematic: bool
    mask_erode_pixels: int


@dataclass
class MLOutputConfig:
    models_subdir: str
    filename_template: str


@dataclass
class MLModelsConfig:
    model: MLModelConfig
    training: MLTrainingConfig
    physics_loss: MLPhysicsLossConfig
    output: MLOutputConfig


def load_paths(config_path: Path) -> ProjectPaths:
    """
    Načte YAML config s cestami a vrátí je jako Path objekty.
    """
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return ProjectPaths(
        project_root=Path(data["project_root_windows"]),
        generated_profiles=Path(data["generated_profiles_windows"]),
        sampling_table=Path(data["sampling_table_windows"]),
        openfoam_run_root=Path(data["openfoam_run_root_windows"]),
        openfoam_bash=Path(data["openfoam_bash_windows"]), 
        openfoam_case_sim=Path(data["openfoam_case_sim_windows"]),
        flow_fields_output=Path(data["flow_fields_output"]),
    )


def load_dataset_build_config(config_path: Path) -> DatasetBuildConfig:
    """Load dataset/build configuration from dataset_config.yaml."""
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    sampling_data = data["sampling"]
    blockmesh_data = data["blockmesh"]

    sampling = SamplingConfig(
        chord=float(sampling_data["chord"]),
        aoa_values=[float(v) for v in sampling_data["aoa_values"]],
        inlet_velocity_values=[float(v) for v in sampling_data["inlet_velocity_values"]],
        camber_values=[int(v) for v in sampling_data["camber_values"]],
        camber_position_values=[int(v) for v in sampling_data["camber_position_values"]],
        thickness_values=[int(v) for v in sampling_data["thickness_values"]],
    )

    blockmesh = BlockMeshBuildConfig(
        z_half=float(blockmesh_data["z_half"]),
        x_min=float(blockmesh_data["x_min"]),
        x_max=float(blockmesh_data["x_max"]),
        x_far=float(blockmesh_data["x_far"]),
        y_min=float(blockmesh_data["y_min"]),
        y_max=float(blockmesh_data["y_max"]),
        n_airfoil_half=int(blockmesh_data["n_airfoil_half"]),
        le_cluster_exp=float(blockmesh_data["le_cluster_exp"]),
        n_streamwise_near=int(blockmesh_data["n_streamwise_near"]),
        n_wall_normal=int(blockmesh_data["n_wall_normal"]),
        n_wake_x=int(blockmesh_data["n_wake_x"]),
        n_z=int(blockmesh_data["n_z"]),
        grading_to_wall=float(blockmesh_data["grading_to_wall"]),
        grading_le_tangent=float(blockmesh_data["grading_le_tangent"]),
        grading_wake_x=float(blockmesh_data["grading_wake_x"]),
        n_far_wake_x=int(blockmesh_data["n_far_wake_x"]),
        grading_far_wake_x=float(blockmesh_data["grading_far_wake_x"]),
        enable_le_cap=bool(blockmesh_data["enable_le_cap"]),
        le_cap_fraction=float(blockmesh_data["le_cap_fraction"]),
        le_cap_power=float(blockmesh_data["le_cap_power"]),
        le_topology_fraction=float(blockmesh_data["le_topology_fraction"]),
        n_le_cap_normal=int(blockmesh_data["n_le_cap_normal"]),
        te_transition_fraction=float(blockmesh_data["te_transition_fraction"]),
        wake_cut_length=float(blockmesh_data["wake_cut_length"]),
    )

    return DatasetBuildConfig(
        template_case_relpath=str(data["template_case_relpath"]),
        blockmesh_cases_subdir=str(data["blockmesh_cases_subdir"]),
        sampling=sampling,
        blockmesh=blockmesh,
    )


def load_solver_config(config_path: Path) -> SolverConfig:
    """Load OpenFOAM solver/run configuration from solver_config.yaml."""
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    case_runner_data = data["case_runner"]
    foam_to_vtk_data = data["foam_to_vtk"]
    run_cases_data = data["run_cases"]

    run_steps = [
        SolverRunStep(
            status_key=(None if s.get("status_key") is None else str(s["status_key"])),
            command=[str(v) for v in s["command"]],
            log_name=str(s["log_name"]),
        )
        for s in case_runner_data["run_steps"]
    ]

    case_runner = CaseRunnerConfig(
        n_procs=int(case_runner_data["n_procs"]),
        required_case_subdirs=[str(v) for v in case_runner_data["required_case_subdirs"]],
        run_steps=run_steps,
    )

    foam_to_vtk = FoamToVtkConfig(
        command=[str(v) for v in foam_to_vtk_data["command"]],
        log_name=str(foam_to_vtk_data["log_name"]),
    )

    run_cases = RunCasesConfig(
        max_workers=int(run_cases_data["max_workers"]),
        cases_subdir=str(run_cases_data["cases_subdir"]),
    )

    return SolverConfig(
        case_runner=case_runner,
        foam_to_vtk=foam_to_vtk,
        run_cases=run_cases,
    )


def load_ml_models_config(config_path: Path) -> MLModelsConfig:
    """Load ML model/training configuration from ml_models_config.yaml."""
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    model_data = data["model"]
    training_data = data["training"]
    physics_data = data["physics_loss"]
    output_data = data["output"]

    model = MLModelConfig(
        name=str(model_data["name"]),
        params=MLModelParamsConfig(
            simple_unet=SimpleUNetModelParamsConfig(
                in_channels=int(model_data["params"]["simple_unet"]["in_channels"]),
                out_channels=int(model_data["params"]["simple_unet"]["out_channels"]),
                encoder_channels=[
                    int(v) for v in model_data["params"]["simple_unet"]["encoder_channels"]
                ],
                bottleneck_channels=int(model_data["params"]["simple_unet"]["bottleneck_channels"]),
            ),
            rans_pinn=RansPinnModelParamsConfig(
                in_channels=int(model_data["params"]["rans_pinn"]["in_channels"]),
                out_channels=int(model_data["params"]["rans_pinn"]["out_channels"]),
                hidden_channels=int(model_data["params"]["rans_pinn"]["hidden_channels"]),
                depth=int(model_data["params"]["rans_pinn"]["depth"]),
            ),
        ),
    )

    training = MLTrainingConfig(
        batch_size=int(training_data["batch_size"]),
        epochs=int(training_data["epochs"]),
        learning_rate=float(training_data["learning_rate"]),
        validation_split=float(training_data["validation_split"]),
        split_seed=int(training_data["split_seed"]),
    )

    physics_loss = MLPhysicsLossConfig(
        nu=float(physics_data["nu"]),
        weight=float(physics_data["weight"]),
        warmup_epochs=int(physics_data["warmup_epochs"]),
        u_scale=float(physics_data["u_scale"]),
        p_scale=float(physics_data["p_scale"]),
        pressure_is_kinematic=bool(physics_data["pressure_is_kinematic"]),
        mask_erode_pixels=int(physics_data["mask_erode_pixels"]),
    )

    output = MLOutputConfig(
        models_subdir=str(output_data["models_subdir"]),
        filename_template=str(output_data["filename_template"]),
    )

    return MLModelsConfig(
        model=model,
        training=training,
        physics_loss=physics_loss,
        output=output,
    )