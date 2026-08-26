from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
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
class ClCdMlpModelParamsConfig:
    input_dim: int
    hidden_dims: list[int]
    output_dim: int
    dropout: float
    cases_subdir: str
    force_coeffs_relpath: str
    tail_window: int


@dataclass
class MLModelParamsConfig:
    simple_unet: SimpleUNetModelParamsConfig
    rans_pinn: RansPinnModelParamsConfig
    clcd_mlp: ClCdMlpModelParamsConfig


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
class MLDataSplitConfig:
    test_fraction: float = 0.2
    seed: int = 42
    cv_strategy: str = "kfold"
    n_folds: int = 5
    cv_enabled: bool = True
    regenerate_on_dataset_change: bool = False


@dataclass
class MLHyperparameterSearchConfig:
    strategy: str = "none"
    n_iter: int = 1
    seed: int = 42
    search_space: dict[str, list[Any]] = field(default_factory=dict)
    manual_coarse_configs: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class MLExperimentConfig:
    data_split: MLDataSplitConfig
    hyperparameter_search: MLHyperparameterSearchConfig


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
    device: str
    model: MLModelConfig
    training: MLTrainingConfig
    physics_loss: MLPhysicsLossConfig
    output: MLOutputConfig
    data_split: MLDataSplitConfig
    hyperparameter_search: MLHyperparameterSearchConfig
    experiments: dict[str, MLExperimentConfig]

    def experiment_for(self, model_name: str) -> MLExperimentConfig:
        return self.experiments.get(
            model_name, MLExperimentConfig(self.data_split, self.hyperparameter_search)
        )


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

    device = str(data.get("device", "auto")).strip().lower()
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {device!r}; expected auto, cpu or cuda")
    model_data = data["model"]
    training_data = data["training"]
    physics_data = data["physics_loss"]
    output_data = data["output"]
    split_data = data.get("data_split", {})
    search_data = data.get("hyperparameter_search", {})

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
            clcd_mlp=ClCdMlpModelParamsConfig(
                input_dim=int(model_data["params"]["clcd_mlp"]["input_dim"]),
                hidden_dims=[int(v) for v in model_data["params"]["clcd_mlp"]["hidden_dims"]],
                output_dim=int(model_data["params"]["clcd_mlp"]["output_dim"]),
                dropout=float(model_data["params"]["clcd_mlp"]["dropout"]),
                cases_subdir=str(model_data["params"]["clcd_mlp"]["cases_subdir"]),
                force_coeffs_relpath=str(model_data["params"]["clcd_mlp"]["force_coeffs_relpath"]),
                tail_window=int(model_data["params"]["clcd_mlp"]["tail_window"]),
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

    def parse_split(values: dict[str, Any]) -> MLDataSplitConfig:
        merged = {**split_data, **values}
        strategy = str(merged.get("cv_strategy", "kfold"))
        if strategy not in {"kfold", "group_kfold_naca"}:
            raise ValueError(f"Unsupported cv_strategy: {strategy}")
        return MLDataSplitConfig(
            test_fraction=float(merged.get("test_fraction", 0.2)),
            seed=int(merged.get("seed", 42)),
            cv_strategy=strategy,
            n_folds=int(merged.get("n_folds", 5)),
            cv_enabled=bool(merged.get("cv_enabled", True)),
            regenerate_on_dataset_change=bool(merged.get("regenerate_on_dataset_change", False)),
        )

    def parse_search(values: dict[str, Any]) -> MLHyperparameterSearchConfig:
        merged = {**search_data, **values}
        strategy = str(merged.get("strategy", "none"))
        if strategy not in {"none", "randomized", "manual_coarse"}:
            raise ValueError(f"Unsupported hyperparameter search strategy: {strategy}")
        return MLHyperparameterSearchConfig(
            strategy=strategy,
            n_iter=int(merged.get("n_iter", 1)),
            seed=int(merged.get("seed", 42)),
            search_space={str(k): list(v) for k, v in merged.get("search_space", {}).items()},
            manual_coarse_configs=[dict(v) for v in merged.get("manual_coarse_configs", [])],
        )

    global_split = parse_split({})
    global_search = parse_search({})
    experiments = {
        str(name): MLExperimentConfig(
            data_split=parse_split((overrides or {}).get("data_split", {})),
            hyperparameter_search=parse_search(
                (overrides or {}).get("hyperparameter_search", {})
            ),
        )
        for name, overrides in data.get("experiments", {}).items()
    }

    return MLModelsConfig(
        device=device,
        model=model,
        training=training,
        physics_loss=physics_loss,
        output=output,
        data_split=global_split,
        hyperparameter_search=global_search,
        experiments=experiments,
    )
