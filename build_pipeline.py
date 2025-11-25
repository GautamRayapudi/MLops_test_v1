import sagemaker
from sagemaker.session import Session
from sagemaker.workflow.parameters import ParameterString, ParameterBoolean
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.model_step import RegisterModel
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThan

from sagemaker.processing import (
    ScriptProcessor,
    ProcessingInput,
    ProcessingOutput,
)
from sagemaker.estimator import Estimator


def build_pipeline(
    region,
    role,
    default_bucket,
    gh_repo_url,
    gh_branch,
    gh_username,
    gh_token,
    ecr_processing_image,
    ecr_training_image,
):

    session = Session()

    # -----------------------------
    # Pipeline Parameters
    # -----------------------------
    input_data = ParameterString(
        name="InputData",
        default_value=f"s3://{default_bucket}/training-data/"
    )

    use_preprocess = ParameterBoolean(
        name="UsePreprocess",
        default_value=True
    )

    # -----------------------------
    # (Optional) Preprocessing Step
    # -----------------------------
    preprocess_step = None

    preprocess_processor = ScriptProcessor(
        role=role,
        image_uri=ecr_processing_image,
        instance_type="ml.m5.large",
        instance_count=1,
        command=["python3"],
        git_config={
            "repo": gh_repo_url,
            "branch": gh_branch,
            "username": gh_username,
            "password": gh_token,
        },
    )

    preprocess_step = ProcessingStep(
        name="PreprocessData",
        processor=preprocess_processor,
        inputs=[
            ProcessingInput(
                source=input_data,
                destination="/opt/ml/processing/input"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="processed_data",
                source="/opt/ml/processing/output"
            )
        ],
        code="src/preprocess.py",
    )

    # -----------------------------
    # Training Step
    # -----------------------------

    # The trick: choose input based on preprocessing flag
    training_input = sagemaker.inputs.TrainingInput(
        s3_data=preprocess_step.properties.ProcessingOutputConfig.Outputs["processed_data"]
        .S3Output.S3Uri
    ).bind(use_preprocess)

    raw_input = sagemaker.inputs.TrainingInput(
        s3_data=input_data
    ).bind(~use_preprocess)

    estimator = Estimator(
        role=role,
        image_uri=ecr_training_image,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        entry_point="src/train.py",
        git_config={
            "repo": gh_repo_url,
            "branch": gh_branch,
            "username": gh_username,
            "password": gh_token,
        },
        sagemaker_session=session,
    )

    train_step = TrainingStep(
        name="TrainModel",
        estimator=estimator,
        inputs={
            "train": training_input  # preprocessing OR raw data
        },
    )

    # -----------------------------
    # Evaluation Step
    # -----------------------------
    eval_processor = ScriptProcessor(
        role=role,
        image_uri=ecr_processing_image,
        instance_type="ml.m5.large",
        instance_count=1,
        command=["python3"],
        git_config={
            "repo": gh_repo_url,
            "branch": gh_branch,
            "username": gh_username,
            "password": gh_token,
        },
    )

    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    eval_step = ProcessingStep(
        name="EvaluateModel",
        processor=eval_processor,
        code="src/evaluate.py",
        inputs=[
            ProcessingInput(
                source=train_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
            )
        ],
        property_files=[evaluation_report],
    )

    # -----------------------------
    # Conditional model registration
    # -----------------------------
    cond_step = ConditionStep(
        name="CheckModelQuality",
        conditions=[
            ConditionGreaterThan(
                left=evaluation_report.prop("accuracy"),
                right=0.80
            )
        ],
        if_steps=[
            RegisterModel(
                name="RegisterTrainedModel",
                model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
                estimator=estimator,
                inference_instances=["ml.m5.large"],
                transform_instances=["ml.m5.large"],
            )
        ],
        else_steps=[],
    )

    # -----------------------------
    # Assemble Pipeline
    # -----------------------------
    pipeline_steps = []

    # preprocess only if enabled
    pipeline_steps.append(preprocess_step)

    pipeline_steps.extend([
        train_step,
        eval_step,
        cond_step
    ])

    pipeline = Pipeline(
        name="GitHubS3FlexiblePipeline",
        parameters=[input_data, use_preprocess],
        steps=pipeline_steps,
        sagemaker_session=session,
    )

    return pipeline
