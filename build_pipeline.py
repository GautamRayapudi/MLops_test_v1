import sagemaker
from sagemaker.session import Session

from sagemaker.workflow.parameters import (
    ParameterString,
    ParameterBoolean,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.model_step import RegisterModel
from sagemaker.workflow.properties import PropertyFile

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

    use_evaluation = ParameterBoolean(
        name="UseEvaluation",
        default_value=True
    )

    # -----------------------------
    # 1. Optional Preprocessing Step
    # -----------------------------
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
                destination="/opt/ml/processing/input",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="processed_data",
                source="/opt/ml/processing/output",
            )
        ],
        code="src/preprocess.py",
    )

    # -----------------------------
    # 2. Training Step
    # -----------------------------
    # Choose training input based on preprocessing flag
    processed_input = sagemaker.inputs.TrainingInput(
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
            "train": processed_input,
        },
    )

    # -----------------------------
    # 3. Optional Evaluation Step
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
    # 4. Model Registration
    # -----------------------------
    # If evaluation is skipped: register the model unconditionally.
    # If evaluation runs: use the evaluation result.

    # Register regardless of evaluation
    register_step = RegisterModel(
        name="RegisterModel",
        model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
        estimator=estimator,
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
    )

    # -----------------------------
    # Pipeline Assembly
    # -----------------------------
    steps = []

    # include preprocess only if enabled
    steps.append(preprocess_step.bind(use_preprocess))

    # training always included
    steps.append(train_step)

    # include evaluation only if enabled
    steps.append(eval_step.bind(use_evaluation))

    # register step always included
    steps.append(register_step)

    pipeline = Pipeline(
        name="GitHubS3OptionalPipeline",
        parameters=[input_data, use_preprocess, use_evaluation],
        steps=steps,
        sagemaker_session=session,
    )

    return pipeline
