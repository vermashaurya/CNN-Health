import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB0, NASNetLarge, ResNet152V2
from tensorflow.keras.preprocessing.image import ImageDataGenerator


CLASS_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}

BACKBONES = {
    "efficientnetb0": {
        "builder": EfficientNetB0,
        "image_size": (256, 256),
        "weights": "imagenet",
    },
    "resnet152v2": {
        "builder": ResNet152V2,
        "image_size": (224, 224),
        "weights": "imagenet",
    },
    "nasnetlarge": {
        "builder": NASNetLarge,
        "image_size": (331, 331),
        "weights": "imagenet",
    },
}


@dataclass
class ExperimentConfig:
    dataset_dir: Path
    csv_path: Path
    output_dir: Path
    backbone: str = "efficientnetb0"
    batch_size: int = 16
    epochs: int = 30
    learning_rate: float = 1e-4
    test_size: float = 0.1
    validation_size: float = 0.1
    random_state: int = 42
    freeze_backbone: bool = True
    smoke_test: bool = False
    workers: int = 1

    @property
    def image_size(self) -> tuple[int, int]:
        return BACKBONES[self.backbone]["image_size"]


def detect_strategy() -> tf.distribute.Strategy:
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        print("Using TPU strategy.")
        return tf.distribute.TPUStrategy(resolver)
    except Exception:
        print("Using default CPU/GPU strategy.")
        return tf.distribute.get_strategy()


def resolve_default_paths(repo_root: Path) -> tuple[Path, Path]:
    data_root = repo_root / "data" / "aptos2019-blindness-detection"
    images_dir = data_root / "train_images"
    csv_path = data_root / "train.csv"
    return images_dir, csv_path


def load_labels(csv_path: Path, dataset_dir: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found at {csv_path}. Place the APTOS train.csv file there or pass --csv-path."
        )
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Image directory not found at {dataset_dir}. Place the train_images folder there or pass --dataset-dir."
        )

    df = pd.read_csv(csv_path)
    required_columns = {"id_code", "diagnosis"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"CSV file is missing required columns: {sorted(missing)}")

    df["image_path"] = df["id_code"].apply(lambda image_id: dataset_dir / f"{image_id}.png")
    df = df[df["image_path"].apply(Path.exists)].copy()
    if df.empty:
        raise ValueError("No matching retinal images were found for the provided CSV file.")

    df["diagnosis"] = df["diagnosis"].astype(int)
    return df


def stratified_split(df: pd.DataFrame, config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=df["diagnosis"],
    )
    adjusted_val_size = config.validation_size / (1.0 - config.test_size)
    train_df, val_df = train_test_split(
        train_df,
        test_size=adjusted_val_size,
        random_state=config.random_state,
        stratify=train_df["diagnosis"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def make_generators(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ExperimentConfig,
):
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    common_args = {
        "x_col": "image_path",
        "y_col": "diagnosis",
        "target_size": config.image_size,
        "batch_size": config.batch_size,
        "class_mode": "sparse",
    }

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df.assign(image_path=train_df["image_path"].astype(str), diagnosis=train_df["diagnosis"].astype(str)),
        shuffle=True,
        **common_args,
    )
    val_generator = eval_datagen.flow_from_dataframe(
        dataframe=val_df.assign(image_path=val_df["image_path"].astype(str), diagnosis=val_df["diagnosis"].astype(str)),
        shuffle=False,
        **common_args,
    )
    test_generator = eval_datagen.flow_from_dataframe(
        dataframe=test_df.assign(image_path=test_df["image_path"].astype(str), diagnosis=test_df["diagnosis"].astype(str)),
        shuffle=False,
        **common_args,
    )
    return train_generator, val_generator, test_generator


def build_model(config: ExperimentConfig, strategy: tf.distribute.Strategy) -> tf.keras.Model:
    backbone_meta = BACKBONES[config.backbone]
    with strategy.scope():
        base_model = backbone_meta["builder"](
            include_top=False,
            weights=backbone_meta["weights"],
            input_shape=(*config.image_size, 3),
        )
        base_model.trainable = not config.freeze_backbone

        model = models.Sequential(
            [
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.4),
                layers.Dense(256, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(len(CLASS_NAMES), activation="softmax"),
            ]
        )
        model.compile(
            optimizer=optimizers.Adam(learning_rate=config.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
    return model


def training_callbacks(output_dir: Path) -> list[callbacks.Callback]:
    return [
        callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_model.keras"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        callbacks.CSVLogger(str(output_dir / "training_log.csv")),
    ]


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path) -> dict:
    report = classification_report(
        y_true,
        y_pred,
        labels=list(CLASS_NAMES.keys()),
        target_names=list(CLASS_NAMES.values()),
        zero_division=0,
        output_dict=True,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(CLASS_NAMES.keys()))
    accuracy = accuracy_score(y_true, y_pred)

    metrics = {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }
    (output_dir / "evaluation.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def train_cnn(config: ExperimentConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_labels(config.csv_path, config.dataset_dir)
    train_df, val_df, test_df = stratified_split(df, config)
    train_generator, val_generator, test_generator = make_generators(train_df, val_df, test_df, config)
    strategy = detect_strategy()
    model = build_model(config, strategy)

    epochs = 2 if config.smoke_test else config.epochs
    print(f"Training {config.backbone} for {epochs} epoch(s).")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=training_callbacks(config.output_dir),
        verbose=1,
    )

    model.save(config.output_dir / "final_model.keras")
    predictions = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_df["diagnosis"].to_numpy()
    metrics = evaluate_predictions(y_true, y_pred, config.output_dir)

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(config.output_dir / "history.csv", index=False)

    split_summary = {
        "train_samples": int(len(train_df)),
        "validation_samples": int(len(val_df)),
        "test_samples": int(len(test_df)),
        "backbone": config.backbone,
        "epochs_requested": int(config.epochs),
        "epochs_ran": int(epochs),
        "smoke_test": bool(config.smoke_test),
        "accuracy": float(metrics["accuracy"]),
    }
    (config.output_dir / "run_summary.json").write_text(json.dumps(split_summary, indent=2))
    print(json.dumps(split_summary, indent=2))


def extract_handcrafted_features(image_paths: Iterable[Path], image_size: tuple[int, int] = (128, 128)) -> np.ndarray:
    features = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            image = image.convert("RGB").resize(image_size)
            image_array = np.asarray(image, dtype=np.float32) / 255.0

        channel_means = image_array.mean(axis=(0, 1))
        channel_stds = image_array.std(axis=(0, 1))
        channel_mins = image_array.min(axis=(0, 1))
        channel_maxs = image_array.max(axis=(0, 1))
        flattened = np.concatenate([channel_means, channel_stds, channel_mins, channel_maxs])
        features.append(flattened)

    return np.asarray(features)


def run_ml_baseline(config: ExperimentConfig) -> None:
    output_dir = config.output_dir / "baseline"
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_labels(config.csv_path, config.dataset_dir)
    train_df, _, test_df = stratified_split(df, config)

    x_train = extract_handcrafted_features(train_df["image_path"])
    x_test = extract_handcrafted_features(test_df["image_path"])
    y_train = train_df["diagnosis"].to_numpy()
    y_test = test_df["diagnosis"].to_numpy()

    baseline = RandomForestClassifier(
        n_estimators=300,
        random_state=config.random_state,
        class_weight="balanced_subsample",
    )
    baseline.fit(x_train, y_train)
    predictions = baseline.predict(x_test)
    metrics = evaluate_predictions(y_test, predictions, output_dir)
    print(json.dumps({"baseline_accuracy": metrics["accuracy"]}, indent=2))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    default_dataset_dir, default_csv_path = resolve_default_paths(repo_root)

    parser = argparse.ArgumentParser(
        description="Blindness detection on diabetic retinopathy fundus images using CNN backbones and an ML baseline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_shared_arguments(target: argparse.ArgumentParser) -> None:
        target.add_argument("--dataset-dir", type=Path, default=default_dataset_dir)
        target.add_argument("--csv-path", type=Path, default=default_csv_path)
        target.add_argument("--output-dir", type=Path, default=repo_root / "artifacts")
        target.add_argument(
            "--backbone",
            choices=sorted(BACKBONES.keys()),
            default="efficientnetb0",
            help="CNN backbone used for transfer learning.",
        )
        target.add_argument("--batch-size", type=int, default=16)
        target.add_argument("--epochs", type=int, default=30)
        target.add_argument("--learning-rate", type=float, default=1e-4)
        target.add_argument("--test-size", type=float, default=0.1)
        target.add_argument("--validation-size", type=float, default=0.1)
        target.add_argument("--random-state", type=int, default=42)
        target.add_argument("--freeze-backbone", action="store_true")
        target.add_argument("--smoke-test", action="store_true")

    train_parser = subparsers.add_parser("train", help="Train a CNN model.")
    add_shared_arguments(train_parser)

    baseline_parser = subparsers.add_parser("baseline", help="Run a classical ML baseline.")
    add_shared_arguments(baseline_parser)

    return parser.parse_args(argv)


def namespace_to_config(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        dataset_dir=args.dataset_dir,
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        backbone=args.backbone,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        test_size=args.test_size,
        validation_size=args.validation_size,
        random_state=args.random_state,
        freeze_backbone=args.freeze_backbone,
        smoke_test=args.smoke_test,
    )


def run_cli(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = namespace_to_config(args)

    if args.command == "train":
        train_cnn(config)
        return 0
    if args.command == "baseline":
        run_ml_baseline(config)
        return 0
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(run_cli())
