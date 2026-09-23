
# ============================================================
# keras51_augment5_men_women_woman_only.py
#
# CNN + Optuna Hyperparameter Optimization
#
# Features
# 1. ImageDataGenerator
# 2. Train / Validation / Test split
# 3. Functional API CNN
# 4. Optuna 50 Trials
# 5. EarlyStopping
# 6. ReduceLROnPlateau
# 7. Optuna Pruning
# 8. Best model automatic saving
# 9. Best hyperparameters automatic saving
# 10. Final test evaluation
# 11. Classification report
# 12. Optuna SQLite storage
# 13. Trial result CSV
# ============================================================


# ============================================================
# 0. 라이브러리
# ============================================================

import os
import gc
import json
import random
import time
import warnings
import pandas as pd
import numpy as np
import pandas as pd
import tensorflow as tf
import optuna
from optuna.integration import TFKerasPruningCallback
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Model

from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPool2D,
    BatchNormalization,
    GlobalAveragePooling2D,
    Dense,
    Dropout
)

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from optuna.integration import TFKerasPruningCallback

from optuna.trial import TrialState


warnings.filterwarnings("ignore")


# ============================================================
# 1. 기본 설정
# ============================================================

DATA_PATH = './_data/image/man_woman/'

IMG_SIZE = (100, 100)

IMG_HEIGHT = 100
IMG_WIDTH = 100

CHANNELS = 3

RANDOM_STATE = 42

N_TRIALS = 50

MAX_EPOCHS = 100

EARLY_STOPPING_PATIENCE = 30

LR_PATIENCE = 8

LR_FACTOR = 0.5

MIN_LR = 1e-7


# ============================================================
# 2. 저장 폴더
# ============================================================

SAVE_PATH = './_save/man_woman_optuna/'

os.makedirs(SAVE_PATH, exist_ok=True)

BEST_MODEL_PATH = os.path.join(
    SAVE_PATH,
    'keras51_optuna_best.keras'
)

BEST_PARAMS_PATH = os.path.join(
    SAVE_PATH,
    'keras51_optuna_best_params.json'
)

TRIAL_CSV_PATH = os.path.join(
    SAVE_PATH,
    'keras51_optuna_trials.csv'
)

STUDY_DB_PATH = os.path.join(
    SAVE_PATH,
    'keras51_optuna.db'
)


# ============================================================
# 3. Random Seed 고정
# ============================================================

random.seed(RANDOM_STATE)

np.random.seed(RANDOM_STATE)

tf.random.set_seed(RANDOM_STATE)


# ============================================================
# 4. GPU 설정
# ============================================================

print("=" * 70)

print("GPU CHECK")

print("=" * 70)

gpus = tf.config.list_physical_devices('GPU')

print("TensorFlow version :", tf.__version__)

print("GPU count         :", len(gpus))

if gpus:

    print("GPU device        :", gpus[0])

    for gpu in gpus:

        try:

            tf.config.experimental.set_memory_growth(
                gpu,
                True
            )

        except Exception:

            pass

else:

    print("GPU not detected")

print()


# ============================================================
# 5. 데이터 폴더 확인
# ============================================================

print("=" * 70)

print("1. DATA FOLDER CHECK")

print("=" * 70)

print("DATA_PATH :", DATA_PATH)

print("Folders   :", os.listdir(DATA_PATH))

print()


# ============================================================
# 6. 이미지 파일 수 확인
# ============================================================

VALID_EXTENSIONS = (
    '.jpg',
    '.jpeg',
    '.png',
    '.bmp'
)


class_names = sorted(
    [
        folder
        for folder in os.listdir(DATA_PATH)
        if os.path.isdir(
            os.path.join(DATA_PATH, folder)
        )
    ]
)


print("=" * 70)

print("2. CLASS CHECK")

print("=" * 70)

print("Classes :", class_names)

print()


for class_name in class_names:

    class_path = os.path.join(
        DATA_PATH,
        class_name
    )

    files = [

        f

        for f in os.listdir(class_path)

        if f.lower().endswith(
            VALID_EXTENSIONS
        )

    ]

    print(
        f"{class_name:15s} : {len(files):6d} images"
    )

print()


# ============================================================
# 7. 전체 이미지 경로 / label 생성
# ============================================================

file_paths = []

labels = []


for class_index, class_name in enumerate(class_names):

    class_path = os.path.join(
        DATA_PATH,
        class_name
    )

    files = [

        f

        for f in os.listdir(class_path)

        if f.lower().endswith(
            VALID_EXTENSIONS
        )

    ]

    for file_name in files:

        file_paths.append(
            os.path.join(
                class_path,
                file_name
            )
        )

        labels.append(class_index)


file_paths = np.array(file_paths)

labels = np.array(labels)


print("=" * 70)

print("3. TOTAL DATA")

print("=" * 70)

print("Total images :", len(file_paths))

print("Labels       :", np.unique(labels))

print()


# ============================================================
# 8. Train / Validation / Test 분할
# ============================================================

X_train, X_temp, y_train, y_temp = train_test_split(

    file_paths,
    labels,

    test_size=0.30,

    random_state=RANDOM_STATE,

    stratify=labels
)


X_val, X_test, y_val, y_test = train_test_split(

    X_temp,
    y_temp,

    test_size=0.50,

    random_state=RANDOM_STATE,

    stratify=y_temp
)


print("=" * 70)

print("4. DATA SPLIT")

print("=" * 70)

print("Train      :", len(X_train))

print("Validation :", len(X_val))

print("Test       :", len(X_test))

print()


# ============================================================
# 9. ImageDataGenerator
#
# 현재 사용하던 augmentation 개념을 유지
# ============================================================

train_datagen = ImageDataGenerator(

    rescale=1.0 / 255.0,

    horizontal_flip=True,

    width_shift_range=0.10,

    height_shift_range=0.10,

    rotation_range=5,

    zoom_range=0.20,

    shear_range=0.70

)


val_test_datagen = ImageDataGenerator(

    rescale=1.0 / 255.0

)


# ============================================================
# 10. Generator 생성 함수
# ============================================================

def create_generator(

    datagen,
    image_paths,
    image_labels,
    batch_size,
    shuffle

):

    dataframe = pd.DataFrame({

        'filename': image_paths,

        'class': image_labels.astype(str)

    })


    generator = datagen.flow_from_dataframe(

        dataframe,

        x_col='filename',

        y_col='class',

        target_size=IMG_SIZE,

        color_mode='rgb',

        class_mode='binary',

        batch_size=batch_size,

        shuffle=shuffle,

        seed=RANDOM_STATE

    )


    return generator


# ============================================================
# 11. CNN Model Builder
# ============================================================

def build_model(trial):

    # --------------------------------------------------------
    # Optuna Hyperparameters
    # --------------------------------------------------------

    filters1 = trial.suggest_categorical(

        'filters1',

        [16, 32, 64]

    )


    filters2 = trial.suggest_categorical(

        'filters2',

        [32, 64, 128]

    )


    filters3 = trial.suggest_categorical(

        'filters3',

        [64, 128, 256]

    )


    kernel_size = trial.suggest_categorical(

        'kernel_size',

        [3, 5]

    )


    dropout1 = trial.suggest_float(

        'dropout1',

        0.10,

        0.50,

        step=0.05

    )


    dropout2 = trial.suggest_float(

        'dropout2',

        0.10,

        0.50,

        step=0.05

    )


    dense_units = trial.suggest_categorical(

        'dense_units',

        [64, 128, 256]

    )


    final_dropout = trial.suggest_float(

        'final_dropout',

        0.10,

        0.50,

        step=0.05

    )


    learning_rate = trial.suggest_float(

        'learning_rate',

        1e-5,

        3e-3,

        log=True

    )


    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    input1 = Input(

        shape=(

            IMG_HEIGHT,

            IMG_WIDTH,

            CHANNELS

        ),

        name='input1'

    )


    # --------------------------------------------------------
    # Conv Block 1
    # --------------------------------------------------------

    x = Conv2D(

        filters1,

        (kernel_size, kernel_size),

        padding='same',

        activation='relu',

        name='conv1'

    )(input1)


    x = BatchNormalization(

        name='bn1'

    )(x)


    x = MaxPool2D(

        (2, 2),

        name='pool1'

    )(x)


    x = Dropout(

        dropout1,

        name='dropout1'

    )(x)


    # --------------------------------------------------------
    # Conv Block 2
    # --------------------------------------------------------

    x = Conv2D(

        filters2,

        (kernel_size, kernel_size),

        padding='same',

        activation='relu',

        name='conv2'

    )(x)


    x = BatchNormalization(

        name='bn2'

    )(x)


    x = MaxPool2D(

        (2, 2),

        name='pool2'

    )(x)


    x = Dropout(

        dropout2,

        name='dropout2'

    )(x)


    # --------------------------------------------------------
    # Conv Block 3
    # --------------------------------------------------------

    x = Conv2D(

        filters3,

        (kernel_size, kernel_size),

        padding='same',

        activation='relu',

        name='conv3'

    )(x)


    x = BatchNormalization(

        name='bn3'

    )(x)


    x = MaxPool2D(

        (2, 2),

        name='pool3'

    )(x)


    # --------------------------------------------------------
    # Global Average Pooling
    # --------------------------------------------------------

    x = GlobalAveragePooling2D(

        name='global_average_pooling'

    )(x)


    # --------------------------------------------------------
    # Dense
    # --------------------------------------------------------

    x = Dense(

        dense_units,

        activation='relu',

        name='dense1'

    )(x)


    x = Dropout(

        final_dropout,

        name='final_dropout'

    )(x)


    # --------------------------------------------------------
    # Binary Classification
    # --------------------------------------------------------

    output1 = Dense(

        1,

        activation='sigmoid',

        name='output'

    )(x)


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = Model(

        inputs=input1,

        outputs=output1,

        name='keras51_optuna_cnn'

    )


    # --------------------------------------------------------
    # Adam optimizer
    # --------------------------------------------------------

    optimizer = Adam(

        learning_rate=learning_rate

    )


    # --------------------------------------------------------
    # Compile
    # --------------------------------------------------------

    model.compile(

        optimizer=optimizer,

        loss='binary_crossentropy',

        metrics=[

            'accuracy'

        ]

    )


    return model


# ============================================================
# 12. Optuna Objective
# ============================================================

def objective(trial):

    print()

    print("=" * 70)

    print(

        f"OPTUNA TRIAL {trial.number + 1}/{N_TRIALS}"

    )

    print("=" * 70)


    # --------------------------------------------------------
    # TensorFlow graph cleanup
    # --------------------------------------------------------

    tf.keras.backend.clear_session()

    gc.collect()


    # --------------------------------------------------------
    # Batch size
    # --------------------------------------------------------

    batch_size = trial.suggest_categorical(

        'batch_size',

        [16, 32, 64]

    )


    # --------------------------------------------------------
    # Generator
    # --------------------------------------------------------

    train_generator = create_generator(

        train_datagen,

        X_train,

        y_train,

        batch_size,

        shuffle=True

    )


    val_generator = create_generator(

        val_test_datagen,

        X_val,

        y_val,

        batch_size,

        shuffle=False

    )


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = build_model(trial)


    # --------------------------------------------------------
    # Trial temporary model path
    # --------------------------------------------------------

    trial_model_path = os.path.join(

        SAVE_PATH,

        f'trial_{trial.number:03d}.keras'

    )


    # --------------------------------------------------------
    # EarlyStopping
    # --------------------------------------------------------

    early_stopping = EarlyStopping(

        monitor='val_accuracy',

        mode='max',

        patience=EARLY_STOPPING_PATIENCE,

        restore_best_weights=True,

        verbose=1

    )


    # --------------------------------------------------------
    # ReduceLROnPlateau
    # --------------------------------------------------------

    reduce_lr = ReduceLROnPlateau(

        monitor='val_loss',

        factor=LR_FACTOR,

        patience=LR_PATIENCE,

        min_lr=MIN_LR,

        verbose=1

    )


    # --------------------------------------------------------
    # Trial Best Model
    # --------------------------------------------------------

    checkpoint = ModelCheckpoint(

        trial_model_path,

        monitor='val_accuracy',

        mode='max',

        save_best_only=True,

        save_weights_only=False,

        verbose=0

    )


    # --------------------------------------------------------
    # Optuna Pruning
    #
    # validation accuracy가 좋지 않은 Trial을
    # 중간에 자동 종료
    # --------------------------------------------------------

    pruning_callback = TFKerasPruningCallback(

        trial,

        'val_accuracy'

    )


    callbacks = [

        early_stopping,

        reduce_lr,

        checkpoint,

        pruning_callback

    ]


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    start_time = time.time()


    try:

        history = model.fit(

            train_generator,

            validation_data=val_generator,

            epochs=MAX_EPOCHS,

            callbacks=callbacks,

            verbose=1

        )


    except optuna.TrialPruned:

        print()

        print(

            f"Trial {trial.number} PRUNED"

        )

        print()

        raise


    training_time = time.time() - start_time


    # --------------------------------------------------------
    # Best validation accuracy
    # --------------------------------------------------------

    best_val_accuracy = max(

        history.history[

            'val_accuracy'

        ]

    )


    best_epoch = (

        np.argmax(

            history.history[

                'val_accuracy'

            ]

        )

        + 1

    )


    # --------------------------------------------------------
    # Trial attributes
    # --------------------------------------------------------

    trial.set_user_attr(

        'best_epoch',

        int(best_epoch)

    )


    trial.set_user_attr(

        'training_time_seconds',

        float(training_time)

    )


    trial.set_user_attr(

        'model_path',

        trial_model_path

    )


    print()

    print("-" * 70)

    print(

        f"Trial {trial.number} COMPLETE"

    )

    print(

        f"Best val_accuracy : {best_val_accuracy:.6f}"

    )

    print(

        f"Best epoch        : {best_epoch}"

    )

    print(

        f"Training time     : {training_time:.2f} sec"

    )

    print("-" * 70)


    # --------------------------------------------------------
    # Memory cleanup
    # --------------------------------------------------------

    del model

    del train_generator

    del val_generator

    gc.collect()

    tf.keras.backend.clear_session()


    # --------------------------------------------------------
    # Optuna objective
    # --------------------------------------------------------

    return best_val_accuracy


# ============================================================
# 13. Optuna Study 생성
# ============================================================

print()

print("=" * 70)

print("5. CREATE OPTUNA STUDY")

print("=" * 70)


storage_url = (

    'sqlite:///'

    + os.path.abspath(

        STUDY_DB_PATH

    )

)


study = optuna.create_study(

    study_name='keras51_men_women_cnn',

    storage=storage_url,

    load_if_exists=True,

    direction='maximize',

    sampler=optuna.samplers.TPESampler(

        seed=RANDOM_STATE

    ),

    pruner=optuna.pruners.MedianPruner(

        n_startup_trials=5,

        n_warmup_steps=5,

        interval_steps=1

    )

)


print("Study name :", study.study_name)

print("Database   :", STUDY_DB_PATH)

print()


# ============================================================
# 14. Optuna Optimization
# ============================================================

print("=" * 70)

print("6. OPTUNA OPTIMIZATION START")

print("=" * 70)

print()

print(f"Number of trials : {N_TRIALS}")

print(f"Maximum epochs   : {MAX_EPOCHS}")

print()

print(

    "Optuna is searching for the best CNN hyperparameters..."

)

print()


optimization_start = time.time()


try:

    study.optimize(

        objective,

        n_trials=N_TRIALS,

        gc_after_trial=True,

        show_progress_bar=True

    )

except KeyboardInterrupt:

    print()

    print("Optimization manually stopped.")

    print("Completed trials will remain in SQLite database.")


optimization_time = (

    time.time()

    - optimization_start

)


# ============================================================
# 15. Trial Statistics
# ============================================================

print()

print("=" * 70)

print("7. OPTUNA RESULT")

print("=" * 70)


trials = study.trials


completed_trials = [

    trial

    for trial in trials

    if trial.state == TrialState.COMPLETE

]


pruned_trials = [

    trial

    for trial in trials

    if trial.state == TrialState.PRUNED

]


failed_trials = [

    trial

    for trial in trials

    if trial.state == TrialState.FAIL

]


print()

print("Total trials     :", len(trials))

print("Completed trials :", len(completed_trials))

print("Pruned trials    :", len(pruned_trials))

print("Failed trials    :", len(failed_trials))

print(

    "Optimization time:",

    f"{optimization_time:.2f} sec"

)

print()


# ============================================================
# 16. Best Trial
# ============================================================

if not completed_trials:

    raise RuntimeError(

        "No completed Optuna trials were found."

    )


best_trial = study.best_trial


print("=" * 70)

print("8. BEST TRIAL")

print("=" * 70)

print()

print("Best Trial Number :", best_trial.number)

print(

    "Best Validation Accuracy :",

    f"{best_trial.value:.6f}"

)

print()

print("Best Hyperparameters")

print("-" * 70)


for key, value in best_trial.params.items():

    print(

        f"{key:20s} : {value}"

    )


print()


# ============================================================
# 17. Best Hyperparameters JSON 저장
# ============================================================

best_params = {

    'trial_number': int(best_trial.number),

    'best_val_accuracy': float(

        best_trial.value

    ),

    'params': best_trial.params

}


with open(

    BEST_PARAMS_PATH,

    'w',

    encoding='utf-8'

) as f:

    json.dump(

        best_params,

        f,

        indent=4,

        ensure_ascii=False

    )


print(

    "Best parameters saved:",

    BEST_PARAMS_PATH

)


# ============================================================
# 18. Trial 결과 CSV 저장
# ============================================================

trials_dataframe = study.trials_dataframe(

    attrs=(

        'number',

        'value',

        'datetime_start',

        'datetime_complete',

        'duration',

        'state',

        'params',

        'user_attrs'

    )

)


trials_dataframe.to_csv(

    TRIAL_CSV_PATH,

    index=False,

    encoding='utf-8-sig'

)


print(

    "Trial results saved:",

    TRIAL_CSV_PATH

)


# ============================================================
# 19. Best Trial 모델 찾기
# ============================================================

best_trial_model_path = best_trial.user_attrs.get(

    'model_path',

    None

)


print()

print("=" * 70)

print("9. BEST TRIAL MODEL")

print("=" * 70)

print()

print(

    "Best trial model:",

    best_trial_model_path

)


# ============================================================
# 20. Best Trial 모델을 최종 모델명으로 복사
# ============================================================

if (

    best_trial_model_path

    and

    os.path.exists(

        best_trial_model_path

    )

):

    best_model = tf.keras.models.load_model(

        best_trial_model_path

    )


    best_model.save(

        BEST_MODEL_PATH

    )


    print()

    print(

        "BEST MODEL SAVED"

    )

    print(

        BEST_MODEL_PATH

    )


else:

    print()

    print(

        "Best trial checkpoint was not found."

    )

    print(

        "The best model will be retrained below."

    )

    best_model = None


# ============================================================
# 21. Best Model이 없을 경우 재학습
# ============================================================

if best_model is None:

    print()

    print("=" * 70)

    print("10. RETRAIN BEST MODEL")

    print("=" * 70)


    tf.keras.backend.clear_session()

    gc.collect()


    # --------------------------------------------------------
    # Best parameter를 이용하여 최종 모델 생성
    # --------------------------------------------------------

    class FixedTrial:

        def __init__(self, params):

            self.params = params


        def suggest_categorical(

            self,

            name,

            choices

        ):

            return self.params[name]


        def suggest_float(

            self,

            name,

            low,

            high,

            step=None,

            log=False

        ):

            return self.params[name]


        def suggest_int(

            self,

            name,

            low,

            high,

            step=1

        ):

            return self.params[name]


    fixed_trial = FixedTrial(

        best_trial.params

    )


    best_model = build_model(

        fixed_trial

    )


    best_batch_size = best_trial.params[

        'batch_size'

    ]


    final_train_generator = create_generator(

        train_datagen,

        X_train,

        y_train,

        best_batch_size,

        shuffle=True

    )


    final_val_generator = create_generator(

        val_test_datagen,

        X_val,

        y_val,

        best_batch_size,

        shuffle=False

    )


    final_callbacks = [

        EarlyStopping(

            monitor='val_accuracy',

            mode='max',

            patience=EARLY_STOPPING_PATIENCE,

            restore_best_weights=True,

            verbose=1

        ),

        ReduceLROnPlateau(

            monitor='val_loss',

            factor=LR_FACTOR,

            patience=LR_PATIENCE,

            min_lr=MIN_LR,

            verbose=1

        ),

        ModelCheckpoint(

            BEST_MODEL_PATH,

            monitor='val_accuracy',

            mode='max',

            save_best_only=True,

            verbose=1

        )

    ]


    best_model.fit(

        final_train_generator,

        validation_data=final_val_generator,

        epochs=MAX_EPOCHS,

        callbacks=final_callbacks,

        verbose=1

    )


    best_model = tf.keras.models.load_model(

        BEST_MODEL_PATH

    )


# ============================================================
# 22. Test Generator
# ============================================================

print()

print("=" * 70)

print("11. FINAL TEST")

print("=" * 70)


test_batch_size = best_trial.params[

    'batch_size'

]


test_generator = create_generator(

    val_test_datagen,

    X_test,

    y_test,

    test_batch_size,

    shuffle=False

)


# ============================================================
# 23. Model Summary
# ============================================================

print()

print("=" * 70)

print("12. BEST MODEL SUMMARY")

print("=" * 70)

print()

best_model.summary()


# ============================================================
# 24. Test Evaluation
# ============================================================

print()

print("=" * 70)

print("13. MODEL.EVALUATE")

print("=" * 70)


test_loss, test_accuracy = best_model.evaluate(

    test_generator,

    verbose=1

)


print()

print(

    f"Test Loss     : {test_loss:.6f}"

)

print(

    f"Test Accuracy : {test_accuracy:.6f}"

)


# ============================================================
# 25. Prediction
# ============================================================

print()

print("=" * 70)

print("14. PREDICTION")

print("=" * 70)


test_generator.reset()


y_probability = best_model.predict(

    test_generator,

    verbose=1

)


y_probability = y_probability.reshape(-1)


y_pred = (

    y_probability >= 0.5

).astype(int)


y_true = y_test


# ============================================================
# 26. Accuracy
# ============================================================

prediction_accuracy = accuracy_score(

    y_true,

    y_pred

)


print()

print(

    "Prediction Accuracy :",

    f"{prediction_accuracy:.6f}"

)


# ============================================================
# 27. Classification Report
# ============================================================

print()

print("=" * 70)

print("15. CLASSIFICATION REPORT")

print("=" * 70)

print()


print(

    classification_report(

        y_true,

        y_pred,

        target_names=class_names,

        digits=4

    )

)


# ============================================================
# 28. Confusion Matrix
# ============================================================

print()

print("=" * 70)

print("16. CONFUSION MATRIX")

print("=" * 70)

print()


cm = confusion_matrix(

    y_true,

    y_pred

)


print(cm)


# ============================================================
# 29. 최종 결과 저장
# ============================================================

final_result = {

    'best_trial': int(

        best_trial.number

    ),

    'best_val_accuracy': float(

        best_trial.value

    ),

    'test_loss': float(

        test_loss

    ),

    'test_accuracy': float(

        test_accuracy

    ),

    'prediction_accuracy': float(

        prediction_accuracy

    ),

    'total_trials': int(

        len(trials)

    ),

    'completed_trials': int(

        len(completed_trials)

    ),

    'pruned_trials': int(

        len(pruned_trials)

    ),

    'failed_trials': int(

        len(failed_trials)

    ),

    'best_params': best_trial.params

}


FINAL_RESULT_PATH = os.path.join(

    SAVE_PATH,

    'keras51_optuna_final_result.json'

)


with open(

    FINAL_RESULT_PATH,

    'w',

    encoding='utf-8'

) as f:

    json.dump(

        final_result,

        f,

        indent=4,

        ensure_ascii=False

    )


# ============================================================
# 30. 최종 완료
# ============================================================

print()

print("=" * 70)

print("OPTUNA CNN OPTIMIZATION COMPLETE")

print("=" * 70)

print()

print(

    f"Best Trial          : {best_trial.number}"

)

print(

    f"Best Val Accuracy   : {best_trial.value:.6f}"

)

print(

    f"Final Test Accuracy : {test_accuracy:.6f}"

)

print()

print(

    f"Completed Trials    : {len(completed_trials)}"

)

print(

    f"Pruned Trials       : {len(pruned_trials)}"

)

print(

    f"Failed Trials       : {len(failed_trials)}"

)

print()

print("Files")

print("-" * 70)

print(

    "Best Model    :",

    BEST_MODEL_PATH

)

print(

    "Best Params   :",

    BEST_PARAMS_PATH

)

print(

    "Trial CSV     :",

    TRIAL_CSV_PATH

)

print(

    "Optuna DB     :",

    STUDY_DB_PATH

)

print(

    "Final Result  :",

    FINAL_RESULT_PATH

)

print()

print("=" * 70)

print("ALL PROCESS FINISHED")

print("=" * 70)

