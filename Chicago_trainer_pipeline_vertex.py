from typing import List
from absl import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow_transform.tf_metadata import schema_utils

from tfx import v1 as tfx
from tfx_bsl.public import tfxio

from tensorflow_metadata.proto.v0 import schema_pb2


_FEATURE_KEYS = ['standardized_trip_miles',
 'standardized_trip_seconds',
 'trip_start_month',
 'trip_start_hour',
 'company_Flash_Cab',
 'company_Blue_Ribbon_Taxi_Association',
 'company_Taxicab_Insurance_Agency_Llc',
 'company_Taxi_Affiliation_Services',
 'company_Globe_Taxi',
 'company_Chicago_Independents',
 'company_City_Service',
 'company_5_Star_Taxi',
 'company_Patriot_Taxi_Dba_Peace_Taxi_Associat',
 'company_Sun_Taxi',
 'company_Medallion_Leasin',
 'company_Choice_Taxi_Association',
 'company_Star_North_Taxi_Management_Llc',
 'company_Taxicab_Insurance_Agency__LLC',
 'company_KOAM_Taxi_Association',
 'company_Choice_Taxi_Association_Inc',
 'company_Chicago_City_Taxi_Association',
 'company_U_Taxicab',
 'company_Top_Cab',
 'company_Taxi_Affiliation_Services_Llc___Yell',
 'company_Chicago_Taxicab',
 'company_3556___36214_RC_Andrews_Cab',
 'company_Top_Cab_Affiliation',
 'company_312_Medallion_Management_Corp',
 'company_5167___71969_5167_Taxi_Inc',
 'company_Metro_Jet_Taxi_A_',
 'company_3591___63480_Chuks_Cab',
 'company_Leonard_Cab_Co',
 'company_2733___74600_Benny_Jona',
 'company_Setare_Inc',
 'company_4053___40193_Adwar_H__Nikola',
 'company_6574___Babylon_Express_Inc_',
 'company_Petani_Cab_Corp',
 'company_4623___27290_Jay_Kim',
 'company_4787___56058_Reny_Cab_Co',
 'company_5062___34841_Sam_Mestas',
 'pickup_community_area_10',
 'pickup_community_area_4',
 'pickup_community_area_59',
 'pickup_community_area_39',
 'pickup_community_area_13',
 'pickup_community_area_29',
 'pickup_community_area_52',
 'pickup_community_area_38',
 'pickup_community_area_32',
 'pickup_community_area_14',
 'pickup_community_area_31',
 'pickup_community_area_8',
 'pickup_community_area_5',
 'pickup_community_area_49',
 'pickup_community_area_28',
 'pickup_community_area_6',
 'pickup_community_area_43',
 'pickup_community_area_61',
 'pickup_community_area_36',
 'pickup_community_area_69',
 'pickup_community_area_34',
 'pickup_community_area_15',
 'pickup_community_area_22',
 'pickup_community_area_75',
 'pickup_community_area_16',
 'pickup_community_area_44',
 'pickup_community_area_68',
 'pickup_community_area_11',
 'pickup_community_area_55',
 'pickup_community_area_42',
 'pickup_community_area_3',
 'pickup_community_area_37',
 'pickup_community_area_54',
 'pickup_community_area_53',
 'pickup_community_area_18',
 'pickup_community_area_48',
 'pickup_community_area_40',
 'pickup_community_area_41',
 'pickup_community_area_46',
 'pickup_community_area_12',
 'pickup_community_area_17',
 'pickup_community_area_71',
 'pickup_community_area_7',
 'pickup_community_area_21',
 'pickup_community_area_66',
 'pickup_community_area_67',
 'pickup_community_area_25',
 'pickup_community_area_73',
 'pickup_community_area_19',
 'pickup_community_area_58',
 'pickup_community_area_45',
 'pickup_community_area_57',
 'pickup_community_area_9',
 'pickup_community_area_60',
 'pickup_community_area_47',
 'pickup_community_area_24',
 'pickup_community_area_27',
 'pickup_community_area_51',
 'pickup_community_area_30',
 'pickup_community_area_70',
 'pickup_community_area_77',
 'pickup_community_area_63',
 'pickup_community_area_74',
 'pickup_community_area_65',
 'pickup_community_area_50',
 'pickup_community_area_23',
 'pickup_community_area_72',
 'pickup_community_area_35',
 'pickup_community_area_26',
 'pickup_community_area_20',
 'pickup_community_area_62',
 'pickup_community_area_2',
 'pickup_community_area_64',
 'pickup_community_area_1',
 'pickup_community_area_33',
 'pickup_community_area_76',
 'pickup_community_area_56',
 'dropoff_community_area_10',
 'dropoff_community_area_8',
 'dropoff_community_area_59',
 'dropoff_community_area_32',
 'dropoff_community_area_21',
 'dropoff_community_area_43',
 'dropoff_community_area_75',
 'dropoff_community_area_28',
 'dropoff_community_area_16',
 'dropoff_community_area_45',
 'dropoff_community_area_4',
 'dropoff_community_area_76',
 'dropoff_community_area_33',
 'dropoff_community_area_24',
 'dropoff_community_area_65',
 'dropoff_community_area_34',
 'dropoff_community_area_15',
 'dropoff_community_area_39',
 'dropoff_community_area_6',
 'dropoff_community_area_77',
 'dropoff_community_area_31',
 'dropoff_community_area_69',
 'dropoff_community_area_56',
 'dropoff_community_area_7',
 'dropoff_community_area_44',
 'dropoff_community_area_51',
 'dropoff_community_area_50',
 'dropoff_community_area_3',
 'dropoff_community_area_49',
 'dropoff_community_area_72',
 'dropoff_community_area_2',
 'dropoff_community_area_38',
 'dropoff_community_area_54',
 'dropoff_community_area_48',
 'dropoff_community_area_40',
 'dropoff_community_area_71',
 'dropoff_community_area_46',
 'dropoff_community_area_11',
 'dropoff_community_area_1',
 'dropoff_community_area_36',
 'dropoff_community_area_14',
 'dropoff_community_area_70',
 'dropoff_community_area_52',
 'dropoff_community_area_22',
 'dropoff_community_area_60',
 'dropoff_community_area_61',
 'dropoff_community_area_13',
 'dropoff_community_area_73',
 'dropoff_community_area_17',
 'dropoff_community_area_41',
 'dropoff_community_area_58',
 'dropoff_community_area_57',
 'dropoff_community_area_23',
 'dropoff_community_area_53',
 'dropoff_community_area_30',
 'dropoff_community_area_29',
 'dropoff_community_area_35',
 'dropoff_community_area_67',
 'dropoff_community_area_12',
 'dropoff_community_area_18',
 'dropoff_community_area_37',
 'dropoff_community_area_27',
 'dropoff_community_area_68',
 'dropoff_community_area_19',
 'dropoff_community_area_62',
 'dropoff_community_area_42',
 'dropoff_community_area_5',
 'dropoff_community_area_25',
 'dropoff_community_area_63',
 'dropoff_community_area_74',
 'dropoff_community_area_66',
 'dropoff_community_area_20',
 'dropoff_community_area_47',
 'dropoff_community_area_55',
 'dropoff_community_area_26',
 'dropoff_community_area_64',
 'dropoff_community_area_9']
_LABEL_KEY = 'fare'

_TRAIN_BATCH_SIZE = 16
_EVAL_BATCH_SIZE = 16

_FEATURE_SPEC = {
    **{
        # The first three features in _FEATURE_KEYS are tf.float32
        feature: tf.io.FixedLenFeature(shape=[1], dtype=tf.float32)
        for feature in _FEATURE_KEYS[:2]
    },
    **{
        # The remaining features in _FEATURE_KEYS are tf.int64
        feature: tf.io.FixedLenFeature(shape=[1], dtype=tf.int64)
        for feature in _FEATURE_KEYS[2:]
    },
    _LABEL_KEY: tf.io.FixedLenFeature(shape=[1], dtype=tf.float32)
}


def _input_fn(file_pattern: List[str],
              data_accessor: tfx.components.DataAccessor,
              schema: schema_pb2.Schema,
              batch_size: int) -> tf.data.Dataset:

    return data_accessor.tf_dataset_factory(
      file_pattern,
      tfxio.TensorFlowDatasetOptions(
          batch_size=batch_size, label_key=_LABEL_KEY),
      schema=schema).repeat()

def _get_distribution_strategy(fn_args: tfx.components.FnArgs):
    if fn_args.custom_config.get('use_gpu', False):
        logging.info('Using MirroredStrategy with one GPU.')
        return tf.distribute.MirroredStrategy(devices=['device:GPU:0'])
    return None


def _make_keras_model() -> tf.keras.Model:
    
    inputs = [keras.layers.Input(shape=(1,), name=f) for f in _FEATURE_KEYS]
    d = keras.layers.concatenate(inputs)
    for i in range(2):
        if i == 0:
            d = keras.layers.Dense(64, activation='relu')(d)
        if i == 1:
            d = keras.layers.Dense(32, activation='relu')(d)
    outputs = keras.layers.Dense(1)(d)

    model = keras.Model(inputs=inputs, outputs=outputs)

    # model = keras.models.Sequential([
    #     keras.layers.Dense(64, activation='relu', input_shape=(len(_FEATURE_KEYS),), kernel_regularizer=tf.keras.regularizers.l2(0.005)),
    #     keras.layers.Dropout(0.4),
    #     keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.005)),
    #     keras.layers.Dense(1)
    # ])
    learning_rate = 2.5e-4
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    model.summary(print_fn=logging.info)
    return model


# TFX Trainer will call this function.
def run_fn(fn_args: tfx.components.FnArgs):
    schema = schema_utils.schema_from_feature_spec(_FEATURE_SPEC)

    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        schema,
        batch_size=_TRAIN_BATCH_SIZE)
    eval_dataset = _input_fn(
        fn_args.eval_files,
        fn_args.data_accessor,
        schema,
        batch_size=_EVAL_BATCH_SIZE)

    strategy = _get_distribution_strategy(fn_args)
    if strategy is None:
        model = _make_keras_model()
    else:
        with strategy.scope():
            model = _make_keras_model()
    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps)

    # The result of the training should be saved in `fn_args.serving_model_dir`
    # directory.
    model.save(fn_args.serving_model_dir, save_format='tf')