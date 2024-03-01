#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf
from tfx.components import CsvExampleGen, StatisticsGen, SchemaGen, ExampleValidator, Transform, Trainer, Tuner
from tfx.proto import example_gen_pb2
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
import os
import pandas as pd


# In[5]:


dataset = pd.read_csv("./Data/SPAM_Text.csv")


# In[7]:


dataset


# In[9]:


dataset['Category'].unique()


# In[10]:


dataset['Category'] = dataset['Category'].replace({'ham':1, 'spam':0})


# In[12]:


dataset['Category'].unique()


# In[40]:


dataset = dataset[['Message', 'Category']]


# In[41]:


dataset


# In[44]:


dataset.to_csv(os.path.join('Data/Spam_Data.csv'), index=False)


# # Set Variabel

# In[4]:


PIPELINE_NAME = "spam-pipeline"
SCHEMA_PIPELINE_NAME = "spam-tfdv-schema"

#Directory untuk menyimpan artifact yang akan dihasilkan
PIPELINE_ROOT = os.path.join('pipelines', PIPELINE_NAME)

# Path to a SQLite DB file to use as an MLMD storage.
METADATA_PATH = os.path.join('metadata', PIPELINE_NAME, 'metadata.db')

# Output directory where created models from the pipeline will be exported.
SERVING_MODEL_DIR = os.path.join('serving_model', PIPELINE_NAME)

# from absl import logging
# logging.set_verbosity(logging.INFO)


# In[5]:


PIPELINE_ROOT, METADATA_PATH,SERVING_MODEL_DIR


# In[6]:


DATA_ROOT = 'Data'


# In[7]:


interactive_context = InteractiveContext(pipeline_root=PIPELINE_ROOT)


# In[8]:


interactive_context


# # Data Ingestion

# In[9]:


output = example_gen_pb2.Output(
    split_config = example_gen_pb2.SplitConfig(splits=[
        example_gen_pb2.SplitConfig.Split(name='train',hash_buckets = 8),
        example_gen_pb2.SplitConfig.Split(name='eval',hash_buckets=2)
    ]))

example_gen = CsvExampleGen(input_base=DATA_ROOT, output_config=output)


# In[10]:


output, type(output), type(example_gen)


# In[11]:


interactive_context.run(example_gen)


# # Data Validation

# ### Summary Statistics

# In[12]:


statistics_gen = StatisticsGen(
    examples = example_gen.outputs['examples'])

interactive_context.run(statistics_gen)


# In[13]:


interactive_context.show(statistics_gen.outputs['statistics'])


# ### Data Scheme

# In[14]:


schema_gen = SchemaGen(statistics = statistics_gen.outputs["statistics"])
interactive_context.run(schema_gen)


# In[15]:


interactive_context.show(schema_gen.outputs['schema'])


# ### Mengidentifikasi Anomali pada Dataset

# In[16]:


example_validator = ExampleValidator(
    statistics = statistics_gen.outputs['statistics'],
    schema = schema_gen.outputs['schema']
)

interactive_context.run(example_validator)


# In[17]:


interactive_context.show(example_validator.outputs['anomalies'])


# # Data Preprocessing

# In[18]:


TRANSFORM_MODULE_FILE = 'spam_transform.py'


# In[19]:


get_ipython().run_cell_magic('writefile', '{TRANSFORM_MODULE_FILE}', '\nimport tensorflow as tf\nLABEL_KEY = "Category"\nFEATURE_KEY = "Message"\ndef transformed_name(key):\n    """Renaming transformed features"""\n    return key + "_xf"\ndef preprocessing_fn(inputs):\n    """\n    Preprocess input features into transformed features\n    \n    Args:\n        inputs: map from feature keys to raw features.\n    \n    Return:\n        outputs: map from feature keys to transformed features.    \n    """\n    \n    outputs = {}\n    \n    outputs[transformed_name(FEATURE_KEY)] = tf.strings.lower(inputs[FEATURE_KEY])\n    \n    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)\n    \n    return outputs\n')


# In[20]:


transform = Transform (
    examples = example_gen.outputs['examples'],
    schema = schema_gen.outputs['schema'],
    module_file = os.path.abspath (TRANSFORM_MODULE_FILE)
)

interactive_context.run(transform)


# # Model Development

# In[21]:


TRAINER_MODULE_FILE = "spam_trainer.py"


# In[22]:


get_ipython().run_cell_magic('writefile', '{TRAINER_MODULE_FILE}', 'import tensorflow as tf\nimport tensorflow_transform as tft \nfrom tensorflow.keras import layers\nimport os  \nimport tensorflow_hub as hub\nfrom tfx.components.trainer.fn_args_utils import FnArgs\n \nLABEL_KEY = "Category"\nFEATURE_KEY = "Message"\n \ndef transformed_name(key):\n    """Renaming transformed features"""\n    return key + "_xf"\n \ndef gzip_reader_fn(filenames):\n    """Loads compressed data"""\n    return tf.data.TFRecordDataset(filenames, compression_type=\'GZIP\')\n \n \ndef input_fn(file_pattern, \n             tf_transform_output,\n             num_epochs,\n             batch_size=64)->tf.data.Dataset:\n    """Get post_tranform feature & create batches of data"""\n    \n    # Get post_transform feature spec\n    transform_feature_spec = (\n        tf_transform_output.transformed_feature_spec().copy())\n    \n    # create batches of data\n    dataset = tf.data.experimental.make_batched_features_dataset(\n        file_pattern=file_pattern,\n        batch_size=batch_size,\n        features=transform_feature_spec,\n        reader=gzip_reader_fn,\n        num_epochs=num_epochs,\n        label_key = transformed_name(LABEL_KEY))\n    return dataset\n \n# os.environ[\'TFHUB_CACHE_DIR\'] = \'/hub_chace\'\n# embed = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4")\n \n# Vocabulary size and number of words in a sequence.\nVOCAB_SIZE = 10000\nSEQUENCE_LENGTH = 100\n \nvectorize_layer = layers.TextVectorization(\n    standardize="lower_and_strip_punctuation",\n    max_tokens=VOCAB_SIZE,\n    output_mode=\'int\',\n    output_sequence_length=SEQUENCE_LENGTH)\n \n \nembedding_dim=16\ndef model_builder():\n    """Build machine learning model"""\n    inputs = tf.keras.Input(shape=(1,), name=transformed_name(FEATURE_KEY), dtype=tf.string)\n    reshaped_narrative = tf.reshape(inputs, [-1])\n    x = vectorize_layer(reshaped_narrative)\n    x = layers.Embedding(VOCAB_SIZE, embedding_dim, name="embedding")(x)\n    x = layers.GlobalAveragePooling1D()(x)\n    x = layers.Dense(64, activation=\'relu\')(x)\n    x = layers.Dense(32, activation="relu")(x)\n    outputs = layers.Dense(1, activation=\'sigmoid\')(x)\n    \n    \n    model = tf.keras.Model(inputs=inputs, outputs = outputs)\n    \n    model.compile(\n        loss = \'binary_crossentropy\',\n        optimizer=tf.keras.optimizers.Adam(0.01),\n        metrics=[tf.keras.metrics.BinaryAccuracy()]\n    \n    )\n    \n    # print(model)\n    model.summary()\n    return model \n \n \ndef _get_serve_tf_examples_fn(model, tf_transform_output):\n    \n    model.tft_layer = tf_transform_output.transform_features_layer()\n    \n    @tf.function\n    def serve_tf_examples_fn(serialized_tf_examples):\n        \n        feature_spec = tf_transform_output.raw_feature_spec()\n        \n        feature_spec.pop(LABEL_KEY)\n        \n        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)\n        \n        transformed_features = model.tft_layer(parsed_features)\n        \n        # get predictions using the transformed features\n        return model(transformed_features)\n        \n    return serve_tf_examples_fn\n    \ndef run_fn(fn_args: FnArgs) -> None:\n    \n    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), \'logs\')\n    \n    tensorboard_callback = tf.keras.callbacks.TensorBoard(\n        log_dir = log_dir, update_freq=\'batch\'\n    )\n    \n    es = tf.keras.callbacks.EarlyStopping(monitor=\'val_binary_accuracy\', mode=\'max\', verbose=1, patience=10)\n    mc = tf.keras.callbacks.ModelCheckpoint(fn_args.serving_model_dir, monitor=\'val_binary_accuracy\', mode=\'max\', verbose=1, save_best_only=True)\n    \n    \n    # Load the transform output\n    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)\n    \n    # Create batches of data\n    train_set = input_fn(fn_args.train_files, tf_transform_output, 10)\n    val_set = input_fn(fn_args.eval_files, tf_transform_output, 10)\n    vectorize_layer.adapt(\n        [j[0].numpy()[0] for j in [\n            i[0][transformed_name(FEATURE_KEY)]\n                for i in list(train_set)]])\n    \n    # Build the model\n    model = model_builder()\n    \n    \n    # Train the model\n    model.fit(x = train_set,\n            validation_data = val_set,\n            callbacks = [tensorboard_callback, es, mc],\n            steps_per_epoch = 1000, \n            validation_steps= 1000,\n            epochs=10)\n    signatures = {\n        \'serving_default\':\n        _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(\n                                    tf.TensorSpec(\n                                    shape=[None],\n                                    dtype=tf.string,\n                                    name=\'examples\'))\n    }\n    model.save(fn_args.serving_model_dir, save_format=\'tf\', signatures=signatures)\n')


# In[23]:


from tfx.proto import trainer_pb2
 
trainer  = Trainer(
    module_file=os.path.abspath(TRAINER_MODULE_FILE),
    examples = transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    schema=schema_gen.outputs['schema'],
    train_args=trainer_pb2.TrainArgs(splits=['train']),
    eval_args=trainer_pb2.EvalArgs(splits=['eval'])
)
interactive_context.run(trainer)


# # Tahapan Analisis dan Validasi Model

# ### Resolver

# In[24]:


from tfx.dsl.components.common.resolver import Resolver 
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import LatestBlessedModelStrategy 
from tfx.types import Channel 
from tfx.types.standard_artifacts import Model, ModelBlessing 
 
model_resolver = Resolver(
    strategy_class= LatestBlessedModelStrategy,
    model = Channel(type=Model),
    model_blessing = Channel(type=ModelBlessing)
).with_id('Latest_blessed_model_resolver')
 
interactive_context.run(model_resolver)


# ### Evaluator

# In[25]:


import tensorflow_model_analysis as tfma 
 
eval_config = tfma.EvalConfig(
    model_specs=[tfma.ModelSpec(label_key='Category')],
    slicing_specs=[tfma.SlicingSpec()],
    metrics_specs=[
        tfma.MetricsSpec(metrics=[
            
            tfma.MetricConfig(class_name='ExampleCount'),
            tfma.MetricConfig(class_name='AUC'),
            tfma.MetricConfig(class_name='FalsePositives'),
            tfma.MetricConfig(class_name='TruePositives'),
            tfma.MetricConfig(class_name='FalseNegatives'),
            tfma.MetricConfig(class_name='TrueNegatives'),
            tfma.MetricConfig(class_name='BinaryAccuracy',
                threshold=tfma.MetricThreshold(
                    value_threshold=tfma.GenericValueThreshold(
                        lower_bound={'value':0.5}),
                    change_threshold=tfma.GenericChangeThreshold(
                        direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                        absolute={'value':0.0001})
                    )
            )
        ])
    ]
 
)


# In[26]:


from tfx.components import Evaluator
evaluator = Evaluator(
    examples=example_gen.outputs['examples'],
    model=trainer.outputs['model'],
    baseline_model=model_resolver.outputs['model'],
    eval_config=eval_config)
 
interactive_context.run(evaluator)


# In[27]:


eval_result = evaluator.outputs['evaluation'].get()[0].uri
tfma_result = tfma.load_eval_result(eval_result)
tfma.view.render_slicing_metrics(tfma_result)
tfma.addons.fairness.view.widget_view.render_fairness_indicator(
    tfma_result
)


# In[32]:


evaluator


# In[34]:


eval_result, tfma_result


# In[55]:


tfma_result[0]


# # Model Deployment

# #### Pusher

# In[28]:


from tfx.components import Pusher 
from tfx.proto import pusher_pb2 
 
pusher = Pusher(
    model=trainer.outputs['model'],
    model_blessing=evaluator.outputs['blessing'],
    push_destination=pusher_pb2.PushDestination(
        filesystem=pusher_pb2.PushDestination.Filesystem(
            base_directory='serving_model_dir/spam-model'))
 
)
 
interactive_context.run(pusher)


# In[ ]:





# In[ ]:




