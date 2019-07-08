## ai-platform
~~~bash
https://8081-dot-7931992-dot-devshell.appspot.com/tree
/datalab/training-data-analyst/courses/machine_learning/deepdive/08_image/labs/mnistmodel/trainer
~~~
- env
~~~bash
MODEL_TYPE = "dnn_dropout"  # "linear", "dnn", "dnn_dropout", or "cnn"
os.environ["MODEL_TYPE"] = MODEL_TYPE
~~~
- local train
~~~bash
rm -rf mnistmodel.tar.gz mnist_trained
gcloud ai-platform local train \
    --module-name=trainer.task \
    --package-path=${PWD}/mnistmodel/trainer \
    -- \
    --output_dir=${PWD}/mnist_trained \
    --train_steps=100 \
    --learning_rate=0.01 \
    --model=$MODEL_TYPE
~~~

- result
~~~bash
# model_fn -> CheckpointSaverHook -> Graph 
INFO:tensorflow:loss = 2.338836, step = 1
INFO:tensorflow:Loss for final step: 0.6199754.

INFO:tensorflow:Saving dict for global step 100: accuracy = 0.9229, global_step = 100, loss = 0.26806042
~~~

- submit train
~~~bash
OUTDIR=gs://${BUCKET}/mnist/trained_${MODEL_TYPE}
JOBNAME=mnist_${MODEL_TYPE}_$(date -u +%y%m%d_%H%M%S)
echo $OUTDIR $REGION $JOBNAME
gsutil -m rm -rf $OUTDIR
gcloud ai-platform jobs submit training $JOBNAME \
    --region=$REGION \
    --module-name=trainer.task \
    --package-path=${PWD}/mnistmodel/trainer \
    --job-dir=$OUTDIR \
    --staging-bucket=gs://$BUCKET \
    --scale-tier=BASIC_GPU \
    --runtime-version=$TFVERSION \
    -- \
    --output_dir=$OUTDIR \
    --train_steps=10000 --learning_rate=0.01 --train_batch_size=512 \
    --model=$MODEL_TYPE --batch_norm
~~~

- tensorboard
~~~bash
from google.datalab.ml import TensorBoard
TensorBoard().start("gs://{}/mnist/trained_{}".format(BUCKET, MODEL_TYPE))
~~~

- model deployment
~~~bash
MODEL_NAME="mnist"
MODEL_VERSION=${MODEL_TYPE}
MODEL_LOCATION=$(gsutil ls gs://${BUCKET}/mnist/trained_${MODEL_TYPE}/export/exporter | tail -1)
echo "Deleting and deploying $MODEL_NAME $MODEL_VERSION from $MODEL_LOCATION ... this will take a few minutes"
#gcloud ml-engine versions delete ${MODEL_VERSION} --model ${MODEL_NAME}
#gcloud ml-engine models delete ${MODEL_NAME}
gcloud ml-engine models create ${MODEL_NAME} --regions $REGION
gcloud ml-engine versions create ${MODEL_VERSION} --model ${MODEL_NAME} --origin ${MODEL_LOCATION} --runtime-version=$TFVERSION
~~~

- predict
이미지를 json으로 만들어서 예측할 이미지(입력값)으로 던진다. 
~~~bash
import json, codecs
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

HEIGHT = 28
WIDTH = 28

mnist = input_data.read_data_sets("mnist/data", one_hot = True, reshape = False)
IMGNO = 5 #CHANGE THIS to get different images
jsondata = {"image": mnist.test.images[IMGNO].reshape(HEIGHT, WIDTH).tolist()}
json.dump(jsondata, codecs.open("test.json", "w", encoding = "utf-8"))
plt.imshow(mnist.test.images[IMGNO].reshape(HEIGHT, WIDTH));
~~~
~~~bash
gcloud ml-engine predict \
    --model=mnist \
    --version=${MODEL_TYPE} \
    --json-instances=./test.json
~~~