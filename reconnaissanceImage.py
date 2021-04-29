# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 16:07:46 2021

@author: dorian, alexandre
"""
import boto3
import sagemaker
import sh as sh
from coverage import env, data

session = sagemaker.Session()
bucket = session.default_bucket()

%env bucket s3://$bucket


region_name = boto3.Session().region_name
algorithm = sagemaker.amazon.amazon_estimator.get_image_uri(region_name, "image-classification", "latest")

print("Using algorithm %s" % algorithm)

%%sh
wget http://data.mxnet.io/data/caltech-256/caltech-256-60-train.rec
wget http://data.mxnet.io/data/caltech-256/caltech-256-60-val.rec

session.upload_data(path='caltech-256-60-train.rec', bucket=bucket, key_prefix='train')
session.upload_data(path='caltech-256-60-val.rec',   bucket=bucket, key_prefix='validation')

s3_train      = 's3://{}/train/'.format(bucket)
s3_validation = 's3://{}/validation/'.format(bucket)
s3_output     = 's3://{}/output'.format(bucket)

%env s3_train      $s3_train
%env s3_validation $s3_validation
%env s3_output     $s3_output

 %%sh
aws s3 ls $s3_train
aws s3 ls $s3_validation

role = sagemaker.get_execution_role()

ic = sagemaker.estimator.Estimator(algorithm,
                                   role,
                                   train_instance_count=1,
                                   train_instance_type='ml.p3.16xlarge',
                                   output_path=s3_output,
                                   sagemaker_session=session)


ic.set_hyperparameters(num_layers=18,               # Train a Resnet-18 model
                       use_pretrained_model=1,      # Fine-tune on our dataset
                       image_shape="3,224,224",   # 3 channels (RGB), 224x224 pixels
                       num_classes=257,             # 256 classes + 1 clutter class
                       num_training_samples=15420,  # Number of training samples
                       mini_batch_size=128,
                       epochs=10,                  # Learn the training samples 10 times
                       learning_rate=0.01,
                       augmentation_type='crop_color_transform' # Add altered images
                       #precision_dtype='float16'    # Train at half-precision to save memory
                      )

train_data = sagemaker.session.s3_input(s3_train,
                                        distribution='FullyReplicated',
                                        content_type='application/x-recordio',
                                        s3_data_type='S3Prefix')

validation_data = sagemaker.session.s3_input(s3_validation,
                                             distribution='FullyReplicated',
                                             content_type='application/x-recordio',
                                             s3_data_type='S3Prefix')

data_channels = {'train': train_data, 'validation': validation_data}

ic.fit(inputs=data_channels, logs=True)

ic_predictor = ic.deploy(initial_instance_count=1,
                         instance_type='ml.c5.2xlarge')

 !wget -O /tmp/test.jpg http://www.vision.caltech.edu/Image_Datasets/Caltech256/images/022.buddha-101/022_0019.jpg
file_name = '/tmp/test.jpg'
# test image
from IPython.display import Image
Image(file_name)

object_categories = ['ak47', 'american-flag', 'backpack', 'baseball-bat', 'baseball-glove', 'basketball-hoop', 'bat', 'bathtub', 'bear', 'beer-mug', 'billiards', 'binoculars', 'birdbath', 'blimp', 'bonsai-101', 'boom-box', 'bowling-ball', 'bowling-pin', 'boxing-glove', 'brain-101', 'breadmaker', 'buddha-101', 'bulldozer', 'butterfly', 'cactus', 'cake', 'calculator', 'camel', 'cannon', 'canoe', 'car-tire', 'cartman', 'cd', 'centipede', 'cereal-box', 'chandelier-101', 'chess-board', 'chimp', 'chopsticks', 'cockroach', 'coffee-mug', 'coffin', 'coin', 'comet', 'computer-keyboard', 'computer-monitor', 'computer-mouse', 'conch', 'cormorant', 'covered-wagon', 'cowboy-hat', 'crab-101', 'desk-globe', 'diamond-ring', 'dice', 'dog', 'dolphin-101', 'doorknob', 'drinking-straw', 'duck', 'dumb-bell', 'eiffel-tower', 'electric-guitar-101', 'elephant-101', 'elk', 'ewer-101', 'eyeglasses', 'fern', 'fighter-jet', 'fire-extinguisher', 'fire-hydrant', 'fire-truck', 'fireworks', 'flashlight', 'floppy-disk', 'football-helmet', 'french-horn', 'fried-egg', 'frisbee', 'frog', 'frying-pan', 'galaxy', 'gas-pump', 'giraffe', 'goat', 'golden-gate-bridge', 'goldfish', 'golf-ball', 'goose', 'gorilla', 'grand-piano-101', 'grapes', 'grasshopper', 'guitar-pick', 'hamburger', 'hammock', 'harmonica', 'harp', 'harpsichord', 'hawksbill-101', 'head-phones', 'helicopter-101', 'hibiscus', 'homer-simpson', 'horse', 'horseshoe-crab', 'hot-air-balloon', 'hot-dog', 'hot-tub', 'hourglass', 'house-fly', 'human-skeleton', 'hummingbird', 'ibis-101', 'ice-cream-cone', 'iguana', 'ipod', 'iris', 'jesus-christ', 'joy-stick', 'kangaroo-101', 'kayak', 'ketch-101', 'killer-whale', 'knife', 'ladder', 'laptop-101', 'lathe', 'leopards-101', 'license-plate', 'lightbulb', 'light-house', 'lightning', 'llama-101', 'mailbox', 'mandolin', 'mars', 'mattress', 'megaphone', 'menorah-101', 'microscope', 'microwave', 'minaret', 'minotaur', 'motorbikes-101', 'mountain-bike', 'mushroom', 'mussels', 'necktie', 'octopus', 'ostrich', 'owl', 'palm-pilot', 'palm-tree', 'paperclip', 'paper-shredder', 'pci-card', 'penguin', 'people', 'pez-dispenser', 'photocopier', 'picnic-table', 'playing-card', 'porcupine', 'pram', 'praying-mantis', 'pyramid', 'raccoon', 'radio-telescope', 'rainbow', 'refrigerator', 'revolver-101', 'rifle', 'rotary-phone', 'roulette-wheel', 'saddle', 'saturn', 'school-bus', 'scorpion-101', 'screwdriver', 'segway', 'self-propelled-lawn-mower', 'sextant', 'sheet-music', 'skateboard', 'skunk', 'skyscraper', 'smokestack', 'snail', 'snake', 'sneaker', 'snowmobile', 'soccer-ball', 'socks', 'soda-can', 'spaghetti', 'speed-boat', 'spider', 'spoon', 'stained-glass', 'starfish-101', 'steering-wheel', 'stirrups', 'sunflower-101', 'superman', 'sushi', 'swan', 'swiss-army-knife', 'sword', 'syringe', 'tambourine', 'teapot', 'teddy-bear', 'teepee', 'telephone-box', 'tennis-ball', 'tennis-court', 'tennis-racket', 'theodolite', 'toaster', 'tomato', 'tombstone', 'top-hat', 'touring-bike', 'tower-pisa', 'traffic-light', 'treadmill', 'triceratops', 'tricycle', 'trilobite-101', 'tripod', 't-shirt', 'tuning-fork', 'tweezer', 'umbrella-101', 'unicorn', 'vcr', 'video-projector', 'washing-machine', 'watch-101', 'waterfall', 'watermelon', 'welding-mask', 'wheelbarrow', 'windmill', 'wine-bottle', 'xylophone', 'yarmulke', 'yo-yo', 'zebra', 'airplanes-101', 'car-side-101', 'faces-easy-101', 'greyhound', 'tennis-shoes', 'toad', 'clutter']

import json
import numpy as np

# Load test image from file
with open(file_name, 'rb') as f:
    payload = f.read()
    payload = bytearray(payload)

# Set content type
ic_predictor.content_type = 'application/x-image'

# Predict image and print JSON predicton
result = json.loads(ic_predictor.predict(payload))

# Print top class
index = np.argmax(result)
print("Result: label - " + object_categories[index] + ", probability - " + str(result[index]))

ic_predictor.delete_endpoint()



