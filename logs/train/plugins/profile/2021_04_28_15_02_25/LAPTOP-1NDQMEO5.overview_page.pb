�	�<,�Z�@�<,�Z�@!�<,�Z�@	�)�y�?�)�y�?!�)�y�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�<,�Z�@���T���?A|a2U0��@Y9EGr�,@*	���̌L�@2|
EIterator::Model::MaxIntraOpParallelism::BatchV2::Shuffle::Zip[0]::MapQk�w�(@!9%G�	yU@)��JY��'@1��NpU@:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::BatchV2a2U0*�+@!E[�Q�X@)�;Nё\�?1���7B*@:Preprocessing2�
MIterator::Model::MaxIntraOpParallelism::BatchV2::Shuffle::Zip[1]::TensorSlice������?!�1�?�?)������?1�1�?�?:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::BatchV2::Shuffle::Zip�5�;N1(@!�N
̢U@)Ǻ���?1���Kx��?:Preprocessing2o
8Iterator::Model::MaxIntraOpParallelism::BatchV2::Shuffle�j+��=(@!��SɭU@)�~j�t��?1��{,���?:Preprocessing2�
RIterator::Model::MaxIntraOpParallelism::BatchV2::Shuffle::Zip[0]::Map::TensorSlice���<,�?!'?�z
�?)���<,�?1'?�z
�?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism�(���+@!�4.��X@)���_vO�?1�be?q�?:Preprocessing2F
Iterator::Model�ׁsF�+@!      Y@)lxz�,C|?1�[��F�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 1.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9�)�y�?IZÇ�X@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���T���?���T���?!���T���?      ��!       "      ��!       *      ��!       2	|a2U0��@|a2U0��@!|a2U0��@:      ��!       B      ��!       J	9EGr�,@9EGr�,@!9EGr�,@R      ��!       Z	9EGr�,@9EGr�,@!9EGr�,@b      ��!       JCPU_ONLYY�)�y�?b qZÇ�X@Y      Y@q�3m#=p@"�
device�Your program is NOT input-bound because only 1.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"CPU: B 