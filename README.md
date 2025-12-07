# SparkLLM
Leveraging PySpark for generating embeddings is the definitive pattern for scaling AI Data Engineering.

use Spark's parallelism to apply the embedding model to millions or billions of text records efficiently.

Since PySpark cannot directly run models like a local Python script, need to set up a way for the Spark workers to communicate with an external Embedding Service (like a model hosted on AWS Bedrock/SageMaker or OpenAI).
