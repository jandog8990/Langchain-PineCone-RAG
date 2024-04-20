# Machine Learning

## Installation

Install the LangChain CLI, OpenAI and Pinecone CLI if you haven't yet

```bash
pip install -U openai 
pip install -U pinecone-client 
pip install -U langchain-cli
pip install -U langchain-core
pip install -U langchain-cohere
pip install -U langchain-community

```

## Adding packages

```bash
# adding packages from 
# https://github.com/langchain-ai/langchain/tree/master/templates
langchain app add $PROJECT_NAME

# adding custom GitHub repo packages
langchain app add --repo $OWNER/$REPO
# or with whole git string (supports other git providers):
# langchain app add git+https://github.com/hwchase17/chain-of-verification

# with a custom api mount point (defaults to `/{package_name}`)
langchain app add $PROJECT_NAME --api_path=/my/custom/path/rag
```

Note: you remove packages by their api path

```bash
langchain app remove my/custom/path/rag
```

## Setup LangSmith (Optional)
LangSmith will help us trace, monitor and debug LangChain applications. 
LangSmith is currently in private beta, you can sign up [here](https://smith.langchain.com/). 
If you don't have access, you can skip this section

```bash
# install langsmith
pip install langsmith
```

```shell
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=<your-api-key>
export LANGCHAIN_PROJECT=<your-project>  # if not specified, defaults to "default"
```

## Update your local .env file
To manage Langchain, OpenAI and Pinecone objects you will need
to create environment variables in your repo .env file

```bash
OPENAI_API_KEY=<your-api-key>
PINE_CONE_API_KEY=<your-pinecone-api-key>
COHERE_API_KEY=<your-cohere-api-key>
INDEX_NAME=<your-pinecone-index-name>
```

## Upload data to Pinecone Serverless
You will need to go to https://pinecone.io and setup a project. In your
project you will setup a serverless index using my script below. This
script will also upload Wikipedia training vector data to your index.

```bash
cd app/
python ParquetUpload.py
```


## Launch LangServe

```bash
cd app/
poetry install 
langchain serve
```

## Running in Docker

This project folder includes a Dockerfile that allows you to easily build and host your LangServe app.

### Building the Image

To build the image, you simply:

```shell
docker build . -t my-langserve-app
```

If you tag your image with something other than `my-langserve-app`,
note it for use in the next step.

### Running the Image Locally

To run the image, you'll need to include any environment variables
necessary for your application.

In the below example, we inject the `OPENAI_API_KEY` environment
variable with the value set in my local environment
(`$OPENAI_API_KEY`)

We also expose port 8080 with the `-p 8080:8080` option.

```shell
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY -p 8080:8080 my-langserve-app
```
