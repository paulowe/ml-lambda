# Deploying ML-models to AWS lambda

### Intro

In this project you will be using the [Serverless] Framework to deploy a Symptom classifier (trained using Sci-kit learn MLP Classifier) 

This setup uses AWS Lambda and Api-Gateway behind the scenes.

Api-Gateway will provide an API endpoint where Symptoms can be classified using the `POST` method with a JSON payload.

The payloads will be forwarded to an AWS lambda function that Loads the model and does the actual classification. 

### Quickstart
1. Install [Docker](https://www.docker.com/community-edition)
2. Install and setup the [AWS Command Line Interface](https://aws.amazon.com/cli/)
    - `pip install awscli`
    - `aws configure` (see [here](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html))
3. Clone this repo and change into the project `cd ml-lambda`
4. Start docker container: `./start_docker.sh` 


### Test lambda function locally

Run the following command with this payload to test the model. You can test your function locally using serverless (only works when you have a valid `serverless.yml`):

 ```
sls invoke local -f prediction_ep --data '{"body":"[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]"}'
 ```
 This tries to emulate the lambda runtime environment and runs the function `categorizer_lambda` with the provided `--data` as the `event`. For more info see [here](https://serverless.com/framework/docs/providers/aws/cli-reference/invoke-local/).
 
 
You can also test like this (no need for a valid `serverless.yml`):
```bash
python -c "import main; print(main.lambda_handler({"body":"[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]"}, None))"
```

----------------------------------------------------------------------------------------------------


## Things to keep in mind

All the ML frameworks in Python are quite heavy weight when it comes to size. However for individual lambda deployments 
(not as part of a cloudformation stack) there is a hard filesize limit of 50 MB enforced by AWS. 

There are a couple of tricks to reduce the size of your deployment zip-file. See [here](https://tech.europace.de/slimifying-aws-lambdas/) and the [resources section](https://github.com/bweigel/ml_at_awslambda_pydatabln2018#resources) for more info.

**Removing service stack?** 

To remove the service in AWS just run `serverless remove` inside the `container` directory.

**How to run `start_docker.sh` in Windows?**

Download Git for Windows (https://gitforwindows.org/) and install it. It comes with the Git BASH which you can use to run `start_docker.sh`.

*If you run into errors:* You might need to set some environmental variables before running the command. 
Do `export AWS_ACCESS_KEY_ID=<your aws access key id> AWS_SECRET_ACCESS_KEY=<your aws secret ccess key>` 

Run `./start_docker.sh` again.

**Why use the serverless framework (and not something like the AWS Serverless Application Model, SAM)?**

The serverless framework is the most mature tooling around for deploying serverless services. It is actively developed, 
has a big community and a rich ecosystem of plugins, which help with keeping our dependencies slim (see https://tech.europace.de/slimifying-aws-lambdas/) among other things.

**What else is inside the `event` that is provided inside the `lambda_handler` function?** 

Depending on the event type that triggers the function (e.g. SNS, SQS, Kinesis, Alexa...) the `event` payload will be different.
We are using ApiGateway Proxy integration in this application. 

## Troubleshooting

1. `make test*` fails with `FileNotFoundError`:
    ```bash
    $ make test-all
    set -eux pipefail
    if [ ! -d ".venv" ]; then \
            export PIPENV_IGNORE_VIRTUALENVS=1, PIPENV_VENV_IN_PROJECT=1 && pipenv lock && pipenv sync --dev; \
    fi
    pipenv run tox
    Traceback (most recent call last):
      File "/var/lang/bin/pipenv", line 11, in <module>
        sys.exit(cli())
      File "/var/lang/lib/python3.6/site-packages/pipenv/vendor/click/core.py", line 722, in __call__
        return self.main(*args, **kwargs)
      File "/var/lang/lib/python3.6/site-packages/pipenv/vendor/click/core.py", line 697, in main
        rv = self.invoke(ctx)
      File "/var/lang/lib/python3.6/site-packages/pipenv/vendor/click/core.py", line 1066, in invoke
        return _process_result(sub_ctx.command.invoke(sub_ctx))
      File "/var/lang/lib/python3.6/site-packages/pipenv/vendor/click/core.py", line 895, in invoke
        return ctx.invoke(self.callback, **ctx.params)
      File "/var/lang/lib/python3.6/site-packages/pipenv/vendor/click/core.py", line 535, in invoke
        return callback(*args, **kwargs)
      File "/var/lang/lib/python3.6/site-packages/pipenv/cli.py", line 637, in run
        do_run(command=command, args=args, three=three, python=python)
      File "/var/lang/lib/python3.6/site-packages/pipenv/core.py", line 2305, in do_run
        do_run_posix(script, command=command)
      File "/var/lang/lib/python3.6/site-packages/pipenv/core.py", line 2285, in do_run_posix
        os.execl(command_path, command_path, *script.args)
      File "/var/lang/lib/python3.6/os.py", line 527, in execl
        execv(file, args)
    FileNotFoundError: [Errno 2] No such file or directory
    ```
    - **Solution**: `make clean && make setup` then run `make test-all`
2. Error when running `serverless deploy`: 
    ```bash
    Serverless: Building custom docker image from Dockerfile...
    
    Error --------------------------------------------------
    
    Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?
    ...

    ```
    - **Solution**: You are running serverless deploy inside a docker container and the serverless-requirements-plugin
    tries to spawn another nested container to package dependecies.
    Run deploy outside a docker container, or set `custom.pythonRequirements.dockerizePip` to `false` in the `serverless.yml`.    

## Resources

- [Dockerhub](https://hub.docker.com/search?q=&type=image) for Docker images.

- [AWS Lambda Python magic. Tips for creating powerful Lambda functions.](https://blog.mapbox.com/aws-lambda-python-magic-e0f6a407ffc6)

- [Serverless](https://serverless.com/framework/)

For other questions outside of this README, ask me [Paul Owe](https://www.paulowe.com)
