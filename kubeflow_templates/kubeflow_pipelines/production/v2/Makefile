
docker-build-push:
	aws ecr get-login-password --region $ACCOUNT_REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$ACCOUNT_REGION.amazonaws.com
	kfp component build src/ --component-filepattern kfp_task.py --push-image