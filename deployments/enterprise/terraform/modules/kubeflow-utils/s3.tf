resource "aws_s3_bucket" "interim" {
  bucket        = var.s3_bucket_ml_ops_interim
  force_destroy = true

  tags = {
    Name        = var.s3_bucket_tag_name
    Environment = var.environment
  }
}

resource "aws_s3_bucket" "model_store" {
  bucket        = var.s3_bucket_ml_ops_model_store
  force_destroy = false

  tags = {
    Name        = var.s3_bucket_tag_name
    Environment = var.environment
  }
}

resource "aws_s3_bucket" "artifact_store" {
  bucket_prefix = var.s3_bucket_kubeflow_artifact_store
  force_destroy = false

  tags = {
    Name        = var.s3_bucket_tag_name
    Environment = var.environment
  }
}
