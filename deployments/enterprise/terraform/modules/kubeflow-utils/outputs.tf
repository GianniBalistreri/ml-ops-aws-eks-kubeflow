output "rds_secret" {
  value = aws_secretsmanager_secret.rds_secret.name
}

output "kubeflow_db_address" {
  value = aws_db_instance.kubeflow_db.address
}