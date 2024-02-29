resource "aws_cloudwatch_log_group" "kubeflow" {
  name              = "/ecs/${var.ecs_service_name}"
  retention_in_days = 120
  tags              = {
    Name = "${var.ecs_service_name}-log-group"
  }
}

resource "aws_cloudwatch_log_stream" "kubeflow" {
  name           = "${var.ecs_service_name}-log-stream"
  log_group_name = aws_cloudwatch_log_group.kubeflow.name
}
