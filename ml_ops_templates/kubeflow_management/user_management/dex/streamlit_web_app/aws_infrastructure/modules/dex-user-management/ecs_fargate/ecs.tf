resource "aws_ecs_cluster" "ecs_fargate" {
  name       = var.ecs_cluster_name
  depends_on = [
    aws_ecr_repository.kubeflow
  ]
}

resource "aws_ecs_task_definition" "kubeflow" {
  family                   = var.ecs_task_definition_name
  execution_role_arn       = aws_iam_role.ecs_task_execution_role.arn
  task_role_arn            = aws_iam_role.ecs_task_execution_role.arn
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.ecs_fargate_cpu
  memory                   = var.ecs_fargate_memory
  container_definitions    = jsonencode([
    {
      name : var.ecs_container_definitions_name,
      image : "${aws_ecr_repository.kubeflow.repository_url}:${var.ecs_cluster_container_image}",
      cpu : var.ecs_container_cpu,
      memory : var.ecs_container_memory,
      networkMode : "awsvpc",
      logConfiguration : {
        "logDriver" : "awslogs",
        "options" : {
          "awslogs-group" : "/ecs/${var.ecs_container_definitions_name}",
          "awslogs-region" : var.aws_region,
          "awslogs-stream-prefix" : "ecs"
        }
      },

      portMappings : [
        {
          "protocol" : "tcp",
          "containerPort" : var.ecs_fargate_port,
          "hostPort" : var.ecs_fargate_port
        }
      ],

      environment : [
        {
          "name": "AWS_ACCESS_KEY_ID"
          "value": var.aws_access_key_id
        },
        {
          "name": "AWS_SECRET_ACCESS_KEY"
          "value": var.aws_secret_access_key
        },
        {
          "name": "AWS_ACCOUNT_ID"
          "value": var.aws_account_id
        },
        {
          "name": "AWS_REGION"
          "value": var.aws_region
        },
        {
          "name": "CLUSTER_NAME"
          "value": var.eks_cluster_name
        },
        {
          "name": "ENVIRONMENT"
          "value": var.environment
        },
        {
          "name" : "GATEWAY_ADDRESS"
          "value" : "0.0.0.0:80"
        },
        {
          "name" : "CACHE_HOST"
          "value" : aws_elasticache_cluster.ecs_fargate.configuration_endpoint
        },
        {
          "name" : "CACHE_PORT"
          "value" : tostring(aws_elasticache_cluster.ecs_fargate.port)
        }
      ]
    },
  ])
}

resource "aws_ecs_service" "main" {
  name            = var.ecs_service_name
  cluster         = aws_ecs_cluster.ecs_fargate.id
  task_definition = aws_ecs_task_definition.kubeflow.arn
  desired_count   = var.ecs_container_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.private_subnets_id
    security_groups  = [var.ecs_task_id]
    assign_public_ip = false
  }

  load_balancer {
    container_name   = var.ecs_container_definitions_name
    container_port   = var.ecs_container_port
    target_group_arn = var.alb_target_group_arn
  }

  depends_on = [
    aws_iam_role_policy_attachment.ecs_task_execution_role
  ]
}

resource "aws_appautoscaling_target" "ecs_autoscaling_target" {
  max_capacity       = var.autoscaling_max_capacity
  min_capacity       = var.autoscaling_min_capacity
  resource_id        = "service/${aws_ecs_cluster.ecs_fargate.name}/${aws_ecs_service.main.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
  depends_on         = [
    aws_ecs_cluster.ecs_fargate,
    aws_ecs_service.main
  ]
}
